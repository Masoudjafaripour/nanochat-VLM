//! High-level generation API: load an exported checkpoint, build a multimodal prompt,
//! prefill, decode with a KV cache, and report timing/memory statistics.

use crate::config::ModelConfig;
use crate::kv_cache::{KvCache, KvStrategy};
use crate::model::Gpt;
use crate::sampling::{Sampler, SamplingParams};
use crate::tokenizer::Tokenizer;
use anyhow::Context;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use serde::Serialize;
use std::path::Path;
use web_time::Instant; // std::time on native, performance.now() on wasm
use vlm_preprocessing::{Patches, PreprocessConfig};

/// One piece of a multimodal prompt.
#[derive(Clone, Debug)]
pub enum Segment {
    Tokens(Vec<u32>),
    /// Preprocessed patches, (num_patches, patch_dim) row-major.
    Image(Patches),
}

#[derive(Clone, Debug)]
pub struct GenerateOptions {
    pub max_new_tokens: usize,
    pub sampling: SamplingParams,
    /// Generation stops after emitting any of these ids (the id itself is returned).
    pub stop_tokens: Vec<u32>,
    pub kv_strategy: KvStrategy,
    /// Prefill in chunks of this many positions (None = whole prompt at once).
    /// Bounds the size of the (T x T) attention matrix for long prompts.
    pub prefill_chunk: Option<usize>,
}

impl Default for GenerateOptions {
    fn default() -> Self {
        Self {
            max_new_tokens: 64,
            sampling: SamplingParams::default(),
            stop_tokens: vec![],
            kv_strategy: KvStrategy::Preallocated,
            prefill_chunk: None,
        }
    }
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct GenerationStats {
    pub prompt_positions: usize,
    pub image_positions: usize,
    pub generated_tokens: usize,
    /// Projector time for all images in the prompt.
    pub vision_ms: f64,
    pub prefill_ms: f64,
    /// Time from start of request (after preprocessing) to the first sampled token.
    pub ttft_ms: f64,
    pub decode_ms: f64,
    /// Decode throughput, excluding the first token.
    pub decode_tokens_per_s: f64,
    pub kv_cache_bytes: usize,
}

pub struct Engine {
    pub model: Gpt,
    pub tokenizer: Option<Tokenizer>,
    pub preprocess: PreprocessConfig,
}

pub fn default_device() -> anyhow::Result<Device> {
    #[cfg(feature = "cuda")]
    {
        if let Ok(d) = Device::new_cuda(0) {
            return Ok(d);
        }
    }
    Ok(Device::Cpu)
}

impl Engine {
    /// Load `config.json`, `model.safetensors` and (if present) `tokenizer.json` from an
    /// export directory produced by `rust/scripts/export_checkpoint.py`.
    pub fn load(dir: impl AsRef<Path>, device: &Device, dtype: DType) -> anyhow::Result<Self> {
        let dir = dir.as_ref();
        let cfg = ModelConfig::from_file(dir.join("config.json")).context("reading config.json")?;
        let weights = dir.join("model.safetensors");
        // SAFETY: the file is memory-mapped read-only; it must not be modified while loaded.
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights], dtype, device)? };
        let preprocess = cfg.preprocess.to_config()?;
        let model = Gpt::load(cfg, vb).context("loading weights")?;
        let tok_path = dir.join("tokenizer.json");
        let tokenizer = if tok_path.exists() { Some(Tokenizer::from_file(tok_path)?) } else { None };
        if let Some(p) = &model.projector {
            anyhow::ensure!(
                p.vision_dim == preprocess.patch_dim(),
                "projector vision_dim {} != patch_dim {} from preprocess config",
                p.vision_dim,
                preprocess.patch_dim()
            );
        }
        Ok(Self { model, tokenizer, preprocess })
    }

    /// Load from in-memory buffers (used by the WASM build, where there is no filesystem).
    pub fn from_buffers(
        config_json: &str,
        safetensors: Vec<u8>,
        tokenizer_json: Option<&str>,
        device: &Device,
        dtype: DType,
    ) -> anyhow::Result<Self> {
        let cfg: ModelConfig = serde_json::from_str(config_json).context("parsing config.json")?;
        cfg.model.validate()?;
        let vb = VarBuilder::from_buffered_safetensors(safetensors, dtype, device)?;
        let preprocess = cfg.preprocess.to_config()?;
        let model = Gpt::load(cfg, vb).context("loading weights")?;
        let tokenizer = match tokenizer_json {
            Some(s) => Some(Tokenizer::from_json(s)?),
            None => None,
        };
        Ok(Self { model, tokenizer, preprocess })
    }

    pub fn tokenizer(&self) -> anyhow::Result<&Tokenizer> {
        self.tokenizer.as_ref().context("export has no tokenizer.json")
    }

    /// nanochat chat format with an optional image inside the user turn:
    /// `<|bos|><|user_start|>[IMAGE]{text}<|user_end|><|assistant_start|>`
    /// Stops on `<|assistant_end|>` or `<|bos|>` like engine.py.
    pub fn chat_prompt(&self, text: &str, image: Option<Patches>) -> anyhow::Result<(Vec<Segment>, Vec<u32>)> {
        let t = self.tokenizer()?;
        let sp = |n: &str| t.special_id(n).with_context(|| format!("tokenizer lacks {n}"));
        let mut segs = vec![Segment::Tokens(vec![t.bos_id(), sp("<|user_start|>")?])];
        if let Some(img) = image {
            segs.push(Segment::Image(img));
        }
        let mut tail = t.encode(text);
        tail.push(sp("<|user_end|>")?);
        tail.push(sp("<|assistant_start|>")?);
        segs.push(Segment::Tokens(tail));
        Ok((segs, vec![sp("<|assistant_end|>")?, t.bos_id()]))
    }

    /// Embed a prompt into (1, T, C); returns (embeddings, image positions, projector ms).
    pub fn embed_segments(&self, segs: &[Segment]) -> anyhow::Result<(Tensor, usize, f64)> {
        let dev = &self.model.device;
        let mut parts = Vec::with_capacity(segs.len());
        let mut img_pos = 0;
        let mut vision_ms = 0.0;
        for s in segs {
            match s {
                Segment::Tokens(ids) if ids.is_empty() => {}
                Segment::Tokens(ids) => {
                    let t = Tensor::new(ids.as_slice(), dev)?.unsqueeze(0)?;
                    parts.push(self.model.embed_tokens(&t)?);
                }
                Segment::Image(p) => {
                    let t0 = Instant::now();
                    let x = Tensor::from_slice(&p.data, (1, p.num_patches, p.patch_dim), dev)?;
                    parts.push(self.model.embed_image(&x)?);
                    vision_ms += t0.elapsed().as_secs_f64() * 1e3;
                    img_pos += p.num_patches;
                }
            }
        }
        anyhow::ensure!(!parts.is_empty(), "empty prompt");
        Ok((Tensor::cat(&parts, 1)?, img_pos, vision_ms))
    }

    pub fn new_cache(&self, strategy: KvStrategy, capacity: usize) -> KvCache {
        let g = &self.model.cfg.model;
        KvCache::new(strategy, g.n_layer, 1, g.n_kv_head, g.head_dim(), capacity, self.model.dtype, &self.model.device)
    }

    /// Generate tokens. `on_token` is called for every new token; return `false` to stop.
    pub fn generate(
        &self,
        segs: &[Segment],
        opts: &GenerateOptions,
        mut on_token: impl FnMut(u32) -> bool,
    ) -> anyhow::Result<(Vec<u32>, GenerationStats)> {
        let start = Instant::now();
        let mut stats = GenerationStats::default();
        let (emb, img_pos, vision_ms) = self.embed_segments(segs)?;
        let t = emb.dim(1)?;
        stats.prompt_positions = t;
        stats.image_positions = img_pos;
        stats.vision_ms = vision_ms;
        let max_pos = self.model.cfg.model.sequence_len * 10;
        anyhow::ensure!(t + opts.max_new_tokens <= max_pos, "prompt + max_new_tokens exceeds rotary cache ({max_pos})");

        let mut cache = self.new_cache(opts.kv_strategy, t + opts.max_new_tokens);
        let mut sampler = Sampler::new(opts.sampling.clone());

        // Prefill (optionally chunked); only the last position's logits are needed.
        let t_prefill = Instant::now();
        let chunk = opts.prefill_chunk.unwrap_or(t).max(1);
        let mut logits = None;
        let mut off = 0;
        while off < t {
            let n = chunk.min(t - off);
            logits = Some(self.model.forward_embeds(&emb.narrow(1, off, n)?, Some(&mut cache), true)?);
            off += n;
        }
        let mut last: Vec<f32> = logits.unwrap().flatten_all()?.to_vec1()?;
        stats.prefill_ms = t_prefill.elapsed().as_secs_f64() * 1e3;

        let mut out = Vec::with_capacity(opts.max_new_tokens);
        let t_decode = Instant::now();
        for step in 0..opts.max_new_tokens {
            let tok = sampler.sample(&last);
            if step == 0 {
                stats.ttft_ms = start.elapsed().as_secs_f64() * 1e3;
            }
            out.push(tok);
            let keep_going = on_token(tok);
            if !keep_going || opts.stop_tokens.contains(&tok) || step + 1 == opts.max_new_tokens {
                break;
            }
            let ids = Tensor::new(&[tok], &self.model.device)?.unsqueeze(0)?;
            last = self.model.forward(&ids, Some(&mut cache), true)?.flatten_all()?.to_vec1()?;
        }
        stats.generated_tokens = out.len();
        // The first token comes from the prefill logits, so decode covers tokens 2..n.
        stats.decode_ms = t_decode.elapsed().as_secs_f64() * 1e3;
        if out.len() > 1 {
            stats.decode_tokens_per_s = (out.len() - 1) as f64 / (stats.decode_ms / 1e3);
        }
        stats.kv_cache_bytes = cache.allocated_bytes();
        Ok((out, stats))
    }

    /// Full-sequence logits without a cache (used by parity tests): (T, vocab).
    pub fn logits(&self, segs: &[Segment]) -> anyhow::Result<Tensor> {
        let (emb, _, _) = self.embed_segments(segs)?;
        Ok(self.model.forward_embeds(&emb, None, false)?.squeeze(0)?)
    }
}
