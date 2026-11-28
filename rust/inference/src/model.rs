//! nanochat GPT + VisionProjector in Candle.
//!
//! Op-for-op port of `nanochat_vlm/gpt.py` and `nanochat_vlm/projector.py`:
//! * token embedding -> parameter-free RMSNorm
//! * per block: x += attn(norm(x)); x += mlp(norm(x))
//! * attention: rotary (half-split, cos/sin rounded to bf16 like gpt.py) -> QK RMSNorm
//!   -> SDPA with scale 1/sqrt(D), MQA/GQA by repeat-interleaving KV heads
//! * MLP: relu(x)^2, no biases anywhere in the GPT
//! * final norm -> lm_head -> softcap * tanh(logits / softcap)
//! * projector: Linear(vision_dim, n_embd, bias) -> LayerNorm(eps 1e-5)
//!
//! Multimodal composition: projected patch embeddings are spliced into the token
//! embedding sequence wherever the prompt has an image, *before* the post-embedding
//! RMSNorm, so image positions go through exactly the same path as `wte` output.
//!
//! Two implementations of the attention block share the weights:
//! * [`AttnImpl::Composed`]: plain candle ops (rotary by narrow/cat, RMSNorm by
//!   sqr/mean/div, explicit mask + softmax + matmuls). Works on any device; the reference.
//! * [`AttnImpl::Fused`] (CPU default): candle's `rope_thd`, the parallel RMSNorm and ReLU²
//!   kernels in `kernels.rs`, and the fused causal-attention kernel in `attention.rs`. Same math, far fewer passes over
//!   memory and allocations. Parity tests cover both.

use crate::attention::CausalAttention;
use crate::kernels::{ReluSquare, RmsNormNoWeight};
use crate::config::{GptConfig, ModelConfig, ProjectorConfig};
use crate::kv_cache::KvCache;
use candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_nn::{Embedding, LayerNorm, Linear, VarBuilder};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttnImpl {
    Composed,
    Fused,
}

/// `F.rms_norm(x, (x.size(-1),))` with eps = None -> torch.finfo(float32).eps.
const RMS_EPS: f32 = f32::EPSILON;

fn rms_norm_composed(x: &Tensor) -> Result<Tensor> {
    let x32 = x.to_dtype(DType::F32)?;
    let ms = x32.sqr()?.mean_keepdim(D::Minus1)?;
    x32.broadcast_div(&(ms + RMS_EPS as f64)?.sqrt()?)?.to_dtype(x.dtype())
}

/// Parameter-free RMSNorm with a cached all-ones weight for candle's fused kernel.
struct RmsNorm {
    ones_embd: Tensor,
    ones_head: Tensor,
}

impl RmsNorm {
    fn apply(&self, x: &Tensor, imp: AttnImpl) -> Result<Tensor> {
        match imp {
            AttnImpl::Composed => rms_norm_composed(x),
            AttnImpl::Fused if x.device().is_cpu() && x.dtype() == DType::F32 => {
                x.contiguous()?.apply_op1_no_bwd(&RmsNormNoWeight { eps: RMS_EPS })
            }
            AttnImpl::Fused => {
                let n = x.dim(D::Minus1)?;
                let w = if n == self.ones_embd.dim(0)? { &self.ones_embd } else { &self.ones_head };
                candle_nn::ops::rms_norm(&x.contiguous()?, w, RMS_EPS)
            }
        }
    }
}

struct Rotary {
    cos: Tensor, // (max_len, D/2)
    sin: Tensor,
    neg_sin: Tensor, // candle's rope rotates the other way; gpt.py == rope(x, cos, -sin)
}

impl Rotary {
    fn new(cfg: &ModelConfig, dtype: DType, dev: &Device) -> Result<Self> {
        let head_dim = cfg.model.head_dim();
        let max_len = cfg.model.sequence_len * 10; // gpt.py: rotary_seq_len = sequence_len * 10
        let half = head_dim / 2;
        // Same float32 math as gpt.py: inv_freq = 1 / base**(arange(0, D, 2) / D)
        let inv: Vec<f32> = (0..half)
            .map(|i| 1.0f32 / (cfg.rope_base as f32).powf((2 * i) as f32 / head_dim as f32))
            .collect();
        let inv = Tensor::from_vec(inv, (1, half), dev)?;
        let t = Tensor::arange(0u32, max_len as u32, dev)?.to_dtype(DType::F32)?.reshape((max_len, 1))?;
        let freqs = t.broadcast_mul(&inv)?;
        let (mut cos, mut sin) = (freqs.cos()?, freqs.sin()?);
        if cfg.rope_bf16 {
            cos = cos.to_dtype(DType::BF16)?;
            sin = sin.to_dtype(DType::BF16)?;
        }
        let (cos, sin) = (cos.to_dtype(dtype)?, sin.to_dtype(dtype)?);
        Ok(Self { neg_sin: sin.neg()?, cos, sin })
    }

    /// x: (B, T, H, D), positions [t0, t0+T).
    fn apply(&self, x: &Tensor, t0: usize, imp: AttnImpl) -> Result<Tensor> {
        let (_b, t, _h, d) = x.dims4()?;
        let half = d / 2;
        if t0 + t > self.cos.dim(0)? {
            candle_core::bail!("sequence length {} exceeds rotary cache {}", t0 + t, self.cos.dim(0)?);
        }
        let cos = self.cos.narrow(0, t0, t)?;
        if imp == AttnImpl::Fused {
            let nsin = self.neg_sin.narrow(0, t0, t)?;
            return candle_nn::rotary_emb::rope_thd(&x.contiguous()?, &cos.contiguous()?, &nsin.contiguous()?);
        }
        let cos = cos.reshape((1, t, 1, half))?;
        let sin = self.sin.narrow(0, t0, t)?.reshape((1, t, 1, half))?;
        let x1 = x.narrow(3, 0, half)?;
        let x2 = x.narrow(3, half, half)?;
        let y1 = (x1.broadcast_mul(&cos)? + x2.broadcast_mul(&sin)?)?;
        let y2 = (x2.broadcast_mul(&cos)? - x1.broadcast_mul(&sin)?)?;
        Tensor::cat(&[y1, y2], 3)
    }
}

struct Attention {
    c_q: Linear,
    c_k: Linear,
    c_v: Linear,
    c_proj: Linear,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
}

impl Attention {
    fn load(cfg: &GptConfig, vb: VarBuilder) -> Result<Self> {
        let hd = cfg.head_dim();
        let lin = |i, o, name| candle_nn::linear_no_bias(i, o, vb.pp(name));
        Ok(Self {
            c_q: lin(cfg.n_embd, cfg.n_head * hd, "c_q")?,
            c_k: lin(cfg.n_embd, cfg.n_kv_head * hd, "c_k")?,
            c_v: lin(cfg.n_embd, cfg.n_kv_head * hd, "c_v")?,
            c_proj: lin(cfg.n_embd, cfg.n_embd, "c_proj")?,
            n_head: cfg.n_head,
            n_kv_head: cfg.n_kv_head,
            head_dim: hd,
        })
    }

    /// (B, H_kv, T, D) -> (B, H, T, D), same as torch.repeat_interleave(dim=1) / SDPA enable_gqa.
    fn repeat_kv(&self, x: Tensor) -> Result<Tensor> {
        let rep = self.n_head / self.n_kv_head;
        if rep == 1 {
            return Ok(x);
        }
        let (b, hkv, t, d) = x.dims4()?;
        x.unsqueeze(2)?.expand((b, hkv, rep, t, d))?.reshape((b, hkv * rep, t, d))
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        x: &Tensor,
        rot: &Rotary,
        norm: &RmsNorm,
        layer: usize,
        cache: Option<&mut KvCache>,
        t0: usize,
        imp: AttnImpl,
    ) -> Result<Tensor> {
        let (b, t, _c) = x.dims3()?;
        let q = self.c_q.forward(x)?.reshape((b, t, self.n_head, self.head_dim))?;
        let k = self.c_k.forward(x)?.reshape((b, t, self.n_kv_head, self.head_dim))?;
        let v = self.c_v.forward(x)?.reshape((b, t, self.n_kv_head, self.head_dim))?;
        let q = norm.apply(&rot.apply(&q, t0, imp)?, imp)?.transpose(1, 2)?.contiguous()?;
        let k = norm.apply(&rot.apply(&k, t0, imp)?, imp)?.transpose(1, 2)?.contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;

        let (k, v) = match cache {
            Some(c) => c.append(layer, &k, &v)?,
            None => (k, v),
        };
        let scale = 1.0 / (self.head_dim as f64).sqrt();

        let y = if imp == AttnImpl::Fused && x.device().is_cpu() && q.dtype() == DType::F32 {
            // K/V may be strided views into the cache buffer; the kernel reads them in place.
            q.apply_op3_no_bwd(&k, &v, &CausalAttention { scale: scale as f32 })?
        } else {
            let tk = k.dim(2)?;
            let k = self.repeat_kv(k)?.contiguous()?;
            let v = self.repeat_kv(v)?.contiguous()?;
            let mut att = (q.matmul(&k.t()?)? * scale)?; // (B, H, T, Tk)
            if t > 1 {
                // Query i sits at absolute position (tk - t + i) and may see keys <= that.
                // Covers engine.py's three cases: no cache, Tq == Tk, and chunked prefill.
                let prefix = tk - t;
                let mask: Vec<f32> = (0..t)
                    .flat_map(|i| (0..tk).map(move |j| if j <= prefix + i { 0.0 } else { f32::NEG_INFINITY }))
                    .collect();
                let mask = Tensor::from_vec(mask, (t, tk), x.device())?.to_dtype(att.dtype())?;
                att = att.broadcast_add(&mask)?;
            }
            candle_nn::ops::softmax_last_dim(&att)?.matmul(&v)?
        };
        let y = y.transpose(1, 2)?.reshape((b, t, self.n_head * self.head_dim))?;
        self.c_proj.forward(&y)
    }
}

struct Mlp {
    c_fc: Linear,
    c_proj: Linear,
}

impl Mlp {
    fn load(cfg: &GptConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            c_fc: candle_nn::linear_no_bias(cfg.n_embd, 4 * cfg.n_embd, vb.pp("c_fc"))?,
            c_proj: candle_nn::linear_no_bias(4 * cfg.n_embd, cfg.n_embd, vb.pp("c_proj"))?,
        })
    }

    fn forward(&self, x: &Tensor, imp: AttnImpl) -> Result<Tensor> {
        let h = self.c_fc.forward(x)?;
        let h = if imp == AttnImpl::Fused && h.device().is_cpu() && h.dtype() == DType::F32 {
            h.contiguous()?.apply_op1_no_bwd(&ReluSquare)?
        } else {
            h.relu()?.sqr()?
        };
        self.c_proj.forward(&h)
    }
}

struct Block {
    attn: Attention,
    mlp: Mlp,
}

pub struct VisionProjector {
    proj: Linear,
    norm: LayerNorm,
    pub vision_dim: usize,
}

impl VisionProjector {
    fn load(cfg: &ProjectorConfig, vb: VarBuilder) -> Result<Self> {
        let proj = candle_nn::linear(cfg.vision_dim, cfg.lm_dim, vb.pp("proj"))?;
        let w = vb.pp("norm").get(cfg.lm_dim, "weight")?;
        let b = vb.pp("norm").get(cfg.lm_dim, "bias")?;
        Ok(Self { proj, norm: LayerNorm::new(w, b, cfg.layer_norm_eps), vision_dim: cfg.vision_dim })
    }

    /// (B, N, vision_dim) -> (B, N, n_embd)
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.norm.forward(&self.proj.forward(x)?)
    }
}

pub struct Gpt {
    pub cfg: ModelConfig,
    wte: Embedding,
    blocks: Vec<Block>,
    lm_head: Linear,
    rotary: Rotary,
    norm: RmsNorm,
    pub projector: Option<VisionProjector>,
    pub device: Device,
    pub dtype: DType,
    pub attn_impl: AttnImpl,
}

impl Gpt {
    /// Expects tensor names exactly as in the PyTorch state dict
    /// (`transformer.wte.weight`, `transformer.h.{i}.attn.c_q.weight`, ..., `lm_head.weight`),
    /// plus `projector.proj.*` / `projector.norm.*` when the config has a projector.
    pub fn load(cfg: ModelConfig, vb: VarBuilder) -> Result<Self> {
        let g = &cfg.model;
        let wte = candle_nn::embedding(g.vocab_size, g.n_embd, vb.pp("transformer.wte"))?;
        let blocks = (0..g.n_layer)
            .map(|i| {
                let bvb = vb.pp(format!("transformer.h.{i}"));
                Ok(Block { attn: Attention::load(g, bvb.pp("attn"))?, mlp: Mlp::load(g, bvb.pp("mlp"))? })
            })
            .collect::<Result<Vec<_>>>()?;
        let lm_head = candle_nn::linear_no_bias(g.n_embd, g.vocab_size, vb.pp("lm_head"))?;
        let projector = match &cfg.projector {
            Some(p) => Some(VisionProjector::load(p, vb.pp("projector"))?),
            None => None,
        };
        let (dev, dtype) = (vb.device().clone(), vb.dtype());
        let rotary = Rotary::new(&cfg, dtype, &dev)?;
        let norm = RmsNorm {
            ones_embd: Tensor::ones(g.n_embd, dtype, &dev)?,
            ones_head: Tensor::ones(g.head_dim(), dtype, &dev)?,
        };
        let attn_impl = if dev.is_cpu() && dtype == DType::F32 { AttnImpl::Fused } else { AttnImpl::Composed };
        Ok(Self { wte, blocks, lm_head, rotary, norm, projector, device: dev, dtype, cfg, attn_impl })
    }

    /// Token ids (B, T) -> raw (un-normalized) embeddings (B, T, C).
    pub fn embed_tokens(&self, ids: &Tensor) -> Result<Tensor> {
        self.wte.forward(ids)
    }

    /// Patches (B, N, vision_dim) -> (B, N, C).
    pub fn embed_image(&self, patches: &Tensor) -> Result<Tensor> {
        match &self.projector {
            Some(p) => p.forward(&patches.to_dtype(self.dtype)?),
            None => candle_core::bail!("this checkpoint has no vision projector"),
        }
    }

    /// Run the transformer on raw embeddings (B, T, C).
    ///
    /// `last_only` computes logits only for the final position (what decoding needs),
    /// which skips a (T x vocab) matmul during prefill.
    pub fn forward_embeds(&self, x: &Tensor, mut cache: Option<&mut KvCache>, last_only: bool) -> Result<Tensor> {
        let t = x.dim(1)?;
        let t0 = cache.as_ref().map(|c| c.pos()).unwrap_or(0);
        if let Some(c) = cache.as_deref_mut() {
            c.reserve(t)?;
        }
        let imp = self.attn_impl;
        let n = &self.norm;
        let mut x = n.apply(x, imp)?;
        for (i, blk) in self.blocks.iter().enumerate() {
            let a = blk.attn.forward(&n.apply(&x, imp)?, &self.rotary, n, i, cache.as_deref_mut(), t0, imp)?;
            x = (x + a)?;
            let m = blk.mlp.forward(&n.apply(&x, imp)?, imp)?;
            x = (x + m)?;
        }
        let mut x = n.apply(&x, imp)?;
        if last_only && t > 1 {
            x = x.narrow(1, t - 1, 1)?;
        }
        let logits = self.lm_head.forward(&x)?.to_dtype(DType::F32)?;
        let cap = self.cfg.softcap;
        (logits / cap)?.tanh()? * cap
    }

    /// Text-only convenience: ids (B, T) -> logits.
    pub fn forward(&self, ids: &Tensor, cache: Option<&mut KvCache>, last_only: bool) -> Result<Tensor> {
        self.forward_embeds(&self.embed_tokens(ids)?, cache, last_only)
    }
}
