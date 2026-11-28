//! `nanochat-vlm`: run an exported nanochat-VLM checkpoint from the command line.
//!
//! ```text
//! nanochat-vlm generate --model rust/artifacts/d20 --image cat.png --prompt "What is this?"
//! nanochat-vlm preprocess --image cat.png
//! nanochat-vlm info --model rust/artifacts/d20
//! ```

use anyhow::Context;
use candle_core::{DType, Device};
use clap::{Parser, Subcommand, ValueEnum};
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;
use vlm_inference::tokenizer::StreamDecoder;
use vlm_inference::{Engine, GenerateOptions, KvStrategy, SamplingParams, Segment};

#[derive(Parser)]
#[command(name = "nanochat-vlm", version, about = "nanochat-VLM inference in Rust")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Clone, Copy, ValueEnum)]
enum DeviceArg {
    Cpu,
    Cuda,
}

#[derive(Clone, Copy, ValueEnum)]
enum KvArg {
    Preallocated,
    Concat,
}

#[derive(Subcommand)]
enum Cmd {
    /// Generate a reply for an (optional) image and a text prompt.
    Generate {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        image: Option<PathBuf>,
        #[arg(long, short)]
        prompt: String,
        #[arg(long, default_value_t = 128)]
        max_new_tokens: usize,
        #[arg(long, default_value_t = 0.0)]
        temperature: f32,
        #[arg(long)]
        top_k: Option<usize>,
        #[arg(long)]
        top_p: Option<f32>,
        #[arg(long, default_value_t = 42)]
        seed: u64,
        /// Feed the prompt as plain text (no chat special tokens).
        #[arg(long)]
        raw: bool,
        #[arg(long, value_enum, default_value_t = DeviceArg::Cpu)]
        device: DeviceArg,
        #[arg(long, value_enum, default_value_t = KvArg::Preallocated)]
        kv_cache: KvArg,
        #[arg(long)]
        prefill_chunk: Option<usize>,
        /// Print generation statistics as JSON to stderr.
        #[arg(long)]
        stats: bool,
    },
    /// Run only the image preprocessing and report shape/timing.
    Preprocess {
        #[arg(long)]
        image: PathBuf,
        #[arg(long, default_value_t = 224)]
        size: usize,
        #[arg(long, default_value_t = 16)]
        patch_size: usize,
    },
    /// Train a BPE tokenizer on text files (one document per line) -> tokenizer.json.
    TrainTokenizer {
        #[arg(long, num_args = 1.., required = true)]
        input: Vec<PathBuf>,
        #[arg(long, default_value_t = 65536)]
        vocab_size: u32,
        #[arg(long)]
        out: PathBuf,
        /// Stop after this many characters (nanochat's tok_train.py uses ~2B).
        #[arg(long)]
        max_chars: Option<usize>,
    },
    /// Tokenize parquet text shards (FineWeb layout) into .bin token shards for training.
    PrepareData {
        /// Directory with *.parquet files (default: ~/.cache/nanochat/base_data). Last file = val.
        #[arg(long)]
        data_dir: Option<PathBuf>,
        /// tokenizer.json (from train-tokenizer or export_checkpoint.py)
        #[arg(long)]
        tokenizer: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value_t = 100_000_000)]
        shard_tokens: usize,
        /// Only use the first N train files (quick runs).
        #[arg(long)]
        max_train_files: Option<usize>,
    },
    /// Bits per byte on pre-tokenized shards (loss_eval.py), on the Rust engine.
    EvalBpb {
        #[arg(long)]
        model: PathBuf,
        /// Output dir of `prepare-data`
        #[arg(long)]
        data_dir: PathBuf,
        #[arg(long, default_value = "val")]
        split: String,
        #[arg(long, default_value_t = 8)]
        batch_size: usize,
        #[arg(long, default_value_t = 512)]
        seq_len: usize,
        #[arg(long, default_value_t = 10)]
        steps: usize,
    },
    /// CORE metric on an eval bundle (core.yaml, eval_meta_data.csv, eval_data/), like base_eval.py.
    EvalCore {
        #[arg(long)]
        model: PathBuf,
        /// default: ~/.cache/nanochat/eval_bundle
        #[arg(long)]
        bundle: Option<PathBuf>,
        #[arg(long)]
        max_per_task: Option<usize>,
        /// Print the full result as JSON on stdout.
        #[arg(long)]
        json: bool,
        /// Write every rendered prompt (JSONL) here, for debugging/parity checks.
        #[arg(long)]
        dump_prompts: Option<PathBuf>,
    },
    /// Print an export's configuration.
    Info {
        #[arg(long)]
        model: PathBuf,
    },
}

fn device(d: DeviceArg) -> anyhow::Result<Device> {
    match d {
        DeviceArg::Cpu => Ok(Device::Cpu),
        DeviceArg::Cuda => Device::new_cuda(0).context("CUDA device unavailable (build with --features cuda)"),
    }
}

fn main() -> anyhow::Result<()> {
    match Cli::parse().cmd {
        Cmd::Generate {
            model, image, prompt, max_new_tokens, temperature, top_k, top_p, seed, raw, device: d,
            kv_cache, prefill_chunk, stats,
        } => {
            let t0 = Instant::now();
            let engine = Engine::load(&model, &device(d)?, DType::F32)?;
            let load_ms = t0.elapsed().as_secs_f64() * 1e3;
            let tok = engine.tokenizer()?;
            let t1 = Instant::now();
            let patches = match &image {
                Some(p) => Some(vlm_preprocessing::preprocess_path(p, &engine.preprocess)?),
                None => None,
            };
            let preprocess_ms = t1.elapsed().as_secs_f64() * 1e3;
            let (segs, stop) = if raw {
                let mut segs = vec![Segment::Tokens(vec![tok.bos_id()])];
                if let Some(p) = patches {
                    segs.push(Segment::Image(p));
                }
                segs.push(Segment::Tokens(tok.encode(&prompt)));
                (segs, vec![tok.bos_id()])
            } else {
                engine.chat_prompt(&prompt, patches)?
            };
            let opts = GenerateOptions {
                max_new_tokens,
                sampling: SamplingParams { temperature, top_k, top_p, seed },
                stop_tokens: stop.clone(),
                kv_strategy: match kv_cache {
                    KvArg::Preallocated => KvStrategy::Preallocated,
                    KvArg::Concat => KvStrategy::Concat,
                },
                prefill_chunk,
            };
            let mut dec = StreamDecoder::default();
            let mut out = std::io::stdout().lock();
            let (_, st) = engine.generate(&segs, &opts, |id| {
                if !stop.contains(&id) {
                    let _ = write!(out, "{}", dec.push(tok, id));
                    let _ = out.flush();
                }
                true
            })?;
            writeln!(out, "{}", dec.finish())?;
            if stats {
                let mut v = serde_json::to_value(&st)?;
                v["load_ms"] = load_ms.into();
                v["preprocess_ms"] = preprocess_ms.into();
                eprintln!("{}", serde_json::to_string_pretty(&v)?);
            }
        }
        Cmd::Preprocess { image, size, patch_size } => {
            let cfg = vlm_preprocessing::PreprocessConfig {
                image_size: (size, size),
                patch_size,
                ..Default::default()
            };
            let t = Instant::now();
            let p = vlm_preprocessing::preprocess_path(&image, &cfg)?;
            let ms = t.elapsed().as_secs_f64() * 1e3;
            let mean = p.data.iter().map(|&v| v as f64).sum::<f64>() / p.data.len() as f64;
            println!("patches: ({}, {})  mean={mean:.6}  time={ms:.2} ms", p.num_patches, p.patch_dim);
        }
        Cmd::TrainTokenizer { input, vocab_size, out, max_chars } => {
            use std::io::BufRead;
            use vlm_inference::tokenizer::{Trainer, NANOCHAT_PATTERN, SPECIAL_TOKENS_VLM};
            let t0 = Instant::now();
            let n_special = SPECIAL_TOKENS_VLM.len() as u32;
            anyhow::ensure!(vocab_size >= 256 + n_special, "vocab_size too small");
            let mut trainer = Trainer::new(NANOCHAT_PATTERN)?;
            let (mut chars, mut batch) = (0usize, Vec::with_capacity(8192));
            'files: for path in &input {
                let f = std::io::BufReader::new(std::fs::File::open(path).with_context(|| path.display().to_string())?);
                for line in f.lines() {
                    let line = line?;
                    chars += line.len();
                    batch.push(line);
                    if batch.len() == 8192 {
                        trainer.feed(&batch);
                        batch.clear();
                    }
                    if max_chars.is_some_and(|m| chars >= m) {
                        break 'files;
                    }
                }
            }
            trainer.feed(&batch);
            let bpe = trainer.train(vocab_size - n_special);
            std::fs::write(&out, bpe.to_json(&SPECIAL_TOKENS_VLM))?;
            println!(
                "trained {} merges on {:.1} MB in {:.1} s -> {}",
                bpe.merges.len(), chars as f64 / 1e6, t0.elapsed().as_secs_f64(), out.display()
            );
        }
        Cmd::PrepareData { data_dir, tokenizer, out, shard_tokens, max_train_files } => {
            let dir = data_dir.unwrap_or_else(|| {
                PathBuf::from(std::env::var("HOME").unwrap_or_default()).join(".cache/nanochat/base_data")
            });
            let tok = vlm_inference::tokenizer::Tokenizer::from_file(&tokenizer)?;
            let files = vlm_dataprep::list_parquet_files(&dir)?;
            anyhow::ensure!(files.len() >= 2, "need at least 2 parquet files in {} (last one is val)", dir.display());
            let (train, val) = files.split_at(files.len() - 1);
            let train = &train[..max_train_files.unwrap_or(train.len()).min(train.len())];
            let t0 = Instant::now();
            let mut meta = serde_json::json!({"tokenizer": tokenizer.display().to_string(), "format": "NCVTOK01"});
            for (split, fs) in [("val", val), ("train", train)] {
                let st = vlm_dataprep::prepare_split(&tok, fs, &out, split, shard_tokens, |f, docs, toks| {
                    eprintln!("[{split}] {} -> {docs} docs, {toks} tokens ({:.0}s)", f.display(), t0.elapsed().as_secs_f64());
                })?;
                meta[split] = serde_json::json!({"files": st.files, "docs": st.docs, "tokens": st.tokens, "shards": st.shards});
            }
            std::fs::write(out.join("meta.json"), serde_json::to_string_pretty(&meta)?)?;
            println!("done in {:.1}s -> {}", t0.elapsed().as_secs_f64(), out.display());
        }
        Cmd::EvalBpb { model, data_dir, split, batch_size, seq_len, steps } => {
            let engine = Engine::load(&model, &Device::Cpu, DType::F32)?;
            let tok = engine.tokenizer()?;
            let mut shards: Vec<PathBuf> = std::fs::read_dir(&data_dir)?
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.file_name().is_some_and(|n| n.to_string_lossy().starts_with(&format!("{split}_")) && n.to_string_lossy().ends_with(".bin")))
                .collect();
            shards.sort();
            anyhow::ensure!(!shards.is_empty(), "no {split}_*.bin in {}", data_dir.display());
            let mut tokens = Vec::new();
            for s in &shards {
                tokens.extend(vlm_dataprep::read_shard(s)?);
            }
            let tb = vlm_eval::token_bytes(tok, engine.model.cfg.model.vocab_size);
            let t0 = Instant::now();
            let (bpb, nats, bytes) = vlm_eval::bpb::evaluate(&engine.model, &tokens, batch_size, seq_len, steps, &tb)?;
            println!("{}", serde_json::json!({"bpb": bpb, "nats": nats, "bytes": bytes, "seconds": t0.elapsed().as_secs_f64()}));
        }
        Cmd::EvalCore { model, bundle, max_per_task, json, dump_prompts } => {
            let engine = Engine::load(&model, &Device::Cpu, DType::F32)?;
            let bundle = bundle.unwrap_or_else(|| {
                PathBuf::from(std::env::var("HOME").unwrap_or_default()).join(".cache/nanochat/eval_bundle")
            });
            let mut dump = match &dump_prompts {
                Some(p) => Some(std::io::BufWriter::new(std::fs::File::create(p)?)),
                None => None,
            };
            let dump_ref = dump.as_mut().map(|w| w as &mut dyn std::io::Write);
            let r = vlm_eval::core::evaluate_bundle(&engine.model, engine.tokenizer()?, &bundle, max_per_task, dump_ref, |l, a, c| {
                eprintln!("{l:<35} accuracy {a:.4} | centered {c:.4}");
            })?;
            if json {
                println!("{}", serde_json::to_string(&r)?);
            } else {
                println!("CORE metric: {:.4}", r.core_metric);
            }
        }
        Cmd::Info { model } => {
            let cfg = vlm_inference::config::ModelConfig::from_file(model.join("config.json"))?;
            println!("{}", serde_json::to_string_pretty(&cfg)?);
        }
    }
    Ok(())
}
