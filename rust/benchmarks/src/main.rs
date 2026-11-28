//! `vlm-bench`: native benchmark driver. Prints one JSON object per run on stdout, so
//! `benchmarks/run_benchmarks.py` can compare it with the PyTorch baseline run in a
//! separate process (peak RSS is per process).

use candle_core::{DType, Device};
use clap::{Parser, Subcommand};
use rand::{Rng, SeedableRng};
use std::path::PathBuf;
use std::time::Instant;
use vlm_inference::{Engine, GenerateOptions, KvStrategy, Segment};
use vlm_preprocessing::{preprocess_batch, preprocess_path, PreprocessConfig};

#[derive(Parser)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Image preprocessing latency (single image, sequential) and parallel batch throughput.
    Preprocess {
        #[arg(long, num_args = 1..)]
        images: Vec<PathBuf>,
        #[arg(long, default_value_t = 50)]
        iters: usize,
    },
    /// TTFT / decode throughput / memory for one configuration.
    Generate {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        image: Option<PathBuf>,
        #[arg(long, default_value_t = 32)]
        prompt_tokens: usize,
        #[arg(long, default_value_t = 128)]
        new_tokens: usize,
        #[arg(long, default_value = "preallocated")]
        kv: String,
        #[arg(long, default_value_t = 3)]
        iters: usize,
        #[arg(long)]
        cuda: bool,
        /// fused (CPU default) | composed (plain candle ops)
        #[arg(long, default_value = "fused")]
        attn: String,
    },
    /// Per-op microbenchmark of the prefill hot path (matmul shapes of a d12 model).
    Ops {
        #[arg(long, default_value_t = 228)]
        t: usize,
        #[arg(long, default_value_t = 768)]
        c: usize,
    },
}

/// Peak resident set size of this process (Linux VmHWM), in MiB.
fn peak_rss_mib() -> f64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("VmHWM:"))
                .and_then(|l| l.split_whitespace().nth(1).and_then(|v| v.parse::<f64>().ok()))
        })
        .map(|kb| kb / 1024.0)
        .unwrap_or(f64::NAN)
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 2]
}

fn main() -> anyhow::Result<()> {
    match Cli::parse().cmd {
        Cmd::Preprocess { images, iters } => {
            let cfg = PreprocessConfig::default();
            for p in &images {
                preprocess_path(p, &cfg)?; // warm-up (page cache, allocator)
            }
            let mut lat = Vec::new();
            for _ in 0..iters {
                for p in &images {
                    let t = Instant::now();
                    std::hint::black_box(preprocess_path(p, &cfg)?);
                    lat.push(t.elapsed().as_secs_f64() * 1e3);
                }
            }
            let mut batch_ms = Vec::new();
            for _ in 0..iters.clamp(1, 10) {
                let t = Instant::now();
                std::hint::black_box(preprocess_batch(&images, &cfg)?);
                batch_ms.push(t.elapsed().as_secs_f64() * 1e3);
            }
            let b = median(batch_ms);
            println!(
                "{}",
                serde_json::json!({
                    "impl": "rust", "images": images.len(), "threads": rayon::current_num_threads(),
                    "single_ms_median": median(lat),
                    "batch_ms_median": b,
                    "batch_images_per_s": images.len() as f64 / (b / 1e3),
                })
            );
        }
        Cmd::Ops { t, c } => {
            use candle_core::{Module, Tensor};
            let dev = Device::Cpu;
            let x = Tensor::randn(0f32, 1.0, (1, t, c), &dev)?;
            let w = Tensor::randn(0f32, 1.0, (4 * c, c), &dev)?;
            let lin = candle_nn::Linear::new(w.clone(), None);
            let x2 = x.reshape((t, c))?;
            let wt = w.t()?.contiguous()?;
            let time = |name: &str, f: &dyn Fn() -> candle_core::Result<Tensor>| -> anyhow::Result<()> {
                f()?;
                let n = 20;
                let t0 = Instant::now();
                for _ in 0..n {
                    std::hint::black_box(f()?);
                }
                let ms = t0.elapsed().as_secs_f64() * 1e3 / n as f64;
                println!("{name:<44} {ms:8.3} ms");
                Ok(())
            };
            let gflop = 2.0 * t as f64 * c as f64 * 4.0 * c as f64 / 1e9;
            println!("matmul ({t}x{c}) @ ({c}x{}) = {gflop:.2} GFLOP", 4 * c);
            time("Linear::forward 3D (x @ W^T, W^T strided)", &|| lin.forward(&x))?;
            time("2D matmul x @ W^T (strided)", &|| x2.matmul(&w.t()?))?;
            time("2D matmul x @ Wt (contiguous)", &|| x2.matmul(&wt))?;
            time("broadcast_matmul 3D @ Wt (contiguous)", &|| x.broadcast_matmul(&wt))?;
            let h = Tensor::randn(0f32, 1.0, (1, 12, t, 64), &dev)?;
            time("attn q@k^T (1,12,T,64)", &|| h.matmul(&h.t()?))?;
            let hid = Tensor::randn(0f32, 1.0, (1, t, 4 * c), &dev)?;
            time("relu().sqr() on (1,T,4C)", &|| hid.relu()?.sqr())?;
            time("residual add (1,T,C)", &|| &x + &x)?;
            let ones = Tensor::ones(c, candle_core::DType::F32, &dev)?;
            time("fused rms_norm (1,T,C)", &|| candle_nn::ops::rms_norm(&x, &ones, 1e-7))?;
            let qh = Tensor::randn(0f32, 1.0, (1, t, 12, 64), &dev)?;
            time("transpose(1,2).contiguous (1,T,12,64)", &|| qh.transpose(1, 2)?.contiguous())?;
            let cs = Tensor::randn(0f32, 1.0, (t, 32), &dev)?;
            time("rope_thd (1,T,12,64)", &|| candle_nn::rotary_emb::rope_thd(&qh, &cs, &cs))?;
            time("rms_norm-like (sqr/mean/div)", &|| {
                let ms = x.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
                x.broadcast_div(&(ms + 1e-7)?.sqrt()?)
            })?;
        }
        Cmd::Generate { model, image, prompt_tokens, new_tokens, kv, iters, cuda, attn } => {
            let dev = if cuda { Device::new_cuda(0)? } else { Device::Cpu };
            let t = Instant::now();
            let mut engine = Engine::load(&model, &dev, DType::F32)?;
            engine.model.attn_impl = match attn.as_str() {
                "fused" => vlm_inference::AttnImpl::Fused,
                "composed" => vlm_inference::AttnImpl::Composed,
                s => anyhow::bail!("unknown attn impl {s}"),
            };
            let load_ms = t.elapsed().as_secs_f64() * 1e3;
            let vocab = engine.model.cfg.model.vocab_size as u32;
            let mut rng = rand::rngs::StdRng::seed_from_u64(0);
            let ids: Vec<u32> = (0..prompt_tokens).map(|_| rng.gen_range(0..vocab)).collect();
            let kv_strategy = match kv.as_str() {
                "preallocated" => KvStrategy::Preallocated,
                "concat" => KvStrategy::Concat,
                s => anyhow::bail!("unknown kv strategy {s}"),
            };
            let opts = GenerateOptions { max_new_tokens: new_tokens, kv_strategy, ..Default::default() };
            let mut runs = Vec::new();
            for i in 0..=iters {
                // Preprocessing is part of the request path, so TTFT here includes it.
                let t0 = Instant::now();
                let mut segs = vec![Segment::Tokens(ids[..1].to_vec())];
                if let Some(p) = &image {
                    segs.push(Segment::Image(preprocess_path(p, &engine.preprocess)?));
                }
                segs.push(Segment::Tokens(ids[1..].to_vec()));
                let pre_ms = t0.elapsed().as_secs_f64() * 1e3;
                let (toks, st) = engine.generate(&segs, &opts, |_| true)?;
                anyhow::ensure!(toks.len() == new_tokens);
                if i > 0 {
                    runs.push((pre_ms + st.ttft_ms, st.decode_tokens_per_s, st.prefill_ms, st.kv_cache_bytes, st.prompt_positions));
                } // iteration 0 is warm-up
            }
            println!(
                "{}",
                serde_json::json!({
                    "impl": format!("rust-candle-{}", if cuda { "cuda" } else { "cpu" }),
                    "kv": kv, "attn": attn, "prompt_positions": runs[0].4, "new_tokens": new_tokens,
                    "ttft_ms": median(runs.iter().map(|r| r.0).collect()),
                    "prefill_ms": median(runs.iter().map(|r| r.2).collect()),
                    "decode_tok_s": median(runs.iter().map(|r| r.1).collect()),
                    "kv_cache_mib": runs[0].3 as f64 / 1048576.0,
                    "load_ms": load_ms,
                    "peak_rss_mib": peak_rss_mib(),
                    "threads": rayon::current_num_threads(),
                })
            );
        }
    }
    Ok(())
}
