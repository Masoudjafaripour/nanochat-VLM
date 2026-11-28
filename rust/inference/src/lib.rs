//! nanochat-VLM inference engine (Candle).
//!
//! * [`config`]    exported `config.json` (GPTConfig + projector + preprocessing)
//! * [`model`]     GPT + VisionProjector, op-for-op with `nanochat_vlm/gpt.py`
//! * [`kv_cache`]  preallocated and concat KV caches
//! * [`sampling`]  greedy / temperature / top-k / top-p
//! * [`tokenizer`] tiktoken-compatible BPE from the rustbpe artifact
//! * [`attention`] fused CPU causal attention (reads the KV cache in place)
//! * [`engine`]    prompt assembly, prefill + decode loop, timing statistics

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

/// The MKL 2020 build shipped by `intel-mkl-src` lacks `hgemm_` (f16 GEMM), which candle
/// references. This engine never runs f16 matmuls on CPU, so a trapping stub is enough
/// to link. If it is ever reached, abort loudly instead of returning garbage.
#[cfg(feature = "mkl")]
#[no_mangle]
pub extern "C" fn hgemm_() {
    eprintln!("hgemm_ (f16 GEMM) is not available in this MKL build");
    std::process::abort();
}

pub mod attention;
pub mod config;
pub mod engine;
pub mod kernels;
pub mod kv_cache;
pub mod model;
pub mod sampling;
pub mod tokenizer;

pub use engine::{default_device, Engine, GenerateOptions, GenerationStats, Segment};
pub use kv_cache::{KvCache, KvStrategy};
pub use model::AttnImpl;
pub use sampling::SamplingParams;
