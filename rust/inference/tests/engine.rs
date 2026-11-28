//! Self-contained engine tests (no Python): a tiny random model is written to a temp dir.

use candle_core::{DType, Device, Tensor};
use std::collections::HashMap;
use std::path::PathBuf;
use vlm_inference::{AttnImpl, Engine, GenerateOptions, KvStrategy, SamplingParams, Segment};
use vlm_preprocessing::Patches;

const PATCH_DIM: usize = 3 * 4 * 4;

fn write_model(n_head: usize, n_kv_head: usize) -> PathBuf {
    // Unique per call: tests run in parallel and each removes its own directory.
    static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let id = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("vlm_engine_test_{}_{id}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let dev = Device::Cpu;
    let (c, v, l) = (64usize, 97usize, 2usize);
    let hd = c / n_head;
    let mut t: HashMap<String, Tensor> = HashMap::new();
    let mut add = |name: &str, shape: &[usize], std: f64| {
        t.insert(name.to_string(), Tensor::randn(0f32, std as f32, shape, &dev).unwrap());
    };
    add("transformer.wte.weight", &[v, c], 1.0);
    add("lm_head.weight", &[v, c], 1.0 / (c as f64).sqrt());
    for i in 0..l {
        let p = format!("transformer.h.{i}");
        let s = 1.0 / (c as f64).sqrt();
        add(&format!("{p}.attn.c_q.weight"), &[n_head * hd, c], s);
        add(&format!("{p}.attn.c_k.weight"), &[n_kv_head * hd, c], s);
        add(&format!("{p}.attn.c_v.weight"), &[n_kv_head * hd, c], s);
        add(&format!("{p}.attn.c_proj.weight"), &[c, c], s);
        add(&format!("{p}.mlp.c_fc.weight"), &[4 * c, c], s);
        add(&format!("{p}.mlp.c_proj.weight"), &[c, 4 * c], 1.0 / (4.0 * c as f64).sqrt());
    }
    add("projector.proj.weight", &[c, PATCH_DIM], 1.0 / (PATCH_DIM as f64).sqrt());
    add("projector.proj.bias", &[c], 0.1);
    add("projector.norm.weight", &[c], 1.0);
    add("projector.norm.bias", &[c], 0.1);
    candle_core::safetensors::save(&t, dir.join("model.safetensors")).unwrap();
    let cfg = serde_json::json!({
        "model": {"sequence_len": 64, "vocab_size": v, "n_layer": l, "n_head": n_head, "n_kv_head": n_kv_head, "n_embd": c},
        "projector": {"vision_dim": PATCH_DIM, "lm_dim": c},
        "preprocess": {"image_size": [8, 8], "patch_size": 4}
    });
    std::fs::write(dir.join("config.json"), cfg.to_string()).unwrap();
    dir
}

fn prompt() -> Vec<Segment> {
    let img = Patches { num_patches: 4, patch_dim: PATCH_DIM, data: (0..4 * PATCH_DIM).map(|i| (i % 17) as f32 / 17.0).collect() };
    vec![Segment::Tokens(vec![1, 5, 9]), Segment::Image(img), Segment::Tokens(vec![3, 3, 7, 11, 2])]
}

fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
    (a - b).unwrap().abs().unwrap().flatten_all().unwrap().max(0).unwrap().to_scalar().unwrap()
}

#[test]
fn fused_and_composed_agree_and_buffers_match_files() {
    for (h, kv) in [(4, 4), (4, 1), (4, 2)] {
        let dir = write_model(h, kv);
        let mut e = Engine::load(&dir, &Device::Cpu, DType::F32).unwrap();
        e.model.attn_impl = AttnImpl::Composed;
        let a = e.logits(&prompt()).unwrap();
        e.model.attn_impl = AttnImpl::Fused;
        let b = e.logits(&prompt()).unwrap();
        assert!(max_abs_diff(&a, &b) < 1e-4, "fused vs composed (h={h}, kv={kv}): {}", max_abs_diff(&a, &b));

        let cfg = std::fs::read_to_string(dir.join("config.json")).unwrap();
        let w = std::fs::read(dir.join("model.safetensors")).unwrap();
        let e2 = Engine::from_buffers(&cfg, w, None, &Device::Cpu, DType::F32).unwrap();
        assert_eq!(max_abs_diff(&b, &e2.logits(&prompt()).unwrap()), 0.0, "from_buffers == load");
        std::fs::remove_dir_all(&dir).ok();
    }
}

#[test]
fn cache_strategies_chunking_and_no_cache_agree() {
    let dir = write_model(4, 2);
    let e = Engine::load(&dir, &Device::Cpu, DType::F32).unwrap();
    let base = GenerateOptions { max_new_tokens: 20, ..Default::default() };
    let (ref_toks, _) = e.generate(&prompt(), &base, |_| true).unwrap();

    // No-cache greedy: recompute the full sequence each step from logits().
    let mut segs = prompt();
    let mut nocache = vec![];
    for _ in 0..20 {
        let l = e.logits(&segs).unwrap();
        let last: Vec<f32> = l.get(l.dim(0).unwrap() - 1).unwrap().to_vec1().unwrap();
        let t = vlm_inference::sampling::argmax(&last);
        nocache.push(t);
        segs.push(Segment::Tokens(vec![t]));
    }
    assert_eq!(ref_toks, nocache, "KV-cached decode == full recompute");

    for (kv, chunk) in [(KvStrategy::Concat, None), (KvStrategy::Preallocated, Some(1)), (KvStrategy::Preallocated, Some(5))] {
        let o = GenerateOptions { kv_strategy: kv, prefill_chunk: chunk, ..base.clone() };
        let (t, st) = e.generate(&prompt(), &o, |_| true).unwrap();
        assert_eq!(t, ref_toks, "{kv:?} chunk={chunk:?}");
        assert_eq!(st.prompt_positions, 12);
        assert_eq!(st.image_positions, 4);
    }
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn stop_tokens_callback_and_sampling() {
    let dir = write_model(4, 4);
    let e = Engine::load(&dir, &Device::Cpu, DType::F32).unwrap();
    let (greedy, _) = e.generate(&prompt(), &GenerateOptions { max_new_tokens: 10, ..Default::default() }, |_| true).unwrap();
    let stop = greedy[3];
    let first = greedy.iter().position(|&t| t == stop).unwrap();
    let o = GenerateOptions { max_new_tokens: 10, stop_tokens: vec![stop], ..Default::default() };
    let (t, _) = e.generate(&prompt(), &o, |_| true).unwrap();
    assert_eq!(t, greedy[..=first].to_vec(), "stops after the stop token");

    let mut n = 0;
    let (t, _) = e.generate(&prompt(), &GenerateOptions { max_new_tokens: 10, ..Default::default() }, |_| { n += 1; n < 4 }).unwrap();
    assert_eq!(t.len(), 4, "callback returning false stops generation");

    let s = SamplingParams { temperature: 0.9, top_k: Some(10), top_p: None, seed: 3 };
    let o = GenerateOptions { max_new_tokens: 15, sampling: s, ..Default::default() };
    let (a, _) = e.generate(&prompt(), &o, |_| true).unwrap();
    let (b, _) = e.generate(&prompt(), &o, |_| true).unwrap();
    assert_eq!(a, b, "seeded sampling is reproducible");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn rejects_prompts_past_rotary_cache() {
    let dir = write_model(4, 4);
    let e = Engine::load(&dir, &Device::Cpu, DType::F32).unwrap();
    // sequence_len 64 -> rotary cache of 640 positions
    let o = GenerateOptions { max_new_tokens: 700, ..Default::default() };
    assert!(e.generate(&prompt(), &o, |_| true).is_err());
    std::fs::remove_dir_all(&dir).ok();
}
