# nanochat-VLM · Rust inference & systems

A Cargo workspace that complements the Python/PyTorch training code with fast preprocessing,
a streaming multimodal data pipeline, a Candle inference engine, and deployment targets
(CLI, REST API, browser). Nothing in `nanochat_vlm/`, `src/`, `rustbpe/` or the notebooks
was modified. Everything here only *reads* their artifacts or imports their modules.

```
rust/
├── Cargo.toml        workspace
├── preprocessing/    image decode, Pillow-exact resize, normalize, patchify      (vlm-preprocessing)
├── tokenizer/        BPE training (port of rustbpe) + tiktoken-compatible encode  (vlm-tokenizer)
├── data/             JSONL manifest → sharded, shuffled, prefetched batches        (vlm-data)
├── dataprep/         parquet text shards → pre-tokenized .bin shards for training (vlm-dataprep)
├── inference/        GPT + VisionProjector in Candle, KV cache, sampling, BPE      (vlm-inference)
├── bindings/         PyO3 module `nanochat_vlm_rs` (NumPy in/out, GIL released)    (vlm-bindings)
├── cli/              `nanochat-vlm` binary                                         (vlm-cli)
├── server/           `nanochat-vlm-server` Axum REST + SSE streaming               (vlm-server)
├── wasm/             wasm-bindgen build + Web Worker demo page                     (vlm-wasm)
├── benchmarks/       `vlm-bench` + PyTorch baseline + orchestrator + results/      (vlm-benchmarks)
├── tests/python/     numerical-equivalence tests against the PyTorch code
└── scripts/          export_checkpoint.py, vlm_reference.py
```

## Quick start

```bash
cd rust
cargo build --release                                  # CLI, server, bench (CPU)
cargo test --release --workspace --exclude vlm-bindings --exclude vlm-wasm
(cd bindings && maturin develop --release)             # Python module nanochat_vlm_rs

cd ..
# Export a checkpoint (trained, or random for testing) into the format Rust reads
python rust/scripts/export_checkpoint.py --checkpoint-dir ~/.cache/nanochat/base_checkpoints/d20 \
       --step <N> [--projector projector_state_dict.pt] --out rust/artifacts/d20
python rust/scripts/export_checkpoint.py --random --depth 8 --out rust/artifacts/random_d8

# CLI (streams tokens; --stats prints TTFT / tok/s / KV memory as JSON on stderr)
rust/target/release/nanochat-vlm generate --model rust/artifacts/random_d8 \
       --image test/assets/sample.png --prompt "Describe the image." --stats

# REST API
rust/target/release/nanochat-vlm-server --model rust/artifacts/random_d8 --max-concurrency 4
curl localhost:8080/v1/generate -H 'content-type: application/json' \
     -d "{\"prompt\":\"Describe it\",\"image_base64\":\"$(base64 -w0 test/assets/sample.png)\",\"stream\":true}"

# Browser demo (needs wasm-bindgen-cli matching Cargo.lock)
rust/wasm/build.sh rust/artifacts/random_d8 && python -m http.server -d rust/wasm/www 8000
```

Pretraining data prep (same file order, train/val split and BOS-per-document as `dataset.py` +
`dataloader.py`; token streams tested identical):

```bash
rust/target/release/nanochat-vlm prepare-data --tokenizer tokenizer.json --out ~/.cache/nanochat/tokens
# training side: rust/scripts/pretokenized_loader.py -> pretokenized_data_loader(dir, B, T, "train", rank, world_size)
```

From Python, the existing PyTorch code can use the Rust pieces directly:

```python
import nanochat_vlm_rs as rs, torch
patches = torch.from_numpy(rs.preprocess("test/assets/sample.png"))   # (196, 768), same values as torchvision
batch = rs.preprocess_batch(paths)                                    # (B, 196, 768), parallel
for b in rs.DataLoader("data/train.jsonl", rs.Tokenizer("tok.json"), batch_size=32, rank=r, world_size=w):
    images, ids = torch.from_numpy(b["patches"]), torch.from_numpy(b["input_ids"])
eng = rs.Engine("rust/artifacts/d20"); tokens, stats = eng.generate([[bos], patches.numpy(), ids])
```

## What the Rust engine implements (and how that was decided)

The repo does not have a trained VLM or a fused VLM module yet. It has the text GPT
(`gpt.py`), the `VisionProjector` (`projector.py`, Linear → LayerNorm), and raw 16×16
patches as vision features (`test/test_projector.py`). The notebooks use dummy encoders.
The Rust engine implements exactly that composition:

```
image → Pillow-exact resize 224 → ToTensor → 16×16 patches (196 × 768)
      → VisionProjector → spliced into the token embedding sequence where the image goes
      → norm → 12 × [attn(rotary, QK-norm, MQA) + MLP(ReLU²)] → norm → lm_head → softcap 15
```

`scripts/vlm_reference.py` defines the same composition in PyTorch, using only the
unmodified modules (`GPT`'s blocks, `VisionProjector`, `engine.KVCache`). Both sides are tested against it.
Prompts are a list of segments (`Tokens` / `Image`), so they don't depend on whether
the tokenizer has reserved `<|image|>`. The current `~/.cache/nanochat/tokenizer` predates that
change and does not have it. Once a real vision encoder (CLIP/SigLIP) is chosen in Python, it plugs
in ahead of the projector. That would be the one new model component to port.

### Candle vs ONNX Runtime → Candle

| | Candle | ONNX Runtime |
|---|---|---|
| Fits the custom architecture (QK-norm, ReLU², bf16 rotary tables, softcap, MQA) | written op for op, one file | needs `torch.onnx` export of every variant |
| Dynamic KV cache, chunked prefill, image splicing | ordinary Rust code | must be expressed as graph inputs/outputs; past-KV plumbing per layer |
| Loads the PyTorch state-dict names directly | yes (safetensors, same keys) | no, needs a re-export per change |
| WASM | same crate compiles to wasm32 | separate onnxruntime-web runtime |
| Peak CPU GEMM speed | slower than MKL (measured below) | uses MLAS, typically faster |

Candle was chosen because the model is small, changes often (it's a learning repo), and
the KV-cache, prefill and multimodal logic is where most of the work is. The cost is that GEMM-bound
prefill on CPU is slower than PyTorch (see the benchmarks). ONNX Runtime would make sense later
for a frozen, production checkpoint.

### Other decisions

* **Pillow-exact resize.** `preprocessing/src/resize.rs` ports Pillow's fixed-point
  separable resampler (bilinear and bicubic, 22-bit coefficients, uint8 rounding between passes).
  Any other resize implementation drifts by a few uint8 levels, which would make Rust-vs-Python
  equivalence untestable. The result is **bit-identical** patches for PNG input.
* **Export format.** `export_checkpoint.py` writes `config.json` + `model.safetensors` (same tensor names
  as the PyTorch state dict, projector under `projector.`) + `tokenizer.json`. The safetensors file is
  written by hand, because the `safetensors` package isn't in the venv.
* **Tokenizer.** `tokenizer/` ports rustbpe's training (merges tested identical to `rustbpe`,
  `nanochat-vlm train-tokenizer` / `rs.train_tokenizer`), leaving `rustbpe/` untouched.
* **Tokenizer reuse.** `rustbpe/` is a PyO3 extension module that can't be linked into a native binary,
  so it isn't touched. The *artifact* it produces (split regex + mergeable ranks, which Python wraps in
  `tiktoken.Encoding`) is exported to JSON, and `inference/src/tokenizer.rs` runs tiktoken's merge
  algorithm on it. The output is tested equal to `encode_ordinary` on a freshly trained rustbpe vocabulary.
* **Fused CPU kernels.** These were added after profiling, not up front. `vlm-bench ops` showed per-head attention
  GEMMs (3.7 ms for 0.08 GFLOP) and allocation-heavy element-wise chains dominating prefill.
  `attention.rs` is a rayon-parallel causal-attention kernel that reads K/V through their strides,
  so the KV cache is used in place and MQA needs no `repeat_interleave` copy. `kernels.rs` adds a
  row-parallel RMSNorm and a single-pass ReLU². Rotary uses candle's `rope_thd` with `-sin`, which is
  algebraically identical to `gpt.py`. The plain-candle path is kept (`AttnImpl::Composed`), used on GPU,
  and tested to the same tolerances.
* **KV cache.** It is preallocated per layer, `reserve()`d once per forward pass, and doubles when full. Writes are
  `slice_set` and reads are strided views. A `Concat` strategy is kept as the measured baseline.
* **Serving.** The engine is shared read-only (`Arc`) and each request gets its own KV cache on Tokio's
  blocking pool, with a semaphore for `--max-concurrency`. SSE streaming stops generation when the client
  disconnects. There is no cross-request batching (see limitations).
* **No new GPU code.** GPU inference uses candle's CUDA backend behind `--features cuda`. Training stays
  in PyTorch, and nothing PyTorch already does well on GPU was rewritten.

## Tests

| suite | what it checks | result |
|---|---|---|
| `cargo test` (18 tests) | resize invariants, patch layout, KV strategies, sampling, tokenizer, data-loader ordering/sharding/skips/shutdown, engine: fused == composed (MHA/MQA/GQA), `from_buffers` == file load, cached decode == full recompute, chunked prefill, stop tokens | all pass |
| `tests/python/test_preprocessing_parity.py` | Rust vs PIL+torchvision on the repo's `sample.png`, 36 resize cases (6 sizes × 3 targets × 2 filters), RGBA, grayscale, CLIP-normalize + bicubic + 14 px patches, batch and bytes APIs | **bit-exact (max diff 0)**; JPEG differs by ≤ 3/255 (different decoder, documented) |
| `tests/python/test_inference_parity.py` | logits vs PyTorch for text-only and image+text prompts on d4 MHA, d4 MQA and d6, for both attention paths; greedy decoding (24 tokens) with preallocated, concat and chunked-prefill caches vs `engine.KVCache`; text-only greedy vs the original `GPT.generate()`; tokenizer vs tiktoken | 45/45 pass, max relative logit error ≈ 1.5e-6, token-identical decoding |
| `tests/python/test_dataloader.py` | every batch equals torchvision + tiktoken, deterministic order across worker counts, seeds, DDP sharding, epochs, corrupt files | 6/6 pass, batches exactly equal |

```bash
python rust/tests/python/test_preprocessing_parity.py
python rust/tests/python/test_inference_parity.py
python rust/tests/python/test_dataloader.py
```

## Benchmarks

`python rust/benchmarks/run_benchmarks.py` (full tables in [benchmarks/results/results.md](benchmarks/results/results.md)).

Measured on an AMD EPYC 7452 (2 × 32 cores, Zen 2, no AVX-512) at 16 threads for both runtimes.
Model: random-init d12 (186M params, vocab 65,536, fp32) + projector. Prompt: image + text, 228 positions.

| metric | PyTorch | Rust | verdict |
|---|---:|---:|---|
| preprocess one image (ms, median) | 29.0 | 9.2 (7.8 via PyO3) | **Rust 3.1× faster** |
| preprocess 32 images in parallel (img/s) | 263 | 594 | **Rust 2.3×** |
| data loading (samples/s, 8 workers) | 203 (torch DataLoader) | 796 | **Rust 3.9×** |
| TTFT, 228 positions (ms) | **174** | 225 | **PyTorch 1.3× faster** |
| TTFT, 996 positions (ms) | **690** | 1,327 | **PyTorch 1.9× faster** (GEMM-bound prefill; MKL beats candle's `gemm`) |
| decode (tok/s) | 19.6 | **31.0** | **Rust 1.6×** (lower per-op overhead at batch 1) |
| peak RSS, 996 positions (MiB) | 2,354 | **1,451** | **Rust uses 38% less** |
| fused kernels vs plain candle | – | TTFT 365 → 225 ms, decode 26 → 31 tok/s | worth keeping |
| preallocated vs concat KV (512 tokens) | – | 30.1 vs 27.1 tok/s | ~11% gain |
| 64 threads (TTFT ms) | 2,927 | 248 | PyTorch collapses across NUMA sockets; Rust degrades mildly |
| server, 1 → 8 concurrent requests | – | 21 → 71 aggregate tok/s, p50 TTFT 231 → 606 ms | scales ~3.4× then saturates |
| PyTorch on RTX A6000 (reference) | TTFT 12 ms, 132 tok/s | – | the GPU wins outright; Rust targets CPU, edge and browser |

So Rust doesn't win automatically. It clearly wins on preprocessing, data loading, decode latency
and memory. PyTorch still wins on compute-bound prefill on CPU, and a GPU beats both.

## Compatibility and limitations

* **JPEG.** The `image` crate's decoder differs from libjpeg-turbo (used by Pillow) by up to 3/255 per pixel, so
  JPEG inputs are close but not bit-exact. PNG/BMP are exact.
* **Sampling.** Seeded sampling is reproducible within Rust but doesn't reproduce `torch.multinomial`'s
  random stream. Greedy decoding is token-identical to Python.
* **Precision.** Rust runs fp32 on CPU. `gpt.py` casts `wte` to bf16 on CUDA. The export stores fp32, so CUDA-trained
  bf16 embeddings are upcast exactly.
* **Engine scope.** It handles batch size 1 per request, with no continuous or cross-request batching, no quantization
  and no paged KV cache. Concurrency comes from running requests in parallel, which scales until the cores saturate (measured).
* **CUDA backend.** `--features cuda` is wired through, but this machine's `nvcc` is 12.0 and the candle CUDA build
  was **not built or benchmarked here**. GPU numbers in the table are PyTorch's, as a reference point.
* **MKL.** `--features mkl` builds (with a stub for the `hgemm_` symbol missing from the MKL 2020 build that
  `intel-mkl-src` ships), but on this AMD EPYC it was *slower* than candle's default `gemm`
  (prefill 622 ms vs 395 ms), so it's off by default.
* **WASM.** It builds (`wasm32-unknown-unknown`, SIMD128, 3.8 MB module) and the generated JS API matches the worker.
  It **was not executed** here (no browser or JS runtime on this machine). The load path it uses, `Engine::from_buffers`,
  is tested natively. It's single-threaded, the wasm32 address space caps fp32 models at roughly 1B parameters, and generation
  can't be interrupted mid-call.
* **Tool use.** The Python `Engine`'s calculator tool state machine (`<|python_start|>` …) isn't ported. It
  stops on `<|assistant_end|>` / `<|bos|>` like engine.py.
