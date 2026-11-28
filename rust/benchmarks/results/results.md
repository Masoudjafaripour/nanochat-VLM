# Benchmark results

AMD EPYC 7452 32-Core Processor (64 cores) · NVIDIA RTX A6000 · 2026-11-28 · threads=16 unless noted

Model for inference rows: random-init nanochat GPT d12 (n_embd 768, 12 layers, vocab 65,536, 186M params, fp32) + VisionProjector; prompt = 1 BOS + 196 image patches + 31 text tokens (228 positions) unless noted. Medians over 3 runs after warm-up.

## 1. Image preprocessing (decode + resize 224 + ToTensor + patchify 16)

32 images (PNG/JPEG, 320x240 to 1920x1080).

| implementation | single image, median ms | batch of all images, ms | images/s (batch) |
|---|---:|---:|---:|
| PIL + torchvision (thread pool for batch) | 29.13 | 124.5 | 257 |
| Rust native (rayon batch) | 9.32 | 55.5 | 576 |
| Rust via PyO3 from Python | 7.84 | 66.0 | 485 |

## 2. Multimodal data loading (128 samples, batch 32, 8 workers)

| implementation | samples/s |
|---|---:|
| torch DataLoader (PIL + torchvision + tiktoken, worker processes) | 63 |
| Rust DataLoader (threads, via PyO3) | 716 |

## 3. Inference: TTFT, decode throughput, memory

TTFT includes image preprocessing, projector and prefill. Decode tok/s excludes the first token.

| implementation | prompt positions | TTFT ms | decode tok/s | peak RSS MiB | notes |
|---|---:|---:|---:|---:|---|
| Rust Candle CPU (fused kernels) | 228 | 226 | 30.2 | 1,451 |  |
| Rust Candle CPU (plain candle ops) | 228 | 341 | 28.3 | 1,451 |  |
| PyTorch CPU (engine.KVCache) | 228 | 178 | 20.4 | 1,713 |  |
| PyTorch CUDA fp32 (reference) | 228 | 12 | 130.9 | 1,510 | different hardware; peak CUDA alloc 925 MiB |
| Rust Candle CPU (fused kernels) | 996 | 1,334 | 25.4 | 1,451 |  |
| Rust Candle CPU (plain candle ops) | 996 | 2,091 | 18.0 | 1,451 |  |
| PyTorch CPU (engine.KVCache) | 996 | 687 | 20.9 | 2,354 |  |
| PyTorch CUDA fp32 (reference) | 996 | 24 | 131.4 | 1,511 | different hardware; peak CUDA alloc 1552 MiB |

## 4. KV cache strategies (128 new tokens)

| implementation | decode tok/s | KV MiB allocated |
|---|---:|---:|
| Rust preallocated | 28.5 | 11.2 |
| Rust concat | 28.3 | 11.2 |
| PyTorch engine.KVCache | 17.5 | — |
| PyTorch no cache (128 tokens) | 10.6 | — |

## 5. Thread scaling (CPU)

| threads | Rust TTFT ms | Rust decode tok/s | PyTorch TTFT ms | PyTorch decode tok/s |
|---:|---:|---:|---:|---:|
| 4 | 373 | 31.0 | 396 | 22.6 |
| 8 | 309 | 29.8 | 221 | 19.8 |
| 16 | 226 | 30.4 | 184 | 19.6 |
| 32 | 204 | 28.2 | 133 | 19.6 |
| 64 | 240 | 16.8 | 3,419 | 9.1 |

## 6. Concurrent serving (Axum server, 32 new tokens per request, image + text)

| concurrent requests | aggregate tok/s | p50 request latency s | p50 TTFT ms |
|---:|---:|---:|---:|
| 1 | 18.3 | 1.75 | 372 |
| 2 | 33.9 | 1.86 | 465 |
| 4 | 51.0 | 2.42 | 612 |
| 8 | 71.6 | 3.57 | 697 |
