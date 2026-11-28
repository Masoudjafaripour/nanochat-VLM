# Benchmark results

AMD EPYC 7452 32-Core Processor (64 cores) · NVIDIA RTX A6000 · 2026-11-28 · threads=16 unless noted

Model for inference rows: random-init nanochat GPT d12 (n_embd 768, 12 layers, vocab 65,536, 186M params, fp32) + VisionProjector; prompt = 1 BOS + 196 image patches + 31 text tokens (228 positions) unless noted. Medians over 3 runs after warm-up.

## 1. Image preprocessing (decode + resize 224 + ToTensor + patchify 16)

32 images (PNG/JPEG, 320x240 to 1920x1080).

| implementation | single image, median ms | batch of all images, ms | images/s (batch) |
|---|---:|---:|---:|
| PIL + torchvision (thread pool for batch) | 29.02 | 121.6 | 263 |
| Rust native (rayon batch) | 9.22 | 53.9 | 594 |
| Rust via PyO3 from Python | 7.80 | 60.2 | 531 |

## 2. Multimodal data loading (512 samples, batch 32, 8 workers)

| implementation | samples/s |
|---|---:|
| torch DataLoader (PIL + torchvision + tiktoken, worker processes) | 203 |
| Rust DataLoader (threads, via PyO3) | 796 |

## 3. Inference: TTFT, decode throughput, memory

TTFT includes image preprocessing, projector and prefill. Decode tok/s excludes the first token.

| implementation | prompt positions | TTFT ms | decode tok/s | peak RSS MiB | notes |
|---|---:|---:|---:|---:|---|
| Rust Candle CPU (fused kernels) | 228 | 225 | 31.0 | 1,451 |  |
| Rust Candle CPU (plain candle ops) | 228 | 365 | 26.1 | 1,451 |  |
| PyTorch CPU (engine.KVCache) | 228 | 174 | 19.6 | 1,716 |  |
| PyTorch CUDA fp32 (reference) | 228 | 12 | 132.0 | 1,509 | different hardware; peak CUDA alloc 925 MiB |
| Rust Candle CPU (fused kernels) | 996 | 1,327 | 25.0 | 1,451 |  |
| Rust Candle CPU (plain candle ops) | 996 | 2,035 | 18.1 | 1,451 |  |
| PyTorch CPU (engine.KVCache) | 996 | 690 | 20.4 | 2,354 |  |
| PyTorch CUDA fp32 (reference) | 996 | 24 | 131.3 | 1,509 | different hardware; peak CUDA alloc 1552 MiB |

## 4. KV cache strategies (512 new tokens)

| implementation | decode tok/s | KV MiB allocated |
|---|---:|---:|
| Rust preallocated | 30.1 | 38.2 |
| Rust concat | 27.1 | 38.2 |
| PyTorch engine.KVCache | 20.2 | — |
| PyTorch no cache (128 tokens) | 11.4 | — |

## 5. Thread scaling (CPU)

| threads | Rust TTFT ms | Rust decode tok/s | PyTorch TTFT ms | PyTorch decode tok/s |
|---:|---:|---:|---:|---:|
| 4 | 368 | 30.8 | 390 | 22.6 |
| 8 | 308 | 30.3 | 235 | 22.1 |
| 16 | 226 | 30.3 | 166 | 19.8 |
| 32 | 203 | 29.8 | 129 | 14.7 |
| 64 | 248 | 21.7 | 2,927 | 9.0 |

## 6. Concurrent serving (Axum server, 32 new tokens per request, image + text)

| concurrent requests | aggregate tok/s | p50 request latency s | p50 TTFT ms |
|---:|---:|---:|---:|
| 1 | 21.0 | 1.52 | 231 |
| 2 | 33.0 | 1.94 | 327 |
| 4 | 57.5 | 2.20 | 374 |
| 8 | 70.6 | 3.23 | 606 |
