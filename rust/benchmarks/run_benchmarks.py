"""
End-to-end benchmark suite: Rust vs the existing PyTorch implementation.

Every inference configuration runs in a fresh process (peak RSS is per process).
Results go to rust/benchmarks/results/{results.json, results.md}.

    cd rust && cargo build --release -p vlm-benchmarks -p vlm-server
    (cd bindings && maturin develop --release)
    cd .. && python rust/benchmarks/run_benchmarks.py [--quick]
"""
import argparse
import base64
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
RUST = os.path.dirname(HERE)
REPO = os.path.dirname(RUST)
BIN = os.path.join(RUST, "target", "release")
SAMPLE = os.path.join(REPO, "test", "assets", "sample.png")
PY = sys.executable
OUT = os.path.join(HERE, "results")


def run_json(cmd, env=None):
    e = dict(os.environ, **(env or {}))
    p = subprocess.run(cmd, capture_output=True, text=True, env=e, cwd=REPO)
    if p.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed:\n{p.stderr[-2000:]}")
    return json.loads(p.stdout.strip().splitlines()[-1])


def rust_gen(model, threads, **kw):
    cmd = [os.path.join(BIN, "vlm-bench"), "generate", "--model", model]
    for k, v in kw.items():
        if v is True:
            cmd.append(f"--{k.replace('_', '-')}")
        elif v is not None:
            cmd += [f"--{k.replace('_', '-')}", str(v)]
    return run_json(cmd, {"RAYON_NUM_THREADS": str(threads)})


def py_gen(model, threads, **kw):
    cmd = [PY, os.path.join(HERE, "python_baseline.py"), "generate", "--model", model, "--threads", str(threads)]
    for k, v in kw.items():
        if v is not None:
            cmd += [f"--{k.replace('_', '-')}", str(v)]
    return run_json(cmd)


def make_images(d, n):
    """A mix of realistic sizes; smooth content so JPEG/PNG sizes are photo-like."""
    sizes = [(640, 480), (1280, 960), (1920, 1080), (800, 800), (320, 240)]
    paths = []
    yy, xx = np.mgrid[0:1080, 0:1920]
    for i in range(n):
        w, h = sizes[i % len(sizes)]
        base = (np.sin(xx[:h, :w] / (17 + i)) + np.cos(yy[:h, :w] / (23 + i))) * 60 + 128
        arr = np.stack([base, np.roll(base, i * 7, 1), 255 - base], -1).clip(0, 255).astype(np.uint8)
        ext = "png" if i % 2 == 0 else "jpg"
        p = os.path.join(d, f"img_{i:03d}.{ext}")
        Image.fromarray(arr).save(p, quality=90) if ext == "jpg" else Image.fromarray(arr).save(p)
        paths.append(p)
    return paths


def bench_preprocess(images, threads, iters):
    res = {}
    res["rust_native"] = run_json([os.path.join(BIN, "vlm-bench"), "preprocess", "--iters", str(iters), "--images", *images],
                                  {"RAYON_NUM_THREADS": str(threads)})
    res["python"] = run_json([PY, os.path.join(HERE, "python_baseline.py"), "preprocess", "--iters", str(iters),
                              "--images", *images])
    # Rust called from Python through the PyO3 binding (includes FFI + NumPy conversion)
    import nanochat_vlm_rs as rs
    for p in images:
        rs.preprocess(p)
    lat = []
    for _ in range(iters):
        for p in images:
            t = time.perf_counter()
            rs.preprocess(p)
            lat.append((time.perf_counter() - t) * 1e3)
    b = []
    for _ in range(min(10, iters)):
        t = time.perf_counter()
        rs.preprocess_batch(images)
        b.append((time.perf_counter() - t) * 1e3)
    bm = statistics.median(b)
    res["rust_pyo3"] = {"single_ms_median": statistics.median(lat), "batch_ms_median": bm,
                        "batch_images_per_s": len(images) / (bm / 1e3)}
    return res


def bench_dataloader(images, workers, n_samples, batch):
    """Rust DataLoader vs torch.utils.data.DataLoader doing the same work (PIL + tiktoken)."""
    import pickle
    import nanochat_vlm_rs as rs
    sys.path.insert(0, os.path.join(RUST, "scripts"))
    from export_checkpoint import tokenizer_json
    with tempfile.TemporaryDirectory() as d:
        man = os.path.join(d, "manifest.jsonl")
        with open(man, "w") as f:
            for i in range(n_samples):
                f.write(json.dumps({"image": images[i % len(images)], "text": f"sample {i}: " + "a caption about the image " * 4}) + "\n")
        enc = pickle.load(open(os.path.expanduser("~/.cache/nanochat/tokenizer/tokenizer.pkl"), "rb"))
        tok_path = os.path.join(d, "tok.json")
        json.dump(tokenizer_json(enc), open(tok_path, "w"))
        tok = rs.Tokenizer(tok_path)

        t = time.perf_counter()
        n = 0
        for b in rs.DataLoader(man, tok, batch_size=batch, num_workers=workers, shuffle=False, drop_last=False):
            n += b["patches"].shape[0]
        rust_sps = n / (time.perf_counter() - t)

        cmd = [PY, "-c", TORCH_LOADER, man, str(workers), str(batch)]
        torch_sps = float(subprocess.run(cmd, capture_output=True, text=True, cwd=REPO, check=True).stdout.strip())
    return {"samples": n_samples, "batch_size": batch, "workers": workers,
            "rust_samples_per_s": rust_sps, "torch_dataloader_samples_per_s": torch_sps}


TORCH_LOADER = r'''
import json, pickle, sys, time, os, torch
from torch.utils.data import Dataset, DataLoader
sys.path.insert(0, os.path.join(os.getcwd(), "rust", "benchmarks"))
from python_baseline import torch_preprocess
man, workers, batch = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
rows = [json.loads(l) for l in open(man)]
enc = pickle.load(open(os.path.expanduser("~/.cache/nanochat/tokenizer/tokenizer.pkl"), "rb"))
class DS(Dataset):
    def __len__(self): return len(rows)
    def __getitem__(self, i):
        r = rows[i]
        return torch_preprocess(r["image"]), torch.tensor(enc.encode_ordinary(r["text"])[:128])
def collate(b):
    ids = torch.nn.utils.rnn.pad_sequence([x[1] for x in b], batch_first=True)
    return torch.stack([x[0] for x in b]), ids
dl = DataLoader(DS(), batch_size=batch, num_workers=workers, collate_fn=collate)
t = time.perf_counter(); n = 0
for p, ids in dl: n += p.shape[0]
print(n / (time.perf_counter() - t))
'''


def bench_concurrency(model, threads, levels, new_tokens):
    img = base64.b64encode(open(SAMPLE, "rb").read()).decode()
    out = []
    for c in levels:
        port = 18200 + c
        srv = subprocess.Popen([os.path.join(BIN, "nanochat-vlm-server"), "--model", model, "--addr", f"127.0.0.1:{port}",
                                "--max-concurrency", str(c)], env=dict(os.environ, RAYON_NUM_THREADS=str(threads)),
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            for _ in range(100):
                try:
                    urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1)
                    break
                except Exception:
                    time.sleep(0.2)
            body = json.dumps({"prompt": "Describe the image in detail.", "image_base64": img,
                               "max_new_tokens": new_tokens, "raw": True}).encode()

            def call(results):
                t = time.perf_counter()
                req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/generate", body, {"content-type": "application/json"})
                r = json.loads(urllib.request.urlopen(req, timeout=600).read())
                results.append((time.perf_counter() - t, r["stats"]["generated_tokens"], r["stats"]["ttft_ms"]))

            call([])  # warm-up
            results = []
            rounds = 2
            t0 = time.perf_counter()
            for _ in range(rounds):
                ths = [threading.Thread(target=call, args=(results,)) for _ in range(c)]
                [th.start() for th in ths]
                [th.join() for th in ths]
            wall = time.perf_counter() - t0
            toks = sum(r[1] for r in results)
            out.append({"concurrency": c, "requests": len(results), "aggregate_tok_s": toks / wall,
                        "p50_latency_s": statistics.median(r[0] for r in results),
                        "p50_ttft_ms": statistics.median(r[2] for r in results)})
        finally:
            srv.terminate()
            srv.wait()
    return out


def fmt(v, nd=1):
    return "—" if v is None else (f"{v:,.{nd}f}" if isinstance(v, float) else str(v))


def write_md(R):
    L = []
    m = R["meta"]
    L.append(f"# Benchmark results\n\n{m['cpu']} · {m['gpu']} · {m['date']} · threads={m['threads']} unless noted\n")
    L.append("Model for inference rows: random-init nanochat GPT d12 (n_embd 768, 12 layers, vocab 65,536, "
             f"{m['params_m']:.0f}M params, fp32) + VisionProjector; prompt = 1 BOS + 196 image patches + "
             "31 text tokens (228 positions) unless noted. Medians over 3 runs after warm-up.\n")
    p = R["preprocess"]
    L.append("## 1. Image preprocessing (decode + resize 224 + ToTensor + patchify 16)\n")
    L.append(f"{p['python']['images']} images (PNG/JPEG, 320x240 to 1920x1080).\n")
    L.append("| implementation | single image, median ms | batch of all images, ms | images/s (batch) |\n|---|---:|---:|---:|")
    for k, name in [("python", "PIL + torchvision (thread pool for batch)"), ("rust_native", "Rust native (rayon batch)"),
                    ("rust_pyo3", "Rust via PyO3 from Python")]:
        r = p[k]
        L.append(f"| {name} | {fmt(r['single_ms_median'], 2)} | {fmt(r['batch_ms_median'], 1)} | {fmt(r['batch_images_per_s'], 0)} |")
    d = R["dataloader"]
    L.append(f"\n## 2. Multimodal data loading ({d['samples']} samples, batch {d['batch_size']}, {d['workers']} workers)\n")
    L.append("| implementation | samples/s |\n|---|---:|")
    L.append(f"| torch DataLoader (PIL + torchvision + tiktoken, worker processes) | {fmt(d['torch_dataloader_samples_per_s'], 0)} |")
    L.append(f"| Rust DataLoader (threads, via PyO3) | {fmt(d['rust_samples_per_s'], 0)} |")
    L.append("\n## 3. Inference: TTFT, decode throughput, memory\n")
    L.append("TTFT includes image preprocessing, projector and prefill. Decode tok/s excludes the first token.\n")
    L.append("| implementation | prompt positions | TTFT ms | decode tok/s | peak RSS MiB | notes |\n|---|---:|---:|---:|---:|---|")
    for r in R["inference"]:
        note = r.get("note", "")
        if "peak_cuda_mib" in r:
            note += f" peak CUDA alloc {r['peak_cuda_mib']:.0f} MiB"
        L.append(f"| {r['label']} | {r['prompt_positions']} | {fmt(r['ttft_ms'], 0)} | {fmt(r['decode_tok_s'], 1)} | "
                 f"{fmt(r['peak_rss_mib'], 0)} | {note.strip()} |")
    L.append(f"\n## 4. KV cache strategies ({R['kv'][0]['new_tokens']} new tokens)\n")
    L.append("| implementation | decode tok/s | KV MiB allocated |\n|---|---:|---:|")
    for r in R["kv"]:
        L.append(f"| {r['label']} | {fmt(r['decode_tok_s'], 1)} | {fmt(r.get('kv_cache_mib'), 1)} |")
    L.append("\n## 5. Thread scaling (CPU)\n")
    L.append("| threads | Rust TTFT ms | Rust decode tok/s | PyTorch TTFT ms | PyTorch decode tok/s |\n|---:|---:|---:|---:|---:|")
    for r in R["threads"]:
        L.append(f"| {r['threads']} | {fmt(r['rust_ttft_ms'], 0)} | {fmt(r['rust_decode'], 1)} | {fmt(r['torch_ttft_ms'], 0)} | {fmt(r['torch_decode'], 1)} |")
    L.append(f"\n## 6. Concurrent serving (Axum server, {R['concurrency'][0]['new_tokens']} new tokens per request, image + text)\n")
    L.append("| concurrent requests | aggregate tok/s | p50 request latency s | p50 TTFT ms |\n|---:|---:|---:|---:|")
    for r in R["concurrency"]:
        L.append(f"| {r['concurrency']} | {fmt(r['aggregate_tok_s'], 1)} | {fmt(r['p50_latency_s'], 2)} | {fmt(r['p50_ttft_ms'], 0)} |")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    T = a.threads
    iters = 3 if a.quick else 20
    os.makedirs(OUT, exist_ok=True)
    model = os.path.join(RUST, "artifacts", "bench_d12")
    if not os.path.exists(os.path.join(model, "model.safetensors")):
        subprocess.run([PY, os.path.join(RUST, "scripts", "export_checkpoint.py"), "--random", "--depth", "12",
                        "--vocab-size", "65536", "--seq-len", "2048", "--out", model], check=True, cwd=REPO)
    import torch
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no GPU"
    cpu = next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), platform.processor())
    R = {"meta": {"cpu": f"{cpu} ({os.cpu_count()} cores)", "gpu": gpu, "date": time.strftime("%Y-%m-%d"), "threads": T,
                  "params_m": 186.19}}

    print("[1/6] preprocessing", flush=True)
    with tempfile.TemporaryDirectory() as d:
        images = make_images(d, 32)
        R["preprocess"] = bench_preprocess(images, T, iters)
        print("[2/6] data loader", flush=True)
        R["dataloader"] = bench_dataloader(images, 8, 128 if a.quick else 512, 32)

    print("[3/6] inference", flush=True)
    inf = []
    for prompt_tokens, nt in [(32, 128), (800, 32)]:
        common = dict(image=SAMPLE, prompt_tokens=prompt_tokens, new_tokens=nt)
        r = rust_gen(model, T, attn="fused", **common); r["label"] = "Rust Candle CPU (fused kernels)"; inf.append(r)
        r = rust_gen(model, T, attn="composed", **common); r["label"] = "Rust Candle CPU (plain candle ops)"; inf.append(r)
        r = py_gen(model, T, kv="cache", **common); r["label"] = "PyTorch CPU (engine.KVCache)"; inf.append(r)
        if torch.cuda.is_available():
            r = py_gen(model, T, kv="cache", device="cuda", **common)
            r["label"] = "PyTorch CUDA fp32 (reference)"; r["note"] = "different hardware;"; inf.append(r)
    R["inference"] = inf

    print("[4/6] kv cache", flush=True)
    nt = 128 if a.quick else 512
    kv = []
    for strat in ["preallocated", "concat"]:
        r = rust_gen(model, T, kv=strat, prompt_tokens=32, new_tokens=nt, iters=2)
        r["label"] = f"Rust {strat}"; kv.append(r)
    r = py_gen(model, T, kv="cache", prompt_tokens=32, new_tokens=nt, iters=2); r["label"] = "PyTorch engine.KVCache"; kv.append(r)
    r = py_gen(model, T, kv="nocache", prompt_tokens=32, new_tokens=min(nt, 128), iters=1)
    r["label"] = f"PyTorch no cache ({min(nt, 128)} tokens)"; kv.append(r)
    for r in kv:
        r.setdefault("new_tokens", nt)
    R["kv"] = kv

    print("[5/6] threads", flush=True)
    th = []
    for n in [4, 8, 16, 32, 64]:
        rr = rust_gen(model, n, image=SAMPLE, new_tokens=32, iters=2)
        pp = py_gen(model, n, image=SAMPLE, new_tokens=32, iters=2)
        th.append({"threads": n, "rust_ttft_ms": rr["ttft_ms"], "rust_decode": rr["decode_tok_s"],
                   "torch_ttft_ms": pp["ttft_ms"], "torch_decode": pp["decode_tok_s"]})
    R["threads"] = th

    print("[6/6] concurrency", flush=True)
    conc = bench_concurrency(model, T, [1, 2, 4, 8], 32)
    for c in conc:
        c["new_tokens"] = 32
    R["concurrency"] = conc

    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(R, f, indent=2)
    md = write_md(R)
    with open(os.path.join(OUT, "results.md"), "w") as f:
        f.write(md)
    print(md)


if __name__ == "__main__":
    main()
