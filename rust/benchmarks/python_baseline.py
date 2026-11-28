"""
PyTorch baseline worker for the benchmarks. Mirrors `vlm-bench` exactly and prints one JSON
object. Each configuration runs in its own process so peak RSS is comparable.

    python rust/benchmarks/python_baseline.py preprocess --images a.png b.png
    python rust/benchmarks/python_baseline.py generate --model rust/artifacts/bench_d12 --image x.png
"""
import argparse
import json
import os
import resource
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts"))


def peak_rss_mib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0  # Linux: KiB


def torch_preprocess(path):
    """test/test_projector.py pipeline."""
    from PIL import Image
    from torchvision import transforms
    img = Image.open(path).convert("RGB")
    x = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])(img)
    p = x.unfold(1, 16, 16).unfold(2, 16, 16)
    return p.permute(1, 2, 0, 3, 4).reshape(p.shape[1] * p.shape[2], -1)


def cmd_preprocess(a):
    for p in a.images:
        torch_preprocess(p)
    lat = []
    for _ in range(a.iters):
        for p in a.images:
            t = time.perf_counter()
            torch_preprocess(p)
            lat.append((time.perf_counter() - t) * 1e3)
    # "batch": what a Python loop would do (thread pool; PIL releases the GIL in decode/resize)
    batch = []
    with ThreadPoolExecutor(os.cpu_count()) as ex:
        for _ in range(min(10, a.iters)):
            t = time.perf_counter()
            list(ex.map(torch_preprocess, a.images))
            batch.append((time.perf_counter() - t) * 1e3)
    b = statistics.median(batch)
    print(json.dumps({"impl": "python-pil-torchvision", "images": len(a.images), "threads": torch.get_num_threads(),
                      "single_ms_median": statistics.median(lat), "batch_ms_median": b,
                      "batch_images_per_s": len(a.images) / (b / 1e3)}))


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


def cmd_generate(a):
    from export_checkpoint import random_gpt
    from nanochat_vlm.engine import KVCache
    from nanochat_vlm.projector import VisionProjector
    from vlm_reference import embed_segments, forward_embeds

    with open(os.path.join(a.model, "config.json")) as f:
        cfg = json.load(f)
    m = cfg["model"]
    dev = torch.device(a.device)
    t = time.perf_counter()
    gpt = random_gpt(m["n_layer"], m["vocab_size"], m["sequence_len"], m["n_kv_head"]).eval()
    proj = VisionProjector(cfg["projector"]["vision_dim"], m["n_embd"]).eval()
    gpt.to(dev)
    gpt.cos, gpt.sin = gpt.cos.to(dev), gpt.sin.to(dev)
    proj.to(dev)
    load_ms = (time.perf_counter() - t) * 1e3
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(0, m["vocab_size"], (a.prompt_tokens,), generator=g).tolist()

    runs = []
    with torch.inference_mode():
        for i in range(a.iters + 1):
            sync(dev)
            t0 = time.perf_counter()
            segs = [("tokens", ids[:1])]
            if a.image:
                segs.append(("image", torch_preprocess(a.image)))
            segs.append(("tokens", ids[1:]))
            emb = embed_segments(gpt, proj, segs)
            T = emb.size(1)
            if a.kv == "cache":  # engine.KVCache, the repo's own inference path
                cache = KVCache(batch_size=1, num_heads=m["n_kv_head"], seq_len=T + a.new_tokens,
                                head_dim=m["n_embd"] // m["n_head"], num_layers=m["n_layer"])
                logits = forward_embeds(gpt, emb, cache)
            else:
                cache = None
                logits = forward_embeds(gpt, emb)
            tok = int(logits[0, -1].argmax())
            sync(dev)
            ttft = (time.perf_counter() - t0) * 1e3
            t1 = time.perf_counter()
            for _ in range(a.new_tokens - 1):
                nxt = gpt.transformer.wte(torch.tensor([[tok]], device=dev))
                if cache is not None:
                    logits = forward_embeds(gpt, nxt, cache)
                else:
                    emb = torch.cat([emb, nxt], 1)
                    logits = forward_embeds(gpt, emb)
                tok = int(logits[0, -1].argmax())
            sync(dev)
            dec = time.perf_counter() - t1
            if i > 0:
                runs.append((ttft, (a.new_tokens - 1) / dec, T))
    out = {"impl": f"pytorch-{dev.type}", "kv": a.kv, "prompt_positions": runs[0][2], "new_tokens": a.new_tokens,
           "ttft_ms": statistics.median(r[0] for r in runs), "decode_tok_s": statistics.median(r[1] for r in runs),
           "load_ms": load_ms, "peak_rss_mib": peak_rss_mib(), "threads": torch.get_num_threads()}
    if dev.type == "cuda":
        out["peak_cuda_mib"] = torch.cuda.max_memory_allocated() / 1048576
    print(json.dumps(out))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("preprocess")
    p.add_argument("--images", nargs="+", required=True)
    p.add_argument("--iters", type=int, default=50)
    g = sub.add_parser("generate")
    g.add_argument("--model", required=True)
    g.add_argument("--image")
    g.add_argument("--prompt-tokens", type=int, default=32)
    g.add_argument("--new-tokens", type=int, default=128)
    g.add_argument("--kv", default="cache", choices=["cache", "nocache"])
    g.add_argument("--iters", type=int, default=3)
    g.add_argument("--device", default="cpu")
    g.add_argument("--threads", type=int, default=None)
    a = ap.parse_args()
    if getattr(a, "threads", None):
        torch.set_num_threads(a.threads)
    {"preprocess": cmd_preprocess, "generate": cmd_generate}[a.cmd](a)
