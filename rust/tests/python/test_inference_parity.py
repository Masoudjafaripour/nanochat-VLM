"""
Numerical-equivalence tests: Rust (Candle) engine vs the PyTorch implementation.

Covers:
  * logits for text-only and image+text prompts (MHA and MQA models)
  * greedy decoding with KV cache (preallocated, concat, chunked prefill) vs engine.KVCache
  * text-only greedy vs the unmodified GPT.generate()
  * tokenizer: Rust BPE vs tiktoken on a freshly trained rustbpe vocabulary

Run from the repo root (after `maturin develop --release` in rust/bindings):
    python rust/tests/python/test_inference_parity.py
"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
RUST = os.path.dirname(os.path.dirname(HERE))
REPO = os.path.dirname(RUST)
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(RUST, "scripts"))

import logging  # noqa: E402
logging.getLogger("rustbpe").setLevel(logging.WARNING)
import nanochat_vlm_rs as rs  # noqa: E402
from export_checkpoint import random_gpt, save_safetensors, tokenizer_json  # noqa: E402
from nanochat_vlm.projector import VisionProjector  # noqa: E402
from vlm_reference import vlm_greedy, vlm_logits  # noqa: E402

SAMPLE = os.path.join(REPO, "test", "assets", "sample.png")
results = []


def record(name, ok, detail):
    results.append(ok)
    print(f"[{'OK ' if ok else 'FAIL'}] {name:<58} {detail}")


def build(out_dir, depth, n_kv_head=None, vocab=512, seed=0, patch=16):
    """Random GPT + projector, exported for Rust. Returns the PyTorch modules too."""
    torch.manual_seed(seed)
    gpt = random_gpt(depth, vocab, 256, n_kv_head, seed).eval()
    proj = VisionProjector(3 * patch * patch, gpt.config.n_embd).eval()
    with torch.no_grad():
        proj.norm.weight.uniform_(0.5, 1.5)
        proj.norm.bias.normal_(0, 0.1)
    tensors = {k: v.float() for k, v in gpt.state_dict().items()}
    tensors.update({f"projector.{k}": v.float() for k, v in proj.state_dict().items()})
    os.makedirs(out_dir, exist_ok=True)
    save_safetensors(tensors, os.path.join(out_dir, "model.safetensors"))
    c = gpt.config
    cfg = {
        "model": dict(sequence_len=c.sequence_len, vocab_size=c.vocab_size, n_layer=c.n_layer,
                      n_head=c.n_head, n_kv_head=c.n_kv_head, n_embd=c.n_embd),
        "projector": {"vision_dim": 3 * patch * patch, "lm_dim": c.n_embd},
        "preprocess": {"image_size": [224, 224], "patch_size": patch},
    }
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg, f)
    return gpt, proj


def rel_err(a, b):
    return float(np.abs(a - b).max() / (np.abs(b).max() + 1e-12))


def model_tests(tmp):
    patches = rs.preprocess(SAMPLE)  # (196, 768), already proven bit-exact vs torchvision
    rng = np.random.default_rng(0)
    for depth, kv in [(4, None), (4, 1), (6, None)]:
        name = f"d{depth}" + ("-MQA" if kv else "")
        d = os.path.join(tmp, name)
        gpt, proj = build(d, depth, kv)
        for attn in ["fused", "composed"]:
            run_model_checks(f"{name}/{attn}", gpt, proj, rs.Engine(d, attn=attn), patches, rng)


def run_model_checks(name, gpt, proj, eng, patches, rng):
    V = gpt.config.vocab_size
    text = rng.integers(0, V, 24).tolist()
    tail = rng.integers(0, V, 9).tolist()

    cases = {
        "text-only": ([text], [("tokens", text)]),
        "image+text": ([text[:3], patches, tail], [("tokens", text[:3]), ("image", torch.from_numpy(patches)), ("tokens", tail)]),
    }
    for cname, (rs_segs, py_segs) in cases.items():
        ref = vlm_logits(gpt, proj, py_segs).numpy()
        got = eng.logits(rs_segs)
        e = rel_err(got, ref)
        same_argmax = float((got.argmax(-1) == ref.argmax(-1)).mean())
        record(f"{name} logits {cname} (T={ref.shape[0]})", e < 1e-4 and same_argmax == 1.0,
               f"max rel err={e:.2e}  argmax agree={same_argmax:.3f}")

    # greedy decoding with KV cache
    segs_rs, segs_py = cases["image+text"]
    ref_tok = vlm_greedy(gpt, proj, segs_py, 24, use_cache=True)
    ref_nocache = vlm_greedy(gpt, proj, segs_py, 24, use_cache=False)
    record(f"{name} python cache vs no-cache (sanity)", ref_tok == ref_nocache, "")
    for strat, chunk in [("preallocated", None), ("concat", None), ("preallocated", 50)]:
        toks, stats = eng.generate(segs_rs, max_new_tokens=24, kv_strategy=strat, prefill_chunk=chunk)
        first_diff = next((i for i, (a, b) in enumerate(zip(toks, ref_tok)) if a != b), None)
        record(f"{name} greedy image+text {strat}{' chunk=50' if chunk else ''}", toks == ref_tok,
               f"24 tokens, first mismatch={first_diff}")

    # text-only greedy vs the original GPT.generate (no modification, no cache)
    py = list(gpt.generate(text, max_tokens=16, temperature=0.0))
    toks, _ = eng.generate([text], max_new_tokens=16)
    record(f"{name} greedy text-only vs GPT.generate()", toks == py, "")


def tokenizer_tests(tmp):
    from nanochat_vlm.tokenizer import RustBPETokenizer
    corpus = open(os.path.join(REPO, "README.md"), encoding="utf-8").read()
    for f in ["gpt.py", "engine.py", "tokenizer.py"]:
        corpus += open(os.path.join(REPO, "nanochat_vlm", f), encoding="utf-8").read()
    tok = RustBPETokenizer.train_from_iterator(iter([corpus] * 3), vocab_size=1024)
    path = os.path.join(tmp, "tokenizer.json")
    with open(path, "w") as f:
        json.dump(tokenizer_json(tok.enc), f)
    rt = rs.Tokenizer(path)
    samples = [corpus[:5000], "Hello world!! 123456 don't you're", "  spaces\n\n\ttabs\r\nCRLF",
               "unicode: naïve café 東京 🚀🚀 ✓", "", "x" * 300]
    ok = all(rt.encode(s) == tok.enc.encode_ordinary(s) for s in samples)
    record("tokenizer encode == tiktoken (rustbpe vocab 1024)", ok, f"{len(samples)} samples")
    ok = all(rt.decode(rt.encode(s)) == s for s in samples)
    record("tokenizer decode roundtrip", ok, "")
    record("tokenizer special ids", rt.special_id("<|assistant_end|>") == tok.encode_special("<|assistant_end|>"), "")


if __name__ == "__main__":
    torch.set_num_threads(os.cpu_count())
    with tempfile.TemporaryDirectory() as tmp:
        model_tests(tmp)
        tokenizer_tests(tmp)
    print("ALL PASSED" if all(results) else f"{results.count(False)} CHECK(S) FAILED")
    sys.exit(0 if all(results) else 1)
