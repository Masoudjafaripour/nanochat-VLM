"""
Export a nanochat-VLM checkpoint for the Rust engine.

Writes an export directory with:
    config.json        GPTConfig + projector dims + preprocessing settings
    model.safetensors  float32 weights, PyTorch state-dict names (projector under "projector.")
    tokenizer.json     split regex + mergeable ranks + special tokens (from tokenizer.pkl)

Usage (from the repo root):
    # a trained base checkpoint (+ optional trained projector state_dict)
    python rust/scripts/export_checkpoint.py --checkpoint-dir ~/.cache/nanochat/base_checkpoints/d20 \
        --step 21400 --projector projector.pt --out rust/artifacts/d20

    # a randomly initialized model, for tests and benchmarks
    python rust/scripts/export_checkpoint.py --random --depth 4 --out rust/artifacts/random_d4

The safetensors file is written by hand (8-byte header length + JSON header + raw bytes),
so the `safetensors` pip package is not required.
"""
import argparse
import base64
import json
import os
import pickle
import struct
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from nanochat_vlm.gpt import GPT, GPTConfig  # noqa: E402
from nanochat_vlm.projector import VisionProjector  # noqa: E402

_DTYPES = {torch.float32: "F32", torch.bfloat16: "BF16", torch.float16: "F16"}


def save_safetensors(tensors, path):
    header, offset, blobs = {}, 0, []
    for name in sorted(tensors):
        t = tensors[name].detach().contiguous().cpu()
        raw = t.view(torch.uint8).numpy().tobytes() if t.dtype == torch.bfloat16 else t.numpy().tobytes()
        header[name] = {"dtype": _DTYPES[t.dtype], "shape": list(t.shape), "data_offsets": [offset, offset + len(raw)]}
        offset += len(raw)
        blobs.append(raw)
    h = json.dumps(header, separators=(",", ":")).encode()
    h += b" " * ((8 - len(h) % 8) % 8)  # align data to 8 bytes
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)))
        f.write(h)
        for b in blobs:
            f.write(b)


def tokenizer_json(enc, bos="<|bos|>"):
    """tiktoken.Encoding -> the JSON the Rust tokenizer reads."""
    ranks = [[base64.b64encode(k).decode(), v] for k, v in sorted(enc._mergeable_ranks.items(), key=lambda kv: kv[1])]
    return {"pattern": enc._pat_str, "mergeable_ranks": ranks, "special_tokens": dict(enc._special_tokens), "bos_token": bos}


def random_gpt(depth, vocab_size, seq_len, n_kv_head=None, seed=0):
    """nanochat sizing rule (model_dim = depth * 64, head_dim 128), random normal weights.

    GPT.init_weights() zeroes lm_head and every c_proj, which would make all logits
    zero; for parity testing we want every weight to matter, so use N(0, 1/sqrt(fan_in)).
    """
    model_dim = depth * 64
    n_head = max(1, (model_dim + 127) // 128)
    cfg = GPTConfig(sequence_len=seq_len, vocab_size=vocab_size, n_layer=depth, n_head=n_head,
                    n_kv_head=n_kv_head or n_head, n_embd=model_dim)
    torch.manual_seed(seed)
    model = GPT(cfg)
    model.init_weights()
    with torch.no_grad():
        for name, p in model.named_parameters():
            p.normal_(0, 1.0 if "wte" in name else p.shape[-1] ** -0.5)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoint-dir")
    ap.add_argument("--step", type=int)
    ap.add_argument("--random", action="store_true")
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--n-kv-head", type=int, default=None)
    ap.add_argument("--vocab-size", type=int, default=None, help="random mode; defaults to the tokenizer's size")
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--projector", help="VisionProjector state_dict (.pt); random mode makes a random one")
    ap.add_argument("--no-projector", action="store_true")
    ap.add_argument("--image-size", type=int, nargs=2, default=[224, 224])
    ap.add_argument("--patch-size", type=int, default=16)
    ap.add_argument("--filter", default="bilinear", choices=["bilinear", "bicubic"])
    ap.add_argument("--tokenizer-dir", default=os.path.expanduser("~/.cache/nanochat/tokenizer"))
    args = ap.parse_args()

    enc = None
    pkl = os.path.join(args.tokenizer_dir, "tokenizer.pkl")
    if os.path.exists(pkl):
        with open(pkl, "rb") as f:
            enc = pickle.load(f)

    if args.random:
        vocab = args.vocab_size or (enc.n_vocab if enc is not None else 512)
        model = random_gpt(args.depth, vocab, args.seq_len, args.n_kv_head, args.seed)
        state = model.state_dict()
        gpt_cfg = model.config.__dict__
    else:
        from nanochat_vlm.checkpoint_manager import load_checkpoint
        assert args.checkpoint_dir and args.step is not None, "--checkpoint-dir and --step are required"
        state, _, meta = load_checkpoint(args.checkpoint_dir, args.step, "cpu")
        state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
        gpt_cfg = meta["model_config"]

    vision_dim = 3 * args.patch_size * args.patch_size
    tensors = {k: v.float() for k, v in state.items() if k in _gpt_keys(gpt_cfg)}
    projector_cfg = None
    if not args.no_projector:
        if args.projector:
            pstate = torch.load(args.projector, map_location="cpu")
        elif args.random:
            torch.manual_seed(args.seed + 1)
            proj = VisionProjector(vision_dim, gpt_cfg["n_embd"])
            with torch.no_grad():  # non-trivial LayerNorm affine params
                proj.norm.weight.uniform_(0.5, 1.5)
                proj.norm.bias.normal_(0, 0.1)
            pstate = proj.state_dict()
        else:
            pstate = None
        if pstate is not None:
            vision_dim = pstate["proj.weight"].shape[1]
            tensors.update({f"projector.{k}": v.float() for k, v in pstate.items()})
            projector_cfg = {"vision_dim": vision_dim, "lm_dim": gpt_cfg["n_embd"], "layer_norm_eps": 1e-5}

    missing = _gpt_keys(gpt_cfg) - set(tensors)
    assert not missing, f"missing tensors: {sorted(missing)[:5]}"

    os.makedirs(args.out, exist_ok=True)
    save_safetensors(tensors, os.path.join(args.out, "model.safetensors"))
    config = {
        "model": {k: gpt_cfg[k] for k in ["sequence_len", "vocab_size", "n_layer", "n_head", "n_kv_head", "n_embd"]},
        "projector": projector_cfg,
        "preprocess": {"image_size": args.image_size, "patch_size": args.patch_size, "filter": args.filter},
        "rope_base": 10000.0,
        "softcap": 15.0,
        "rope_bf16": True,
    }
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    if enc is not None:
        with open(os.path.join(args.out, "tokenizer.json"), "w") as f:
            json.dump(tokenizer_json(enc), f)
    n = sum(t.numel() for t in tensors.values())
    print(f"exported {len(tensors)} tensors ({n / 1e6:.2f}M params) to {args.out}"
          f" | projector={'yes' if projector_cfg else 'no'} | tokenizer={'yes' if enc else 'no'}")


def _gpt_keys(c):
    keys = {"transformer.wte.weight", "lm_head.weight"}
    for i in range(c["n_layer"]):
        for n in ["attn.c_q", "attn.c_k", "attn.c_v", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"]:
            keys.add(f"transformer.h.{i}.{n}.weight")
    return keys


if __name__ == "__main__":
    main()
