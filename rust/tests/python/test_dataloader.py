"""
Rust DataLoader correctness from Python: every batch must equal the PyTorch reference
pipeline (test/test_projector.py preprocessing + tiktoken encode_ordinary), with
deterministic order, disjoint DDP shards and corrupt files skipped.

    python rust/tests/python/test_dataloader.py
"""
import json
import os
import pickle
import sys
import tempfile

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
RUST = os.path.dirname(os.path.dirname(HERE))
REPO = os.path.dirname(RUST)
sys.path.insert(0, os.path.join(RUST, "scripts"))
sys.path.insert(0, os.path.join(RUST, "tests", "python"))

import nanochat_vlm_rs as rs  # noqa: E402
from export_checkpoint import tokenizer_json  # noqa: E402
from test_preprocessing_parity import reference_patches  # noqa: E402

ok_all = []


def check(name, ok, detail=""):
    ok_all.append(ok)
    print(f"[{'OK ' if ok else 'FAIL'}] {name:<52} {detail}")


def main():
    enc = pickle.load(open(os.path.expanduser("~/.cache/nanochat/tokenizer/tokenizer.pkl"), "rb"))
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as d:
        rows = []
        for i in range(37):
            p = os.path.join(d, f"im{i}.png")
            if i == 5:
                open(p, "wb").write(b"corrupt")
            else:
                h, w = int(rng.integers(20, 300)), int(rng.integers(20, 300))
                Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8)).save(p)
            rows.append({"image": os.path.basename(p), "text": f"caption {i} " + "naïve café " * (i % 5)})
        man = os.path.join(d, "m.jsonl")
        with open(man, "w") as f:
            f.write("\n".join(json.dumps(r) for r in rows) + "\n")
        tp = os.path.join(d, "tok.json")
        json.dump(tokenizer_json(enc), open(tp, "w"))
        tok = rs.Tokenizer(tp)

        def load(**kw):
            return list(rs.DataLoader(man, tok, batch_size=4, num_workers=kw.pop("workers", 6), drop_last=False, **kw))

        batches = load(shuffle=True, seed=1)
        seen, exact = [], True
        for b in batches:
            for k, idx in enumerate(b["indices"]):
                ref = reference_patches(os.path.join(d, rows[idx]["image"]))
                ids = enc.encode_ordinary(rows[idx]["text"])[:128]
                L = int(b["lengths"][k])
                exact &= np.array_equal(b["patches"][k], ref) and b["input_ids"][k, :L].tolist() == ids
                exact &= bool((b["input_ids"][k, L:] == 0).all())
            seen += b["indices"].tolist()
        check("batches == torchvision + tiktoken reference", exact, f"{len(seen)} samples")
        check("corrupt sample skipped, rest covered", sorted(seen) == [i for i in range(37) if i != 5])
        again = [i for b in load(shuffle=True, seed=1, workers=1) for i in b["indices"].tolist()]
        check("same seed -> same order (6 vs 1 workers)", again == seen)
        other = [i for b in load(shuffle=True, seed=2) for i in b["indices"].tolist()]
        check("different seed -> different order", other != seen)
        shards = [[i for b in load(shuffle=True, seed=1, rank=r, world_size=3) for i in b["indices"].tolist()] for r in range(3)]
        flat = sorted(sum(shards, []))
        check("3 DDP ranks: disjoint and complete", flat == sorted(seen) and len(set(flat)) == len(flat))
        two = list(rs.DataLoader(man, tok, batch_size=4, epochs=2, drop_last=True))
        check("2 epochs, drop_last", [b["epoch"] for b in two] == [0] * 9 + [1] * 9 and all(b["patches"].shape[0] == 4 for b in two),
              f"{len(two)} batches")
    print("ALL PASSED" if all(ok_all) else "SOME CHECKS FAILED")
    sys.exit(0 if all(ok_all) else 1)


if __name__ == "__main__":
    main()
