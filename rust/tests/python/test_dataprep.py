"""
`nanochat-vlm prepare-data` vs the Python pipeline (dataset.py order + tiktoken, BOS per doc):
token streams must be identical, shards must split cleanly, and the pretokenized loader must
yield the same (inputs, targets) as slicing the reference stream. Also times both.

    python rust/tests/python/test_dataprep.py
"""
import json
import os
import pickle
import subprocess
import sys
import tempfile
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
RUST = os.path.dirname(os.path.dirname(HERE))
REPO = os.path.dirname(RUST)
sys.path.insert(0, os.path.join(RUST, "scripts"))
from export_checkpoint import tokenizer_json  # noqa: E402
from pretokenized_loader import open_shard, pretokenized_data_loader  # noqa: E402

CLI = os.path.join(RUST, "target", "release", "nanochat-vlm")
ok_all = []


def check(name, ok, detail=""):
    ok_all.append(bool(ok))
    print(f"[{'OK ' if ok else 'FAIL'}] {name:<50} {detail}")


def main():
    enc = pickle.load(open(os.path.expanduser("~/.cache/nanochat/tokenizer/tokenizer.pkl"), "rb"))
    bos = enc._special_tokens["<|bos|>"]
    base = open(os.path.join(REPO, "README.md"), encoding="utf-8").read()
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as d:
        data, out = os.path.join(d, "data"), os.path.join(d, "out")
        os.makedirs(data)
        files = []
        for f in range(4):  # 3 train + 1 val, several row groups each
            docs = []
            for _ in range(3000):
                a = int(rng.integers(0, len(base) - 400))
                docs.append(base[a:a + int(rng.integers(20, 400))] + " naïve 東京 🚀")
            p = os.path.join(data, f"shard_{f:05d}.parquet")
            pq.write_table(pa.table({"text": docs}), p, row_group_size=700)
            files.append(docs)
        open(os.path.join(data, "shard_00009.parquet.tmp"), "w").write("partial download")  # must be ignored
        tok = os.path.join(d, "tok.json")
        json.dump(tokenizer_json(enc), open(tok, "w"))

        t = time.perf_counter()
        subprocess.run([CLI, "prepare-data", "--data-dir", data, "--tokenizer", tok, "--out", out,
                        "--shard-tokens", "500000"], check=True, capture_output=True)
        t_rust = time.perf_counter() - t

        t = time.perf_counter()
        ref = {}
        for split, docs_list in [("train", files[:3]), ("val", files[3:])]:
            stream = []
            for docs in docs_list:
                for ids in enc.encode_ordinary_batch(docs, num_threads=os.cpu_count()):
                    stream.append(bos)
                    stream.extend(ids)
            ref[split] = np.array(stream)
        t_py = time.perf_counter() - t

        meta = json.load(open(os.path.join(out, "meta.json")))
        for split in ["train", "val"]:
            got = np.concatenate([open_shard(os.path.join(out, n)) for n, _ in meta[split]["shards"]]).astype(np.int64)
            check(f"{split}: token stream == Python reference", np.array_equal(got, ref[split]),
                  f"{len(got):,} tokens, {len(meta[split]['shards'])} shard(s)")
            check(f"{split}: doc count", meta[split]["docs"] == 3000 * (3 if split == "train" else 1))
        check("train split into multiple shards", len(meta["train"]["shards"]) > 1)

        B, T = 4, 64
        r = ref["train"]
        it = pretokenized_data_loader(out, B, T, "train", device="cpu")
        exact = True
        for w in range(5):
            x, y = next(it)
            exact &= np.array_equal(x.numpy().ravel(), r[w * B * T: w * B * T + B * T])
            exact &= np.array_equal(y.numpy().ravel(), r[w * B * T + 1: w * B * T + B * T + 1])
            exact &= x.dtype == torch_int32 and y.dtype == torch_int64
        check("loader batches == reference slices (+dtypes)", exact)
        v, BT = ref["val"], B * T
        n_windows = (len(v) - 1) // BT
        exact = True
        for rank in range(2):
            got = [x.numpy().ravel() for x, _ in pretokenized_data_loader(out, B, T, "val", rank=rank, world_size=2, device="cpu")]
            want = [v[w * BT:(w + 1) * BT] for w in range(rank, n_windows, 2)]
            exact &= len(got) == len(want) and all(np.array_equal(g, e) for g, e in zip(got, want))
        check("DDP: 2 ranks get disjoint interleaved windows", exact, f"{n_windows} windows")
        print(f"[INFO] prepare 12k docs: Rust CLI {t_rust:.2f}s (incl. parquet read + write) | "
              f"Python tiktoken {os.cpu_count()} threads {t_py:.2f}s (encode only)")
    print("ALL PASSED" if all(ok_all) else "SOME CHECKS FAILED")
    sys.exit(0 if all(ok_all) else 1)


import torch  # noqa: E402

torch_int32, torch_int64 = torch.int32, torch.int64

if __name__ == "__main__":
    main()
