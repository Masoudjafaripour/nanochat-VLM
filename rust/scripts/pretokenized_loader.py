"""
Read `.bin` token shards written by `nanochat-vlm prepare-data` and yield training batches
with the same contract as `nanochat_vlm.dataloader.tokenizing_distributed_data_loader`:
`inputs` int32 (B, T) and `targets` int64 (B, T), endless for "train".

Tokenization already happened in Rust, so this is just memory-mapped slicing.
DDP: rank r takes windows r, r + world, r + 2*world, ... (disjoint across ranks).
"""
import glob
import os

import numpy as np
import torch

MAGIC = b"NCVTOK01"
HEADER = 32


def open_shard(path):
    with open(path, "rb") as f:
        h = f.read(HEADER)
    assert h[:8] == MAGIC, f"{path}: not a NCVTOK01 shard"
    width = int.from_bytes(h[8:12], "little")
    n = int.from_bytes(h[16:24], "little")
    dtype = {2: np.uint16, 4: np.uint32}[width]
    return np.memmap(path, dtype=dtype, mode="r", offset=HEADER, shape=(n,))


def pretokenized_data_loader(data_dir, B, T, split, rank=0, world_size=1, device="cuda", loop=None):
    shards = sorted(glob.glob(os.path.join(data_dir, f"{split}_*.bin")))
    assert shards, f"no {split}_*.bin shards in {data_dir}"
    loop = (split == "train") if loop is None else loop
    need = B * T + 1  # +1 for the last target, as in dataloader.py
    while True:
        tokens = np.concatenate([open_shard(p) for p in shards]) if len(shards) > 1 else open_shard(shards[0])
        for w in range(rank, (len(tokens) - 1) // (B * T), world_size):
            chunk = torch.from_numpy(tokens[w * B * T: w * B * T + need].astype(np.int64))
            inputs = chunk[:-1].view(B, T).to(device=device, dtype=torch.int32, non_blocking=True)
            targets = chunk[1:].view(B, T).to(device=device, non_blocking=True)
            yield inputs, targets
        if not loop:
            return
