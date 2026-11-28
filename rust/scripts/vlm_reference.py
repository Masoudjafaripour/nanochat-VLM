"""
PyTorch reference for the multimodal forward pass that the Rust engine implements.

Built only from the existing, unmodified modules (`nanochat_vlm.gpt.GPT`,
`nanochat_vlm.projector.VisionProjector`, `nanochat_vlm.engine.KVCache`). The repo
has no fused VLM class yet, so this file defines the composition used by both sides:

    embeddings = concat over prompt segments of
                   wte(token_ids)            for text segments
                   projector(patches)        for image segments
    x = norm(embeddings)  ->  GPT blocks  ->  norm  ->  lm_head  ->  softcap

This is GPT.forward() with the `wte(idx)` line generalized to accept image segments.
"""
import torch

from nanochat_vlm.engine import KVCache


def embed_segments(gpt, projector, segments):
    """segments: list of ("tokens", list[int]) | ("image", FloatTensor (N, vision_dim))."""
    device = gpt.get_device()
    parts = []
    for kind, value in segments:
        if kind == "tokens":
            if len(value) == 0:
                continue
            ids = torch.tensor([value], dtype=torch.long, device=device)
            parts.append(gpt.transformer.wte(ids))
        elif kind == "image":
            parts.append(projector(value.unsqueeze(0).to(device)))
        else:
            raise ValueError(kind)
    return torch.cat(parts, dim=1)


def forward_embeds(gpt, x, kv_cache=None):
    """GPT.forward from the embedding step onward (same code path, embeds as input)."""
    B, T, _ = x.size()
    T0 = 0 if kv_cache is None else kv_cache.get_pos()
    cos_sin = gpt.cos[:, T0:T0 + T], gpt.sin[:, T0:T0 + T]
    norm = torch.nn.functional.rms_norm
    x = norm(x, (x.size(-1),))
    for block in gpt.transformer.h:
        x = block(x, cos_sin, kv_cache)
    x = norm(x, (x.size(-1),))
    softcap = 15
    logits = gpt.lm_head(x)
    return softcap * torch.tanh(logits / softcap)


@torch.inference_mode()
def vlm_logits(gpt, projector, segments):
    """Full-sequence logits (T, vocab) without a KV cache."""
    return forward_embeds(gpt, embed_segments(gpt, projector, segments))[0].float()


@torch.inference_mode()
def vlm_greedy(gpt, projector, segments, max_new_tokens, use_cache=True):
    """Greedy decoding. With use_cache, uses engine.KVCache exactly as Engine.generate does."""
    emb = embed_segments(gpt, projector, segments)
    cfg = gpt.config
    if not use_cache:
        out = []
        for _ in range(max_new_tokens):
            tok = int(forward_embeds(gpt, emb)[0, -1].argmax())
            out.append(tok)
            nxt = gpt.transformer.wte(torch.tensor([[tok]], device=emb.device))
            emb = torch.cat([emb, nxt], dim=1)
        return out
    cache = KVCache(batch_size=1, num_heads=cfg.n_kv_head, seq_len=emb.size(1) + max_new_tokens,
                    head_dim=cfg.n_embd // cfg.n_head, num_layers=cfg.n_layer)
    logits = forward_embeds(gpt, emb, cache)
    out = []
    for _ in range(max_new_tokens):
        tok = int(logits[0, -1].argmax())
        out.append(tok)
        ids = torch.tensor([[tok]], device=emb.device)
        logits = forward_embeds(gpt, gpt.transformer.wte(ids), cache)
    return out
