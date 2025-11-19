""" 
BPE tokenizer for VLM models:

It extends the base BPE tokenizer with specific configurations for VLM
which may include handling of multimodal input tokens.
For simplicity, I started with huggingface's implementation.
RustBPE_vlm will be implemented later for performance optimization.
"""

import os, copy
from functools import lru_cache

SPECIAL_TOKENS_VLM = [
    "<|bos|>",
    "<|user_start|>",
    "<|user_end|>",
    "<|assistant_start|>",
    "<|assistant_end|>",
    "<|python_start|>",
    "<|python_end|>",
    "<|output_start|>",
    "<|output_end|>",
    "<|image|>",          # single image token for VLM input
]
