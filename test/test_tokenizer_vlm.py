from nanochat_vlm.tokenizer import RustBPETokenizer
from nanochat_vlm.tokenizer_vlm import SPECIAL_TOKENS_VLM

text_iter = [
    "Hello world, this is a small sample document.",
    "The quick brown fox jumps over the lazy dog.",
    "Vision language models fuse image and text tokens.",
]

tokenizer = RustBPETokenizer.train_from_iterator(text_iter, vocab_size=300, special_tokens=SPECIAL_TOKENS_VLM)

assert tokenizer.get_vocab_size() == 300
assert set(tokenizer.get_special_tokens()) == set(SPECIAL_TOKENS_VLM)

image_id = tokenizer.encode_special("<|image|>")
assert tokenizer.decode([image_id]) == "<|image|>"

print("OK: <|image|> reserved at id", image_id)
