"""
Rust BPE training (rust/tokenizer) vs the original rustbpe: identical merges, and the
resulting tokenizer encodes exactly like tiktoken built from rustbpe's ranks.

    python rust/tests/python/test_tokenizer_training.py
"""
import base64, glob, json, logging, os, sys, time

logging.getLogger("rustbpe").setLevel(logging.WARNING)
import rustbpe  # noqa: E402
import tiktoken  # noqa: E402
import nanochat_vlm_rs as rs  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, REPO)
from nanochat_vlm.tokenizer import SPLIT_PATTERN  # noqa: E402

texts = []
for p in sorted(glob.glob(os.path.join(REPO, "**", "*.py"), recursive=True) + glob.glob(os.path.join(REPO, "*.md"))):
    if "venv" not in p:
        texts += open(p, encoding="utf-8", errors="ignore").read().split("\n\n")
texts = texts * 3
ok = True
for vocab in [300, 1000, 4000]:
    t = time.perf_counter()
    ref = rustbpe.Tokenizer()
    ref.train_from_iterator(iter(texts), vocab, pattern=SPLIT_PATTERN)
    t_ref = time.perf_counter() - t
    t = time.perf_counter()
    js = json.loads(rs.train_tokenizer(texts, vocab, SPLIT_PATTERN, ["<|bos|>"]))
    t_rs = time.perf_counter() - t
    ref_ranks = {bytes(k): v for k, v in ref.get_mergeable_ranks()}
    got_ranks = {base64.b64decode(k): v for k, v in js["mergeable_ranks"]}
    same = ref_ranks == got_ranks
    enc = tiktoken.Encoding("ref", pat_str=SPLIT_PATTERN, mergeable_ranks=ref_ranks, special_tokens={"<|bos|>": len(ref_ranks)})
    tok = rs.Tokenizer.from_json(json.dumps(js))
    sample = "\n".join(texts[:200]) + " naïve 東京 🚀 12345"
    same_enc = tok.encode(sample) == enc.encode_ordinary(sample)
    ok &= same and same_enc
    print(f"[{'OK ' if same and same_enc else 'FAIL'}] vocab {vocab:>5}: merges identical={same} encode==tiktoken={same_enc}"
          f"  (rustbpe {t_ref:.2f}s, rust {t_rs:.2f}s)")
print("ALL PASSED" if ok else "FAILED")
sys.exit(0 if ok else 1)
