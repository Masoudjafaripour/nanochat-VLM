"""
Rust eval (`nanochat-vlm eval-core` / `eval-bpb`) vs the unmodified Python eval
(`src/base_eval.evaluate_model`, `nanochat_vlm.loss_eval.evaluate_bpb`) on the same
random model, tokenizer and a synthetic eval bundle covering all three task types,
0/1/3/10-shot (both branches of random.sample).

    python rust/tests/python/test_eval_parity.py
"""
import json
import logging
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
logging.getLogger("rustbpe").setLevel(logging.WARNING)

from export_checkpoint import random_gpt, save_safetensors, tokenizer_json  # noqa: E402
from pretokenized_loader import pretokenized_data_loader  # noqa: E402

CLI = os.path.join(RUST, "target", "release", "nanochat-vlm")
ok_all = []


def check(name, ok, detail=""):
    ok_all.append(bool(ok))
    print(f"[{'OK ' if ok else 'FAIL'}] {name:<48} {detail}")


def make_bundle(d, text, rng):
    words = text.split()
    phrase = lambda n: " ".join(words[i] for i in rng.integers(0, len(words), n))  # noqa: E731
    tasks = [
        ("mc_0shot", "multiple_choice", 0, 40, " "),
        ("mc_3shot", "multiple_choice", 3, 40, "\nAnswer: "),
        ("schema_1shot", "schema", 1, 40, " "),
        ("lm_10shot", "language_modeling", 10, 120, " "),  # n > setsize -> set branch of sample()
    ]
    os.makedirs(os.path.join(d, "eval_data"))
    yaml_lines, csv = ["icl_tasks:"], ["Eval Task,Random baseline"]
    for label, ttype, shots, n, delim in tasks:
        rows = []
        for _ in range(n):
            if ttype == "multiple_choice":
                rows.append({"query": phrase(8), "choices": [phrase(int(rng.integers(1, 4))) for _ in range(4)], "gold": int(rng.integers(0, 4))})
            elif ttype == "schema":
                rows.append({"context_options": [phrase(6) for _ in range(3)], "continuation": phrase(4), "gold": int(rng.integers(0, 3))})
            else:  # short continuations so a random model sometimes gets them right
                rows.append({"context": "  " + phrase(10) + " \n", "continuation": words[int(rng.integers(0, 5))]})
        with open(os.path.join(d, "eval_data", f"{label}.jsonl"), "w") as f:
            f.write("\n".join(json.dumps(r) for r in rows) + "\n")
        yaml_lines += [f"- label: {label}", f"  dataset_uri: {label}.jsonl", f"  num_fewshot: [{shots}]",
                       f"  icl_task_type: {ttype}", f"  continuation_delimiter: {json.dumps(delim)}"]
        csv.append(f"{label},{[25, 25, 33.3, 0][len(csv) - 1]}")
    open(os.path.join(d, "core.yaml"), "w").write("\n".join(yaml_lines) + "\n")
    open(os.path.join(d, "eval_meta_data.csv"), "w").write("\n".join(csv) + "\n")


def main():
    torch.set_num_threads(16)
    from nanochat_vlm.tokenizer import RustBPETokenizer
    text = open(os.path.join(REPO, "README.md"), encoding="utf-8").read()
    for f in ["gpt.py", "engine.py"]:
        text += open(os.path.join(REPO, "nanochat_vlm", f), encoding="utf-8").read()
    tok = RustBPETokenizer.train_from_iterator(iter([text] * 3), vocab_size=1024)
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as d:
        # model + tokenizer export
        gpt = random_gpt(4, tok.get_vocab_size(), 256, None, seed=0).eval()
        mdir = os.path.join(d, "model")
        os.makedirs(mdir)
        save_safetensors({k: v.float() for k, v in gpt.state_dict().items()}, os.path.join(mdir, "model.safetensors"))
        c = gpt.config
        json.dump({"model": dict(sequence_len=c.sequence_len, vocab_size=c.vocab_size, n_layer=c.n_layer, n_head=c.n_head,
                                 n_kv_head=c.n_kv_head, n_embd=c.n_embd), "projector": None},
                  open(os.path.join(mdir, "config.json"), "w"))
        json.dump(tokenizer_json(tok.enc), open(os.path.join(mdir, "tokenizer.json"), "w"))

        # ---- CORE ----
        os.environ["NANOCHAT_BASE_DIR"] = d
        bundle = os.path.join(d, "eval_bundle")
        os.makedirs(bundle)
        make_bundle(bundle, text, rng)
        sys.path.insert(0, os.path.join(REPO, "src"))
        import base_eval
        py = base_eval.evaluate_model(gpt, tok, "cpu", max_per_task=-1)
        dump = os.path.join(d, "prompts.jsonl")
        out = subprocess.run([CLI, "eval-core", "--model", mdir, "--bundle", bundle, "--json", "--dump-prompts", dump],
                             capture_output=True, text=True)
        assert out.returncode == 0, out.stderr
        rs = json.loads(out.stdout)
        rs_res = dict(rs["results"])
        for label, acc in py["results"].items():
            check(f"CORE {label} accuracy (float32 in Python)", abs(rs_res[label] - acc) < 1e-6, f"python {acc:.4f} rust {rs_res[label]:.4f}")
        # every rendered prompt (few-shot sampling + templates) vs core_eval.py
        import random
        import yaml
        from nanochat_vlm import core_eval as ce
        render = {"multiple_choice": ce.render_prompts_mc, "schema": ce.render_prompts_schema, "language_modeling": ce.render_prompts_lm}
        tasks = {t["label"]: t for t in yaml.safe_load(open(os.path.join(bundle, "core.yaml")))["icl_tasks"]}
        dumped = [json.loads(l) for l in open(dump)]
        same, cache = 0, {}
        for row in dumped:
            t = tasks[row["label"]]
            if row["label"] not in cache:
                data = [json.loads(l) for l in open(os.path.join(bundle, "eval_data", t["dataset_uri"]))]
                random.Random(1337).shuffle(data)
                cache[row["label"]] = data
            data, idx, k = cache[row["label"]], row["idx"], t["num_fewshot"][0]
            shots = []
            if k > 0:
                r = random.Random(1234 + idx)
                shots = [data[i] for i in r.sample([i for i in range(len(data)) if i != idx], k)]
            same += render[t["icl_task_type"]](data[idx], t.get("continuation_delimiter", " "), shots) == row["prompts"]
        check("every rendered prompt == core_eval.py", same == len(dumped), f"{same}/{len(dumped)} examples")
        check("CORE metric", abs(rs["core_metric"] - py["core_metric"]) < 1e-6,
              f"python {py['core_metric']:.6f} rust {rs['core_metric']:.6f}")

        # ---- bits per byte ----
        from nanochat_vlm.loss_eval import evaluate_bpb
        data = os.path.join(d, "tokens")
        os.makedirs(data)
        import pyarrow as pa
        import pyarrow.parquet as pq
        for i in range(2):
            docs = [text[a:a + 300] for a in rng.integers(0, len(text) - 300, 400)]
            pq.write_table(pa.table({"text": docs}), os.path.join(d, f"shard_{i:05d}.parquet"))
        subprocess.run([CLI, "prepare-data", "--data-dir", d, "--tokenizer", os.path.join(mdir, "tokenizer.json"),
                        "--out", data], check=True, capture_output=True)
        vocab = tok.get_vocab_size()
        special = set(tok.get_special_tokens())
        token_bytes = torch.tensor([0 if tok.decode([i]) in special else len(tok.decode([i]).encode("utf-8"))
                                    for i in range(vocab)], dtype=torch.int32)  # exactly tok_train.py's recipe
        B, T, steps = 4, 128, 6
        py_bpb = evaluate_bpb(gpt, pretokenized_data_loader(data, B, T, "val", device="cpu"), steps, token_bytes)
        out = subprocess.run([CLI, "eval-bpb", "--model", mdir, "--data-dir", data, "--batch-size", str(B),
                              "--seq-len", str(T), "--steps", str(steps)], capture_output=True, text=True, check=True)
        rs_bpb = json.loads(out.stdout)["bpb"]
        check("bits per byte", abs(rs_bpb - py_bpb) / py_bpb < 1e-5, f"python {py_bpb:.6f} rust {rs_bpb:.6f}")
    print("ALL PASSED" if all(ok_all) else "SOME CHECKS FAILED")
    sys.exit(0 if all(ok_all) else 1)


if __name__ == "__main__":
    main()
