# 🧠 nanochat-VLM

### A minimal Vision-Language Model trained end-to-end in the spirit of Karpathy’s **[nanochat](https://github.com/karpathy/nanochat)**


**nanochat-VLM** extends the original **nanochat** idea toward multimodality: training a small but complete **Vision-Language Model (VLM)** with the same philosophy—minimal code, full ownership, and clear learning value (hopefully with similar cost of training).

The **primary goal** of this repo is to **train a VLM end-to-end using nanochat-style scripts** (tokenizer → LM → multimodal training), not just to demo inference.
A small set of notebooks is included as a **secondary, exploratory path** to understand and validate VLM components in isolation.

This project is designed for **learning, research, and hackability**, not scale.

---

## 🚀 Goals

* Rebuild the **core nanochat LLM pipeline** (tokenization → transformer → training).
* Extend it to a **Vision-Language Model** using a vision encoder + fusion.
* Train the model **script-first**, nanochat-style (not notebook-first).
* Keep the system transparent, minimal, and easy to modify.
* Enable experimentation on limited compute (single GPU or small server).

---

## 📚 Learning & Training Milestones

### Primary Path (Main Objective)

1. **Tokenizer (Text)**

   * Train and evaluate a BPE tokenizer on large-scale text.
   * Cache and reuse tokenizer artifacts across machines.

2. **Base Language Model (Text-only)**

   * Train a GPT-style decoder-only transformer from scratch.
   * Validate with loss curves and standard benchmarks.

3. **Vision Encoder Integration**

   * Add a pretrained vision encoder (CLIP / ViT).
   * Project visual features into the LM embedding space.

4. **Multimodal Training**

   * Fuse image tokens with text tokens.
   * Train a VLM end-to-end (nanochat-style scripts).

5. **Multimodal Chat & Evaluation**

   * Enable image + text chat.
   * Evaluate on small VLM datasets (captioning, VQA-style tasks).

### Secondary Path (Notebooks – Supporting, Not Primary)

A small set of notebooks is provided in
👉 **[notebooks/](https://github.com/Masoudjafaripour/nanochat-VLM/tree/main/notebooks)**:

* `ch1-vlm-tokenizer-training.ipynb` — vision tokenization / patch embeddings
* `ch2_vlm_projector.ipynb` — projecting vision embeddings into LM embedding space
* `ch3_vlm_fusion.ipynb` — fusing image + text tokens, attention masks
* `ch4_vlm_forward.ipynb` — full multimodal forward pass (inference only)
* `ch5_vlm_train_minimal.ipynb` — first learning step, training only the projector
* `ch6_vlm_generation.ipynb` — multimodal generation / qualitative evaluation

These notebooks **do not replace** the main training pipeline and are **not the primary goal** of the project. They exist to support understanding and correctness before scaling training.

---

## 🧠 Algorithm / Pipeline

**1. Tokenizer.** A byte-level BPE tokenizer (via [`rustbpe`](rustbpe/), a small Rust training routine, exported for fast inference through `tiktoken`/HuggingFace `tokenizers`) trained directly on the pretraining corpus. Special tokens (`<|bos|>`, `<|user_start|>`, `<|assistant_start|>`, `<|image|>`, …) are reserved up front in [`nanochat_vlm/tokenizer_vlm.py`](nanochat_vlm/tokenizer_vlm.py) so the same vocabulary can later carry image placeholders.

**2. Base language model.** A GPT-style decoder-only Transformer ([`nanochat_vlm/gpt.py`](nanochat_vlm/gpt.py)) with: rotary embeddings (no learned positional embeddings), QK-norm, untied input/output embeddings, ReLU² MLP activation, bias-free linear layers, parameter-free RMSNorm, and Multi-Query Attention for cheaper inference. Optimized with a mix of **Muon** (hidden matrices) and **AdamW** (embeddings/scalars), see [`nanochat_vlm/muon.py`](nanochat_vlm/muon.py) / [`nanochat_vlm/adamw.py`](nanochat_vlm/adamw.py). Data is streamed from FineWeb-Edu parquet shards ([`nanochat_vlm/dataset.py`](nanochat_vlm/dataset.py), [`nanochat_vlm/dataloader.py`](nanochat_vlm/dataloader.py)).

**3. Vision → language bridge (in progress).** A pretrained vision encoder (CLIP/ViT/DINO) produces patch embeddings, which a linear/MLP **projector** maps into the LM's embedding space. Image tokens are prepended to the text sequence (`[IMG]* + text`) with a matching attention mask, so the frozen (or fine-tuned) LM attends over both modalities jointly. This is currently developed and validated in the `notebooks/` (secondary path) before being promoted into `src/` training scripts.

**4. Training stages** (nanochat-style): tokenizer training → base pretraining → (planned) vision-projector alignment → (planned) multimodal fusion training → (planned) chat SFT / eval. Model size scales with a single `depth` knob (`model_dim = depth * 64`, heads/layers derived from it), so the same scripts run a tiny CPU-sized model or a large multi-GPU model.

---

## 🧩 Folder Structure

```
nanochat-VLM/
├── dev/                      # experiments, data prep, and utilities
├── nanochat_vlm/             # core library
│   ├── tokenizer.py          # RustBPE-backed tokenizer wrapper (train/save/load/encode/decode)
│   ├── tokenizer_vlm.py      # VLM special tokens (incl. <|image|>)
│   ├── dataset.py            # FineWeb-Edu parquet shard download + iteration
│   ├── dataloader.py         # distributed token data loader
│   ├── gpt.py                # GPT model definition (rotary, QK-norm, MQA, ...)
│   ├── muon.py / adamw.py    # optimizers (Muon for matrices, AdamW for the rest)
│   ├── engine.py             # inference/generation engine
│   ├── core_eval.py          # CORE benchmark evaluation
│   ├── loss_eval.py          # bits-per-byte loss evaluation
│   ├── checkpoint_manager.py # save/load checkpoints
│   ├── common.py             # device/DDP setup, logging, base-dir helpers
│   ├── configurator.py       # lightweight CLI-arg override helper
│   └── report.py             # writes markdown run reports
├── rustbpe/                  # Rust BPE tokenizer trainer (built via maturin)
├── src/                      # training & evaluation entry points (primary path)
│   ├── tok_train.py          # tokenizer training
│   ├── tok_eval.py           # tokenizer evaluation
│   ├── base_train.py         # text-only LM pretraining
│   └── base_eval.py          # CORE benchmark evaluation of a trained model
├── notebooks/                # secondary, exploratory VLM notebooks (see above)
├── speedrun.sh                # reference end-to-end nanochat run script (upstream nanochat paths; see note below)
└── README.md                  # project overview
```

> **Status note:** the VLM stages (vision encoder integration, fusion, multimodal training/eval, chat SFT/web UI) are being prototyped in `notebooks/` and are not yet promoted to `src/` scripts. `speedrun.sh` is carried over from upstream nanochat and still references its original module layout (`scripts.*`, `nanochat.*`); use the commands below (`src.*`, `nanochat_vlm.*`) to run this repo's pipeline instead.

---

## 🛠️ Setup

```bash
# create & activate a virtual environment
python -m venv vlm-venv
source vlm-venv/bin/activate      # macOS/Linux
# .\vlm-venv\Scripts\Activate.ps1   # Windows

# install Python dependencies
pip install torch wandb pyarrow pandas requests tiktoken tokenizers jinja2 psutil pyyaml maturin

# build the Rust BPE tokenizer (requires the Rust toolchain: https://rustup.rs)
maturin develop --release --manifest-path rustbpe/Cargo.toml
```

By default all caches/checkpoints/data go to `~/.cache/nanochat` (override with the `NANOCHAT_BASE_DIR` env var).

---

## ▶️ How to Run

```bash
# 1) Download a few FineWeb-Edu shards (each ~100MB; -n -1 downloads all 1822)
python -m nanochat_vlm.dataset -n 8

# 2) Train the tokenizer on ~2B characters
python -m src.tok_train --max_chars=2000000000
python -m src.tok_eval

# 3) Pretrain the base language model
# GPU (single node, e.g. 8xH100):
torchrun --standalone --nproc_per_node=8 -m src.base_train -- --depth=20

# CPU / Apple Silicon (tiny smoke-test model):
python -m src.base_train --depth=4 --max_seq_len=512 --device_batch_size=1 \
    --eval_tokens=512 --core_metric_every=-1 --total_batch_size=512 --num_iterations=20

# 4) Evaluate the trained base model on CORE benchmarks
python -m src.base_eval --max-per-task 100
```

**Vision-Language exploration (secondary path):** open the notebooks in order —
`ch1` → `ch2` → `ch3` → `ch4` → `ch5` → `ch6` in [`notebooks/`](notebooks/) — to walk through vision tokenization, projector training, fusion, forward pass, minimal training, and generation.

---

## ⚙️ License

MIT License
Copyright (c) 2025 Masoud Jafaripour
Based on work by Andrej Karpathy (nanochat, MIT License)

---

### 🌟 Vision

Train a **small but real VLM**, end-to-end, with code simple enough to understand and modify—
**multimodality made teachable, not magical.**
