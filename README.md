# 🧠 nanochat-VLM

### A minimal Vision-Language Model trained end-to-end in the spirit of Karpathy’s **[nanochat](https://github.com/karpathy/nanochat)**


**nanochat-VLM** extends the original **nanochat** idea toward multimodality: training a small but complete **Vision-Language Model (VLM)** with the same philosophy—minimal code, full ownership, and clear learning value.

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
👉 **[notebooks/](https://github.com/Masoudjafaripour/nanochat-VLM/tree/main/notebooks)**

* inspect vision tokenization,
* test projector and fusion logic,
* debug multimodal forward passes,
* and run minimal alignment experiments.

These notebooks **do not replace** the main training pipeline and are **not the primary goal** of the project. They exist to support understanding and correctness before scaling training.

---

## 🧩 Folder Structure

```
nanochat-VLM/
├── dev/                    # experiments, data prep, and utilities
├── nanochat_vlm/           # core library (tokenizer, dataset, model components)
├── rustbpe/                # Rust-based BPE tokenizer
├── src/                    # main training & evaluation scripts (primary path)
│   ├── tok_train.py        # tokenizer training
│   ├── tok_eval.py         # tokenizer evaluation
│   ├── base_train.py       # text-only LM pretraining
│   ├── mid_train.py        # chat & multimodal alignment
│   ├── chat_sft.py         # supervised fine-tuning
│   ├── chat_eval.py        # evaluation
│   └── chat_web.py         # chat UI
├── notebooks/              # secondary, exploratory VLM notebooks
│   ├── vlm_token.ipynb
│   ├── vlm_projector.ipynb
│   ├── vlm_fusion.ipynb
│   └── ...
├── tests/                  # small unit tests
├── speedrun.sh             # end-to-end nanochat-style training script
└── README.md               # project overview
```

---

## 🛠️ Setup

```bash
# create virtual environment
python -m venv vlm-venv

# activate (Windows)
.\vlm-venv\Scripts\Activate.ps1

# install dependencies - will be added soon
pip install -r requirements.txt
```

---

## ⚙️ License

MIT License
Copyright (c) 2025 Masoud Jafaripour
Based on work by Andrej Karpathy (nanochat, MIT License)

---

### 🌟 Vision

Train a **small but real VLM**, end-to-end, with code simple enough to understand and modify—
**multimodality made teachable, not magical.**
