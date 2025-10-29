# 🧠 nanochat-VLM

### A minimal Vision-Language Model built from scratch, inspired by Karpathy's **nanochat**

**nanochat-VLM** aims to teach and demonstrate how to extend a minimal language model into a multimodal system that understands both **text and images**.
It is designed for learning, research, and hackability — not for scale.

---

## 🚀 Goals

* Rebuild the **core LLM pipeline** (tokenization → transformer → generation).
* Add **vision capability** via a lightweight encoder and fusion mechanism.
* Keep the implementation fully transparent and under 1K lines of Python.
* Enable easy experimentation on a single GPU or CPU laptop.

---

## 📚 Learning Milestones

1. **Milestone 1A – Tokenizer**
   Learn how text → token IDs → text works using `tiktoken` or a toy BPE.

2. **Milestone 1B – MiniGPT (Text-only)**
   Implement a simple decoder-only Transformer (embedding, attention, MLP).

3. **Milestone 2 – Vision Encoder + Fusion**
   Add a ViT or CLIP-based image encoder, project visual features into the LLM embedding space, and fuse them as prefix tokens.

4. **Milestone 3 – Multimodal Chat**
   Extend the chat interface to accept both text and image inputs, producing grounded text responses.

5. **Milestone 4 – Evaluation & Scaling**
   Train or fine-tune on small multimodal datasets (e.g., COCO captions) and visualize performance.

---

## 🧩 Folder Structure

```
nanochat-VLM/
├── src/                # scripts & core modules
│   ├── tok_train.py    # tokenizer training / testing
│   ├── mingpt.py       # minimal Transformer
│   ├── train_base.py   # text pretraining
│   ├── vision_encoder.py # ViT / CLIP visual encoder
│   └── chat_web.py     # multimodal chat UI (FastAPI)
├── data/               # small text/image datasets
├── tests/              # small unit tests
└── README.md           # you are here
```

---

## 🛠️ Setup

```bash
# create virtual environment
python -m venv vlm-venv
# activate (Windows)
.\vlm-venv\Scripts\Activate.ps1
# install dependencies
pip install -r requirements.txt
```

---

## ⚙️ License

MIT License
Copyright (c) 2025 Masoud Jafaripour
Based on work by Andrej Karpathy (nanochat, MIT License)

---

### 🌟 Vision

Make multimodality **understandable**, **teachable**, and **hackable** — one milestone at a time.

### My issues: 
## Tokenizer Setup

**Issue:** `FileNotFoundError: tokenizer.pkl not found in ~/.cache/nanochat/tokenizer/`

**Cause:** The tokenizer must be trained once and cached. When moving between machines (Windows ↔ Linux), the cache doesn't transfer automatically.

**Solution:** Copy tokenizer files from the machine where it was trained to `~/.cache/nanochat/tokenizer/` on the new machine.

**Files:**
- `tokenizer.pkl` - Trained BPE tokenizer (vocabulary, merge rules, special tokens)
- `token_bytes.pt` - Byte representations for bits-per-byte evaluation metric

**Commands:**
```bash
# Create cache directory
mkdir -p ~/.cache/nanochat/tokenizer/

# Copy files (adjust source path as needed)
cp /path/to/tokenizer.pkl ~/.cache/nanochat/tokenizer/
cp /path/to/token_bytes.pt ~/.cache/nanochat/tokenizer/
```

**Note:** Train the tokenizer once using the training script before first model training. The cached files can be reused across training runs.