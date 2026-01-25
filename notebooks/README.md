# nanochat-VLM — Phase 1 (VLM Foundations, No LLM Training)

This phase builds a **minimal, correct Vision–Language pipeline** on top of a pretrained nanochat text backbone, **without training the LLM**.

## Notebook Sequence

1. **vlm_token.ipynb**  
   Load a vision encoder (CLIP / ViT / DINO).  
   Image → patch embeddings. Inspect shapes and batching.

2. **vlm_projector.ipynb**  
   Map vision embeddings → LM embedding dimension using a linear/MLP projector.

3. **vlm_fusion.ipynb**  
   Define fusion strategy (prepend image tokens, `[IMG] + text`).  
   Build joint token sequence + attention masks.

4. **vlm_forward.ipynb**  
   Run full multimodal forward pass: image + text → logits.  
   Inference only, no training.

5. **vlm_train_minimal.ipynb**  
   First learning step.  
   Freeze LM + vision encoder, train **only the projector** on a tiny dataset.

6. **vlm_eval.ipynb**  
   Qualitative evaluation: captioning, VQA-style prompts, failure analysis.

## Principle

> **Make it run → make it correct → then make it learn.**

This phase focuses on **correct multimodal wiring**, not scale or performance.
