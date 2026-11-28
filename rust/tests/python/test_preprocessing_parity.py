"""
Numerical-equivalence test: Rust preprocessing vs the PIL + torchvision reference
used in test/test_projector.py.

Run from the repo root after `maturin develop --release` in rust/bindings:
    python rust/tests/python/test_preprocessing_parity.py
"""
import io
import os
import sys
import tempfile

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

import nanochat_vlm_rs as rs

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SAMPLE = os.path.join(REPO, "test", "assets", "sample.png")


def reference_patches(path, size=(224, 224), patch=16, interp=transforms.InterpolationMode.BILINEAR, mean=None, std=None):
    """Exactly the steps of test/test_projector.py (plus optional Normalize)."""
    img = Image.open(path).convert("RGB")
    tf = [transforms.Resize(size, interpolation=interp), transforms.ToTensor()]
    if mean is not None:
        tf.append(transforms.Normalize(mean, std))
    x = transforms.Compose(tf)(img)
    p = x.unfold(1, patch, patch).unfold(2, patch, patch)
    nh, nw = p.shape[1], p.shape[2]
    return p.permute(1, 2, 0, 3, 4).reshape(nh * nw, -1).numpy()


def check(name, ref, got, atol):
    assert ref.shape == got.shape, f"{name}: shape {ref.shape} vs {got.shape}"
    diff = np.abs(ref - got).max()
    status = "OK " if diff <= atol else "FAIL"
    print(f"[{status}] {name:<48} max|diff|={diff:.3e} (atol {atol:.0e})")
    return diff <= atol


def main():
    ok = True
    # 1) the repo's real sample image, default config: must be bit-exact
    ok &= check("sample.png default", reference_patches(SAMPLE), rs.preprocess(SAMPLE), 0.0)

    # 2) uint8 resize alone, both filters, against PIL directly
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as d:
        sizes = [(224, 224), (37, 501), (640, 480), (1000, 13), (225, 223), (96, 96)]
        targets = [(224, 224), (64, 32), (448, 336)]
        for i, (h, w) in enumerate(sizes):
            arr = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
            path = os.path.join(d, f"rand_{i}.png")
            Image.fromarray(arr).save(path)
            for (th, tw) in targets:
                for filt, pil in [("bilinear", Image.BILINEAR), ("bicubic", Image.BICUBIC)]:
                    ref = np.asarray(Image.open(path).convert("RGB").resize((tw, th), pil)).astype(np.int16)
                    got = rs.load_and_resize(path, th, tw, filt).astype(np.int16)
                    ok &= check(f"resize {h}x{w}->{th}x{tw} {filt}", ref, got, 0)

        # 3) RGBA and grayscale inputs (convert("RGB") semantics)
        rgba = rng.integers(0, 256, (120, 90, 4), dtype=np.uint8)
        Image.fromarray(rgba, "RGBA").save(os.path.join(d, "rgba.png"))
        gray = rng.integers(0, 256, (77, 150), dtype=np.uint8)
        Image.fromarray(gray, "L").save(os.path.join(d, "gray.png"))
        for n in ["rgba.png", "gray.png"]:
            p = os.path.join(d, n)
            ok &= check(f"{n} default", reference_patches(p), rs.preprocess(p), 0.0)

        # 4) Normalize (CLIP stats) + bicubic: float math, allow fp32 rounding
        mean, std = [0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]
        cfg = rs.PreprocessConfig(image_size=(224, 224), patch_size=14, filter="bicubic", mean=mean, std=std)
        ref = reference_patches(SAMPLE, patch=14, interp=transforms.InterpolationMode.BICUBIC, mean=mean, std=std)
        ok &= check("sample.png CLIP-normalize bicubic p14", ref, rs.preprocess(SAMPLE, cfg), 1e-6)

        # 5) batch API == per-image API
        paths = [SAMPLE, os.path.join(d, "rgba.png"), os.path.join(d, "gray.png")]
        batch = rs.preprocess_batch(paths)
        ok &= check("preprocess_batch vs per-image", np.stack([rs.preprocess(p) for p in paths]), batch, 0.0)

        # 6) JPEG: decoders differ (libjpeg-turbo vs zune-jpeg) -> documented tolerance, not exact
        jpg = os.path.join(d, "sample.jpg")
        Image.open(SAMPLE).convert("RGB").save(jpg, quality=90)
        diff = np.abs(reference_patches(jpg) - rs.preprocess(jpg))
        print(f"[INFO] JPEG decode difference: max={diff.max()*255:.1f}/255, mean={diff.mean()*255:.3f}/255 (decoder-dependent)")

        # 7) bytes API
        with open(SAMPLE, "rb") as f:
            ok &= check("preprocess_bytes(sample.png)", reference_patches(SAMPLE), rs.preprocess_bytes(f.read()), 0.0)

    print("ALL PASSED" if ok else "SOME CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
