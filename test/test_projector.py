import json

import torch
from PIL import Image
from torchvision import transforms

from nanochat_vlm.projector import VisionProjector

IMAGE_PATH = "test/assets/sample.png"
RESULTS_PATH = "test/test_projector_results.json"
PATCH_SIZE = 16
LM_DIM = 768

# 1) load a real image and turn it into a (C, H, W) tensor in [0, 1]
image = Image.open(IMAGE_PATH).convert("RGB")
to_tensor = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
pixels = to_tensor(image)  # (3, 224, 224)

# 2) split into non-overlapping patches, flatten each patch to a vector
#    this is the same patchify step a ViT does before its own encoder
c, h, w = pixels.shape
patches = pixels.unfold(1, PATCH_SIZE, PATCH_SIZE).unfold(2, PATCH_SIZE, PATCH_SIZE)
num_patches_h, num_patches_w = patches.shape[1], patches.shape[2]
patches = patches.permute(1, 2, 0, 3, 4).reshape(num_patches_h * num_patches_w, -1)
vision_feats = patches.unsqueeze(0)  # add batch dim: (1, num_patches, vision_dim)

vision_dim = vision_feats.shape[-1]  # PATCH_SIZE * PATCH_SIZE * 3
num_patches = vision_feats.shape[1]

# 3) run the real patch features through the projector
projector = VisionProjector(vision_dim, LM_DIM)
lm_feats = projector(vision_feats)

assert lm_feats.shape == (1, num_patches, LM_DIM)

# 4) record the result for this image so runs can be inspected/compared later
result = {
    "image": IMAGE_PATH,
    "patch_size": PATCH_SIZE,
    "num_patches": num_patches,
    "vision_dim": vision_dim,
    "lm_dim": LM_DIM,
    "output_shape": list(lm_feats.shape),
    "output_mean": lm_feats.mean().item(),
    "output_std": lm_feats.std().item(),
    "output_norm": lm_feats.norm(dim=-1).mean().item(),
}

results = []
try:
    with open(RESULTS_PATH) as f:
        results = json.load(f)
except FileNotFoundError:
    pass
results = [r for r in results if r["image"] != IMAGE_PATH] + [result]
with open(RESULTS_PATH, "w") as f:
    json.dump(results, f, indent=2)

print(f"OK: {IMAGE_PATH} -> {num_patches} patches of dim {vision_dim} -> projected to {tuple(lm_feats.shape)}")
print(f"Saved result to {RESULTS_PATH}")
