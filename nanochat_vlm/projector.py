# Creating a projector for the VLM model to project image features into the text embedding space.
# VisionProjector class (linear → LayerNorm) dims passed as constructor args

import torch
import torch.nn as nn


class VisionProjector(nn.Module):
    def __init__(self, vision_dim, lm_dim):
        super().__init__()
        self.proj = nn.Linear(vision_dim, lm_dim)
        self.norm = nn.LayerNorm(lm_dim)

    def forward(self, x):
        # x: (batch, num_patches, vision_dim) -> (batch, num_patches, lm_dim)
        return self.norm(self.proj(x))
