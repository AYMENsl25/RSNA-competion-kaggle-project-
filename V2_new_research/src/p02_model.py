"""P02: P00 encoder and head with target-specific max over valid MRI windows."""

import torch

from p00_model import P00


class P02(P00):
    def forward(self, x, mask):
        if x.ndim != 6 or x.shape[3] != 3 or mask.dtype != torch.bool or mask.shape != x.shape[:3]:
            raise ValueError(f"bad shapes x={tuple(x.shape)} mask={tuple(mask.shape)} {mask.dtype}")
        if not mask.flatten(1).any(1).all():
            raise ValueError("study with no valid window")
        selected = x[mask]
        if not torch.isfinite(selected).all():
            raise ValueError("nonfinite pixels in a valid window")
        logits = self.head(self.encoder(selected))
        study = mask.flatten(1).nonzero()[:, 0]
        pooled = logits.new_full((x.shape[0], logits.shape[1]), float("-inf"))
        return pooled.scatter_reduce(0, study[:, None].expand_as(logits), logits, "amax", include_self=True)
