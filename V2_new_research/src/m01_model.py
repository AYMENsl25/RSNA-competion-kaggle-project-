"""M01: M00 matched comparison. Same random-init ResNet18 shared encoder and input encoding, but the
shared max + linear head is replaced by 12 finding-specific learned attention queries over valid
windows (all roles jointly); each finding reads its own pooled vector -> one logit per target.
TARGETS, masked_bce and predict_proba are reused unchanged from M00."""
import torch
import torch.nn as nn
from torchvision.models import resnet18

from m00_model import TARGETS, masked_bce, predict_proba  # noqa: F401


class M01(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = resnet18(weights=None)
        self.encoder.fc = nn.Identity()
        self.head = nn.Linear(512, len(TARGETS))   # row k = finding k readout; created before queries
        self.queries = nn.Parameter(torch.randn(len(TARGETS), 512) * 0.02)  # so encoder/head init == M00

    def forward(self, x, mask):
        """x [B,R,W,3,H,W2] float, mask [B,R,W] bool -> logits [B,12]. Only valid windows are
        encoded; padded windows get exactly zero attention, so they cannot influence the output."""
        if x.ndim != 6 or x.shape[3] != 3 or mask.dtype != torch.bool or mask.shape != x.shape[:3]:
            raise ValueError(f"bad shapes x={tuple(x.shape)} mask={tuple(mask.shape)} {mask.dtype}")
        has = mask.flatten(1).any(1)
        if not has.all():
            raise ValueError(f"studies with no valid window (batch idx {(~has).nonzero().flatten().tolist()})")
        sel = x[mask]
        if not torch.isfinite(sel).all():
            raise ValueError("nonfinite pixels in a valid window")
        feats = self.encoder(sel)                                   # [n,512]
        valid = mask.flatten(1)                                     # [B,N]
        dense = feats.new_zeros(valid.shape + feats.shape[1:])      # [B,N,512], padded slots stay 0
        dense[valid] = feats
        scores = torch.einsum("bnd,kd->bkn", dense, self.queries) / feats.shape[1] ** 0.5
        attn = scores.masked_fill(~valid[:, None, :], float("-inf")).softmax(-1)   # [B,12,N]
        pooled = torch.einsum("bkn,bnd->bkd", attn, dense)          # [B,12,512]
        return (pooled * self.head.weight).sum(-1) + self.head.bias
