"""F2 paired heads: same frozen features, projection and finding-specific pooling.

B alone adds within-series context and physical geometry. It never attends across
unregistered series inside the sequence block. Padded tokens cannot enter a pool.
"""
import torch
from torch import nn


class Head(nn.Module):
    def __init__(self, features=384, width=128, geometry=False):
        super().__init__()
        self.geometry = geometry
        self.project = nn.Sequential(nn.LayerNorm(features), nn.Linear(features, width), nn.GELU())
        if geometry:
            self.slot = nn.Embedding(6, width)
            self.position = nn.Sequential(nn.Linear(5, width), nn.Tanh(), nn.Linear(width, width))
            layer = nn.TransformerEncoderLayer(width, 4, width * 2, dropout=.1,
                                               batch_first=True, norm_first=True)
            self.context = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)
        self.attention = nn.Sequential(nn.Linear(width, width), nn.Tanh(), nn.Linear(width, 12))
        self.weight = nn.Parameter(torch.empty(12, width))
        self.bias = nn.Parameter(torch.zeros(12))
        nn.init.normal_(self.weight, std=.02)

    def forward(self, features, valid, geometry):
        # [batch, six slots, fixed windows, features]
        if not valid.flatten(1).any(1).all():
            raise ValueError("empty study")
        x = self.project(features)
        if self.geometry:
            x = x + self.position(geometry) + self.slot.weight[None, :, None]
            b, s, w, d = x.shape
            flat, mask = x.reshape(b * s, w, d), valid.reshape(b * s, w)
            # Transformer cannot accept an all-padded row. Missing slots stay zero.
            keep = mask.any(1)
            out = torch.zeros_like(flat)
            out[keep] = self.context(flat[keep], src_key_padding_mask=~mask[keep])
            x = out.reshape(b, s, w, d)
        x, valid = x.flatten(1, 2), valid.flatten(1, 2)
        attn = self.attention(x).masked_fill(~valid[..., None], -torch.inf).softmax(1)
        pooled = torch.einsum("bwc,bwd->bcd", attn, x)
        return (pooled * self.weight).sum(-1) + self.bias


def masked_bce(logits, target, observed):
    if not observed.any():
        raise ValueError("no observed cells")
    loss = nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return loss[observed].mean()
