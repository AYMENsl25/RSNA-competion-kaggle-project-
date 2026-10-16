"""V3 shared eval path: mirror, window selection, normalisation, model heads.
Used by src/train_v3.py and bundled verbatim into notebooks/v3_submission (parity depends on it)."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from m00_data import ROLES, window_centers
from m00_model import M00, TARGETS

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def mirror(vol, plane):
    """Patient L<->R mirror in canonical cache space: sagittal slices run along +x, so reverse
    slice order; coronal/axial columns run along +x, so flip columns. Self-inverse."""
    return vol[::-1] if plane == "Sagittal" else vol[..., ::-1]


def fill_windows(x, m, vol, n, rng=None, positions=None):
    """Write up to n 3-slice windows of uint8 vol [L,S,S] into x [n,3,S,S], m [n].
    rng None -> evenly spaced centres (eval, --slab); else random (P00 train).
    ponytail: all slices treated usable (cache nan-fills bad slices; bad_slices=0 so far)."""
    for k, c in enumerate(window_centers(np.ones(len(vol), bool), n, rng)):
        x[k], m[k] = vol[c - 1:c + 2], True
        if positions is not None:
            positions[k] = c / (len(vol) - 1)


def prep(x, res):
    """uint8 [B,R,W,3,S,S] -> ImageNet-normalised float [B,R,W,3,res,res] (same device)."""
    lead = x.shape[:3]
    x = x.flatten(0, 2).float().div_(255)
    if res != x.shape[-1]:
        x = F.interpolate(x, size=(res, res), mode="bilinear", antialias=True, align_corners=False)
    x = (x - MEAN.to(x.device)) / STD.to(x.device)
    return x.view(*lead, 3, res, res)


TIMM = {"convnext_tiny": "convnext_tiny.fb_in22k_ft_in1k"}  # same ImageNet mean/std as MEAN/STD


class V3(M00):
    """head='max': exactly M00/P00 (same state_dict keys). head='attn': per-finding gated
    attention-MIL (Ilse 2018) over all valid windows of all planes + learned per-finding plane
    bias; logit_f = <pooled_f, head.weight[f]> + head.bias[f]."""

    def __init__(self, head="max", hidden=128, backbone="resnet18"):
        super().__init__()
        self.head_type = head
        if backbone != "resnet18":  # timm encoder, random init; weights come from pretrained/checkpoint
            import timm
            self.encoder = timm.create_model(TIMM[backbone], pretrained=False, num_classes=0)
            self.head = nn.Linear(self.encoder.num_features, len(TARGETS))
        d = self.head.in_features
        if head == "attn":
            self.att_v = nn.Linear(d, hidden)
            self.att_u = nn.Linear(d, hidden)
            self.att_w = nn.Linear(hidden, len(TARGETS))
            self.plane_bias = nn.Parameter(torch.zeros(len(ROLES), len(TARGETS)))
        elif head == "context":
            self.project = nn.Sequential(nn.Linear(d, 384), nn.LayerNorm(384))
            self.slot_embed = nn.Embedding(6, 384)
            self.position = nn.Sequential(nn.Linear(1, 384), nn.GELU(), nn.Linear(384, 384))
            layer = nn.TransformerEncoderLayer(384, 8, 768, dropout=0.1, batch_first=True, norm_first=True)
            self.context = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
            self.att_w = nn.Linear(384, len(TARGETS))
            self.head = nn.Linear(384, len(TARGETS))
        elif head != "max":
            raise ValueError(head)

    def forward(self, x, mask, positions=None):
        if self.head_type == "max":
            return super().forward(x, mask)
        if x.ndim != 6 or mask.shape != x.shape[:3] or mask.shape[1] not in (3, 6) \
                or not mask.flatten(1).any(1).all():
            raise ValueError(f"bad shapes / empty study x={tuple(x.shape)} mask={tuple(mask.shape)}")
        if self.head_type == "context":
            if mask.shape[1] != 6 or positions is None or positions.shape != mask.shape:
                raise ValueError("context head requires six slots and matching positions")
            if not torch.isfinite(positions[mask]).all() or not ((positions[mask] >= 0) & (positions[mask] <= 1)).all():
                raise ValueError("valid positions must be finite in [0,1]")
            features = self.project(self.encoder(x[mask]))
            h = features.new_zeros(*mask.shape, 384)
            h[mask] = features
            slot = self.slot_embed(torch.arange(6, device=x.device))[None, :, None]
            pos = positions.masked_fill(~mask, 0).to(features.dtype)
            h = h + slot + self.position(pos[..., None])
            valid = mask.flatten(1)
            h = self.context(h.flatten(1, 2), src_key_padding_mask=~valid)
            a = self.att_w(h).float().masked_fill(~valid[..., None], float("-inf")).softmax(1)
            pooled = torch.einsum("bnf,bnd->bfd", a, h.float())
            return (pooled * self.head.weight.float()).sum(-1) + self.head.bias.float()
        feats = self.encoder(x[mask])                                       # [n,d]
        h = feats.new_zeros(*mask.shape, feats.shape[1])
        h[mask] = feats                                                      # [B,R,W,512]
        a = self.att_w(torch.tanh(self.att_v(h)) * torch.sigmoid(self.att_u(h)))  # [B,R,W,12]
        # SLOTS6 order is Sag(fs1,fs0), Cor(fs1,fs0), Ax(fs1,fs0).
        # Tie each contrast pair to its existing plane bias: no new parameters,
        # sequence embeddings or checkpoint format changes in the N3a bridge.
        bias = self.plane_bias if mask.shape[1] == 3 else self.plane_bias.repeat_interleave(2, dim=0)
        a = (a.float() + bias[None, :, None]).masked_fill(~mask[..., None], float("-inf"))
        a = a.flatten(1, 2).softmax(1)                                       # [B,R*W,12]
        pooled = torch.einsum("bnf,bnd->bfd", a, h.flatten(1, 2).float())    # [B,12,512]
        return (pooled * self.head.weight.float()).sum(-1) + self.head.bias.float()
