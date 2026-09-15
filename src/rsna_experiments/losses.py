"""Masked per-target objectives; validation always uses the same plain BCE."""
import torch
from torch import nn
from torch.nn import functional as F


class MaskedObjective(nn.Module):
    def __init__(self, cfg, train_y, train_w):
        super().__init__()
        self.cfg = cfg
        active = train_w.sum(0) > 0
        if not active.any():
            raise ValueError('No active labels in training')
        self.register_buffer('active', active)
        # Fixed training-set denominators avoid changing target scale with sparse minibatches.
        self.register_buffer('mean_weight', train_w.mean(0).clamp_min(1e-8))
        pos = (train_y*train_w).sum(0)
        neg = ((1-train_y)*train_w).sum(0)
        self.register_buffer('pos_weight', ((neg+1)/(pos+1)).clamp(.25, 4))
        if cfg['loss'] == 'focal':
            from monai.losses import FocalLoss
            self.focal = FocalLoss(gamma=cfg['gamma'], reduction='none', use_softmax=False)

    def forward(self, logits, y, w):
        logits = logits.float()
        kind = self.cfg['loss']
        if kind == 'focal':
            raw = self.focal(logits, y)
        elif kind == 'asl':
            p = logits.sigmoid()
            pn = (1-p+self.cfg['asl_clip']).clamp(max=1)
            raw = -(y*(1-p).pow(self.cfg['gamma_pos'])*p.clamp_min(1e-8).log()
                    + (1-y)*(1-pn).pow(self.cfg['gamma_neg'])*pn.clamp_min(1e-8).log())
        else:
            target = y*(1-self.cfg['label_smoothing'])+.5*self.cfg['label_smoothing'] if kind == 'smooth' else y
            raw = F.binary_cross_entropy_with_logits(logits, target, reduction='none',
                        pos_weight=self.pos_weight if kind == 'pos_weight' else None)
        loss = ((raw*w).mean(0)/self.mean_weight)[self.active].mean()
        if kind == 'bce_rank':
            pairs = []
            for j in range(y.shape[1]):
                positive = logits[(y[:, j] == 1) & (w[:, j] > 0), j]
                negative = logits[(y[:, j] == 0) & (w[:, j] > 0), j]
                if positive.numel() and negative.numel():
                    pairs.append(F.softplus(-(positive[:, None]-negative[None, :])).mean())
            if pairs:
                loss = loss + self.cfg['ranking_weight']*torch.stack(pairs).mean()
        return loss
