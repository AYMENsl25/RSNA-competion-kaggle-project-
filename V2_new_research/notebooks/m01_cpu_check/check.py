"""M00: random-init ResNet18 shared encoder, shared max over valid windows (all roles jointly),
one linear head -> 12 logits. No pretrained weights, no network access."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
           "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]


class M00(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = resnet18(weights=None)
        self.encoder.fc = nn.Identity()
        self.head = nn.Linear(512, len(TARGETS))

    def forward(self, x, mask):
        """x [B,R,W,3,H,W2] float, mask [B,R,W] bool -> logits [B,12]. Only valid windows are
        encoded, so padded windows cannot influence features, BatchNorm stats or the max."""
        if x.ndim != 6 or x.shape[3] != 3 or mask.dtype != torch.bool or mask.shape != x.shape[:3]:
            raise ValueError(f"bad shapes x={tuple(x.shape)} mask={tuple(mask.shape)} {mask.dtype}")
        has = mask.flatten(1).any(1)
        if not has.all():
            raise ValueError(f"studies with no valid window (batch idx {(~has).nonzero().flatten().tolist()})")
        sel = x[mask]
        if not torch.isfinite(sel).all():
            raise ValueError("nonfinite pixels in a valid window")
        feats = self.encoder(sel)                                   # [n,512]
        b = mask.flatten(1).nonzero()[:, 0]                         # study index per valid window
        pooled = feats.new_full((x.shape[0], feats.shape[1]), float("-inf"))
        pooled = pooled.scatter_reduce(0, b[:, None].expand_as(feats), feats, "amax", include_self=True)
        return self.head(pooled)


@torch.no_grad()
def predict_proba(model, x, mask):
    """Finite [B,12] probabilities in [0,1] or a clear error."""
    model.eval()
    p = torch.sigmoid(model(x, mask))
    if not torch.isfinite(p).all():
        raise FloatingPointError("nonfinite probabilities")
    return p


def masked_bce(logits, targets, observed):
    """Mean BCE over observed cells only. observed bool [B,12]; unobserved targets ignored."""
    if logits.shape != targets.shape or logits.shape != observed.shape or observed.dtype != torch.bool:
        raise ValueError("logits, targets and boolean observed mask must have identical shapes")
    if not observed.any():
        raise ValueError("no observed targets in batch")
    selected_logits, selected_targets = logits[observed], targets[observed].float()
    if not torch.isfinite(selected_logits).all() or not torch.isfinite(selected_targets).all():
        raise ValueError("nonfinite observed logits or targets")
    return F.binary_cross_entropy_with_logits(selected_logits, selected_targets)

"""M01: M00 matched comparison. Same random-init ResNet18 shared encoder and input encoding, but the
shared max + linear head is replaced by 12 finding-specific learned attention queries over valid
windows (all roles jointly); each finding reads its own pooled vector -> one logit per target.
TARGETS, masked_bce and predict_proba are reused unchanged from M00."""
import torch
import torch.nn as nn
from torchvision.models import resnet18




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


import json
torch.set_num_threads(2)
torch.manual_seed(0)
a = M00().state_dict()
torch.manual_seed(0)
net = M01().eval()
b = net.state_dict()
assert set(b) - set(a) == {"queries"}
assert all(torch.equal(a[k], b[k]) for k in a)
with torch.no_grad(): net.queries.normal_()
g = torch.Generator().manual_seed(1)
x = torch.rand(2, 3, 4, 3, 32, 32, generator=g)
mask = torch.zeros(2, 3, 4, dtype=torch.bool)
mask[0, 0, :3] = True
mask[1, 1, :2] = True
mask[1, 2, :4] = True
ref = predict_proba(net, x, mask)
x_pad = x.clone(); x_pad[~mask] = float("nan")
assert torch.equal(ref, predict_proba(net, x_pad, mask))
with torch.no_grad():
    for i in range(2):
        f = net.encoder(x[i][mask[i]])
        weight = (net.queries @ f.T / 512 ** 0.5).softmax(-1)
        expected = torch.sigmoid(((weight @ f) * net.head.weight).sum(-1) + net.head.bias)
        assert torch.allclose(expected, ref[i], atol=1e-5)
solo = torch.cat([predict_proba(net, x[i:i+1], mask[i:i+1]) for i in range(2)])
assert torch.allclose(ref, solo, atol=1e-5)
x_other = x.clone(); x_other[1] = torch.rand_like(x_other[1])
assert torch.allclose(ref[0], predict_proba(net, x_other, mask)[0], atol=1e-5)
bad_mask = mask.clone(); bad_mask[1] = False
try: net(x, bad_mask); raise AssertionError("empty study accepted")
except ValueError: pass
torch.manual_seed(0)
train_model = M01().train()
logits = train_model(x, mask)
assert logits.shape == (2, 12)
loss = masked_bce(logits, torch.ones(2, 12), torch.ones(2, 12, dtype=torch.bool))
loss.backward()
assert torch.isfinite(loss) and train_model.queries.grad.abs().sum() > 0
assert all(torch.isfinite(p.grad).all() for p in train_model.parameters() if p.grad is not None)
print(json.dumps({"status": "passed", "attention_mask": True, "batch_isolation": True,
                  "manual_attention": True, "finite_gradients": True, "init_match": True}))
