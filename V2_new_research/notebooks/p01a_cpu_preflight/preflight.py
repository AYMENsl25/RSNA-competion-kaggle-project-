"""P01A offline CPU preflight; synthetic pixels only, no patient images or labels."""
import json
import time
from pathlib import Path

import torch

# Embedded m00_model.py SHA256: e8410ff5ebcfcf2036c5da18538fc194fe6db24eb88f274981bfba9de3ed531e
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

# Embedded p01a_model.py SHA256: 44d9eeb68390fb8eee461cae24496fd125b59fa09d8db663da7136154f9609bc
"""P01A: official ImageNet CoAtNet-0 weights, with M00's exact shared-max forward."""

import hashlib
from pathlib import Path

import torch
import torch.nn as nn


TIMM_NAME = "coatnet_0_rw_224.sw_in1k"
FEATURE_DIM = 768
WEIGHT_FILE = "coatnet_0_rw_224_sw_in1k.safetensors"
WEIGHT_SHA256 = "efe01c8c139a4b0c360f508d10ae1217f7e3f3431fd85824438ed1110773366c"
WEIGHT_PATHS = (
    Path("/kaggle/input/rsna-v2-official-coatnet0-imagenet1k") / WEIGHT_FILE,
    Path("/kaggle/input/datasets/aymenslimani/rsna-v2-official-coatnet0-imagenet1k") / WEIGHT_FILE,
)
CLASSIFIER_SHAPES = {"head.fc.weight": (1000, FEATURE_DIM), "head.fc.bias": (1000,)}


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verified_pretrained_encoder(path=None, expected_sha256=WEIGHT_SHA256):
    """Load local safetensors only; reject any state mismatch beyond 1000-way classifier."""
    import timm
    from safetensors.torch import load_file

    if path is None:
        path = next((p for p in WEIGHT_PATHS if p.is_file()), None)
        if path is None:
            raise FileNotFoundError(f"P01A weights absent from {WEIGHT_PATHS}")
    path = Path(path)
    if _sha256(path) != expected_sha256:
        raise ValueError("P01A weight SHA256 mismatch")
    model = timm.create_model(TIMM_NAME, pretrained=False, num_classes=0)
    if model.num_features != FEATURE_DIM:
        raise ValueError(f"unexpected CoAtNet feature dim {model.num_features}")
    state = load_file(str(path), device="cpu")
    for key, shape in CLASSIFIER_SHAPES.items():
        if key not in state or tuple(state[key].shape) != shape:
            raise ValueError(f"missing or invalid classifier: {key}")
    state = {k: v for k, v in state.items() if k not in CLASSIFIER_SHAPES}
    model.load_state_dict(state, strict=True)
    return model


class P01A(M00):
    """Only encoder and linear-head size differ from M00; forward is inherited verbatim."""

    def __init__(self, weight_path=None, expected_sha256=WEIGHT_SHA256):
        nn.Module.__init__(self)
        self.encoder = verified_pretrained_encoder(weight_path, expected_sha256)
        self.head = nn.Linear(FEATURE_DIM, len(TARGETS))


def main():
    t0 = time.perf_counter()
    import timm
    import safetensors
    torch.set_num_threads(2)
    model = P01A()
    x = torch.rand(1, 1, 2, 3, 224, 224)
    mask = torch.tensor([[[True, False]]])
    x[:, :, 1] = float('nan')
    model.eval()
    with torch.no_grad():
        y1 = model(x, mask)
        x[:, :, 1] = 1.0
        y2 = model(x, mask)
    assert y1.shape == (1, 12) and torch.isfinite(y1).all()
    assert torch.equal(y1, y2), 'padded window changed logits'
    model.train()
    y = model(x, mask)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(y, torch.zeros_like(y))
    loss.backward()
    assert model.head.weight.grad is not None and torch.isfinite(model.head.weight.grad).all()
    result = {
        'status': 'passed', 'checkpoint_sha256': WEIGHT_SHA256,
        'model': TIMM_NAME, 'torch': torch.__version__, 'timm': timm.__version__,
        'safetensors': safetensors.__version__, 'elapsed_s': time.perf_counter() - t0,
        'logits_shape': list(y.shape), 'parameter_count': sum(p.numel() for p in model.parameters()),
    }
    Path('/kaggle/working/p01a_cpu_preflight.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result))

if __name__ == '__main__':
    main()
