"""P00: M00 architecture with strictly verified ImageNet-1K ResNet18 initialization."""
from pathlib import Path
import hashlib
import torch
import torch.nn as nn

from torchvision.models import resnet18
from m00_model import M00

WEIGHT_FILE = "resnet18-f37072fd.pth"
WEIGHT_SHA256 = "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
WEIGHT_DIRS = (
    Path("/kaggle/input/rsna-v2-official-resnet18-imagenet-v1"),
    Path("/kaggle/input/datasets/aymenslimani/rsna-v2-official-resnet18-imagenet-v1"),
)


def verified_pretrained_encoder():
    hits = {p.resolve() for d in WEIGHT_DIRS if d.is_dir() for p in d.rglob(WEIGHT_FILE)}
    if len(hits) != 1:
        raise FileNotFoundError(f"expected exactly one {WEIGHT_FILE}, got {len(hits)}")
    path = hits.pop()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    if digest.hexdigest() != WEIGHT_SHA256:
        raise ValueError("pretrained ResNet18 weight SHA256 mismatch")
    # The source file is torchvision ResNet18_Weights.IMAGENET1K_V1.
    weights = torch.load(path, map_location="cpu", weights_only=True)
    encoder = resnet18(weights=None)
    encoder.load_state_dict(weights, strict=True)
    encoder.fc = nn.Identity()
    return encoder


class P00(M00):
    def __init__(self):
        super().__init__()
        self.encoder = verified_pretrained_encoder()
