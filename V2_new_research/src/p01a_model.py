"""P01A: official ImageNet CoAtNet-0 weights, with M00's exact shared-max forward."""

import hashlib
from pathlib import Path

import torch
import torch.nn as nn

from m00_model import M00, TARGETS

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
