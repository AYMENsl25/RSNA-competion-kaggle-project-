"""Synthetic CPU checks for P01A (pretrained timm CoAtNet-0 encoder, M00 pooling/head).
Run: python -m pytest tests/test_p01a.py -q
Uses a fake random-init state dict in the official key layout; no network, no competition data.
Set P01A_WEIGHT_FILE=<path to official model.safetensors> to also check the real file."""
import hashlib
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

torch = pytest.importorskip("torch")
timm = pytest.importorskip("timm")
st = pytest.importorskip("safetensors.torch")

import m00_model as M00  # noqa: E402
import p01a_model as P  # noqa: E402


def write_fixture(path, state):
    st.save_file({k: v.contiguous() for k, v in state.items()}, str(path))
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def fake(tmp_path_factory):
    """Fake 'official' file: full 1000-class timm state dict (backbone + head.fc)."""
    torch.manual_seed(0)
    state = timm.create_model(P.TIMM_NAME, pretrained=False).state_dict()
    path = tmp_path_factory.mktemp("w") / P.WEIGHT_FILE
    return path, write_fixture(path, state), state


def test_strict_load_drops_only_classifier(fake):
    path, sha, state = fake
    enc = P.verified_pretrained_encoder(path, sha)
    got = enc.state_dict()
    assert set(state) - set(got) == set(P.CLASSIFIER_SHAPES) and set(got) <= set(state)
    assert all(torch.equal(got[k], state[k]) for k in got)
    assert enc(torch.rand(2, 3, 224, 224)).shape == (2, P.FEATURE_DIM)


def test_rejects_bad_weight_files(fake, tmp_path):
    path, sha, state = fake
    with pytest.raises(ValueError, match="SHA256"):
        P.verified_pretrained_encoder(path, "0" * 64)
    backbone_key = next(k for k in state if not k.startswith("head."))
    cases = {"missing_backbone": ({k: v for k, v in state.items() if k != backbone_key}, RuntimeError),
             "extra_key": ({**state, "stray.weight": torch.zeros(1)}, RuntimeError),
             "no_classifier": ({k: v for k, v in state.items() if k not in P.CLASSIFIER_SHAPES}, ValueError)}
    for name, (bad, exc) in cases.items():
        p = tmp_path / f"{name}.safetensors"
        with pytest.raises(exc):
            P.verified_pretrained_encoder(p, write_fixture(p, bad))


def test_default_path_never_downloads():
    with pytest.raises(FileNotFoundError):    # only the fixed Kaggle mounts are searched
        P.verified_pretrained_encoder()


def batch():
    """Two studies [2,R=3,W=2,3,224,224]; 4 valid windows total."""
    g = torch.Generator().manual_seed(1)
    x = torch.rand(2, 3, 2, 3, 224, 224, generator=g)
    m = torch.zeros(2, 3, 2, dtype=torch.bool)
    m[0, 0, :2] = True
    m[1, 1, 0] = m[1, 2, 1] = True
    return x, m


def test_model_is_m00_with_swapped_encoder(fake):
    path, sha, _ = fake
    model = P.P01A(path, sha)
    assert P.P01A.forward is M00.M00.forward and P.TARGETS is M00.TARGETS
    assert (model.head.in_features, model.head.out_features) == (P.FEATURE_DIM, 12)
    assert set(model.state_dict()) == {f"encoder.{k}" for k in model.encoder.state_dict()} | {"head.weight", "head.bias"}


def test_padded_windows_ignored(fake):
    path, sha, _ = fake
    model = P.P01A(path, sha)
    x, m = batch()
    x2 = x.clone()
    x2[~m] = float("nan")
    ref = M00.predict_proba(model, x, m)
    assert ref.shape == (2, 12) and torch.equal(ref, M00.predict_proba(model, x2, m))
    model.train()                               # batch-stat layers see only valid windows too
    with torch.no_grad():
        assert torch.equal(model(x, m), model(x2, m))


def test_train_step_shapes_and_finite_grads(fake):
    path, sha, _ = fake
    model = P.P01A(path, sha).train()
    x, m = batch()
    logits = model(x, m)
    assert logits.shape == (2, 12)
    loss = M00.masked_bce(logits, torch.ones(2, 12), torch.ones(2, 12, dtype=torch.bool))
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    first_encoder_param = next(model.encoder.parameters())          # stem conv: gradient reaches the input end
    assert first_encoder_param.grad.abs().sum() > 0 and model.head.weight.grad.abs().sum() > 0


@pytest.mark.skipif(not os.environ.get("P01A_WEIGHT_FILE"), reason="official weight file not provided")
def test_official_weight_file():
    enc = P.verified_pretrained_encoder(os.environ["P01A_WEIGHT_FILE"], P.WEIGHT_SHA256)
    assert enc.eval()(torch.rand(1, 3, 224, 224)).shape == (1, P.FEATURE_DIM)
