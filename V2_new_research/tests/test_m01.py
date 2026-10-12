"""Synthetic CPU checks for M01 masked attention. Run: python -m pytest tests/test_m01.py -q"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

import m00_model as M00  # noqa: E402
import m01_model as M  # noqa: E402


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    net = M.M01().eval()
    with torch.no_grad():
        net.queries.normal_()          # sharp, query-specific attention so masking errors show up
    return net


def batch():
    """Two studies [2,R=3,W=4,3,32,32] with different valid-window patterns."""
    g = torch.Generator().manual_seed(1)
    x = torch.rand(2, 3, 4, 3, 32, 32, generator=g)
    m = torch.zeros(2, 3, 4, dtype=torch.bool)
    m[0, 0, :3] = True                 # study 0: one role, 3 windows
    m[1, 1, :2] = m[1, 2, :4] = True   # study 1: two roles, 6 windows
    return x, m


def test_reuses_m00_loss_and_proba():
    assert M.masked_bce is M00.masked_bce and M.predict_proba is M00.predict_proba and M.TARGETS is M00.TARGETS


def test_init_matches_m00_except_queries():
    torch.manual_seed(0)
    a = M00.M00().state_dict()
    torch.manual_seed(0)
    b = M.M01().state_dict()
    assert set(b) - set(a) == {"queries"} and all(torch.equal(a[k], b[k]) for k in a)


def test_padded_windows_ignored_and_matches_manual_attention(model):
    x, m = batch()
    ref = M.predict_proba(model, x, m)
    x2 = x.clone()
    x2[~m] = float("nan")              # garbage / nonfinite in padded slots
    assert torch.equal(ref, M.predict_proba(model, x2, m))
    with torch.no_grad():
        for i in range(2):
            f = model.encoder(x[i][m[i]])                                       # [n,512]
            a = (model.queries @ f.T / 512 ** 0.5).softmax(-1)                  # [12,n]
            expect = torch.sigmoid(((a @ f) * model.head.weight).sum(-1) + model.head.bias)
            assert torch.allclose(expect, ref[i], atol=1e-5)


def test_batch_isolation(model):
    x, m = batch()
    both = M.predict_proba(model, x, m)
    solo = torch.cat([M.predict_proba(model, x[i:i + 1], m[i:i + 1]) for i in range(2)])
    assert torch.allclose(both, solo, atol=1e-5)
    x2 = x.clone()
    x2[1] = torch.rand_like(x2[1])     # changing study 1 must not move study 0
    assert torch.allclose(both[0], M.predict_proba(model, x2, m)[0], atol=1e-5)


def test_rejects_bad_inputs(model):
    x, m = batch()
    m0 = m.clone()
    m0[1] = False
    with pytest.raises(ValueError, match="no valid window"):
        model(x, m0)
    x2 = x.clone()
    x2[0, 0, 0, 0, 0, 0] = float("inf")
    with pytest.raises(ValueError, match="nonfinite"):
        model(x2, m)
    with pytest.raises(ValueError):
        model(x, m.int())


def test_train_step_finite_grads():
    torch.manual_seed(0)
    net = M.M01().train()
    x, m = batch()
    logits = net(x, m)
    assert logits.shape == (2, 12)
    loss = M.masked_bce(logits, torch.ones(2, 12), torch.ones(2, 12, dtype=torch.bool))
    loss.backward()
    assert torch.isfinite(loss) and net.queries.grad is not None and net.queries.grad.abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)
