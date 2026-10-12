import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
import torch
from torch import nn

from p02_model import P02


class StubEncoder(nn.Module):
    def forward(self, x):
        return x[:, :, 0, 0]


def model():
    p = P02.__new__(P02)
    nn.Module.__init__(p)
    p.encoder = StubEncoder()
    p.head = nn.Linear(3, 2, bias=False)
    with torch.no_grad():
        p.head.weight.copy_(torch.tensor([[1., 0., 0.], [0., 1., 0.]]))
    return p


def test_target_specific_max_padding_and_permutation():
    p = model()
    x = torch.zeros(1, 1, 3, 3, 1, 1, requires_grad=True)
    with torch.no_grad():
        x[0, 0, 0, :, 0, 0] = torch.tensor([5., 1., 0.])
        x[0, 0, 1, :, 0, 0] = torch.tensor([2., 7., 0.])
        x[0, 0, 2, :, 0, 0] = torch.tensor([999., 999., 0.])
    mask = torch.tensor([[[True, True, False]]])
    y = p(x, mask)
    torch.testing.assert_close(y, torch.tensor([[5., 7.]]))
    torch.testing.assert_close(p(x[:, :, [1, 0, 2]], mask), y)
    y.sum().backward()
    assert x.grad[0, 0, 0, 0, 0, 0] == 1
    assert x.grad[0, 0, 1, 1, 0, 0] == 1
    assert x.grad[0, 0, 2].abs().sum() == 0


def test_rejects_empty_and_nonfinite():
    p = model()
    x = torch.zeros(1, 1, 1, 3, 1, 1)
    with pytest.raises(ValueError):
        p(x, torch.zeros(1, 1, 1, dtype=torch.bool))
    x[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError):
        p(x, torch.ones(1, 1, 1, dtype=torch.bool))
