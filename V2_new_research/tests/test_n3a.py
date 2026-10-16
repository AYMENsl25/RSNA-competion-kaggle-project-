"""Six-slot attention: plane mapping, missing slots, gradients and old-checkpoint parity."""
import importlib.util
import sys
import unittest
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from v3_model import V3


def tiny_encoder():
    return nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 512))


class SixSlotAttentionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = V3("attn").eval()
        self.model.encoder = tiny_encoder()
        self.model.plane_bias.data.normal_()
        self.x = torch.randn(2, 3, 4, 3, 8, 8)
        self.mask = torch.ones(2, 3, 4, dtype=torch.bool)

    def test_duplicate_contrasts_preserve_plane_bias_and_logits(self):
        with torch.no_grad():
            a = self.model(self.x, self.mask)
            b = self.model(self.x.repeat_interleave(2, 1), self.mask.repeat_interleave(2, 1))
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)

    def test_missing_contrast_and_nan_padding_are_invisible(self):
        x = self.x.repeat_interleave(2, 1)
        mask = self.mask.repeat_interleave(2, 1)
        mask[:, 1::2] = False
        x[:, 1::2] = float("nan")
        with torch.no_grad():
            torch.testing.assert_close(self.model(x, mask), self.model(self.x, self.mask))

    def test_additional_slots_receive_finite_gradients(self):
        x = self.x.repeat_interleave(2, 1).detach().requires_grad_()
        y = self.model(x, self.mask.repeat_interleave(2, 1))
        y.square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(float(x.grad[:, 1::2].abs().sum()), 0)
        self.assertTrue(torch.isfinite(self.model.plane_bias.grad).all())

    def test_empty_study_and_unknown_slot_count_rejected(self):
        with self.assertRaises(ValueError):
            self.model(self.x, torch.zeros_like(self.mask))
        with self.assertRaises(ValueError):
            self.model(self.x[:, :2], self.mask[:, :2])

    @unittest.skipUnless((ROOT / "outputs/n1_attn_random/source/v3_model.py").exists(), "needs N1 source snapshot")
    def test_three_slot_path_matches_frozen_n1_implementation(self):
        spec = importlib.util.spec_from_file_location("frozen_n1_v3", ROOT / "outputs/n1_attn_random/source/v3_model.py")
        old = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(old)
        reference = old.V3("attn").eval()
        reference.encoder = tiny_encoder()
        reference.load_state_dict(self.model.state_dict(), strict=True)
        with torch.no_grad():
            torch.testing.assert_close(reference(self.x, self.mask), self.model(self.x, self.mask), atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
