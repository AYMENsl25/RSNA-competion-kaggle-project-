import importlib.util
from pathlib import Path
import unittest
import torch

spec = importlib.util.spec_from_file_location('f4', Path(__file__).resolve().parents[1] / 'notebooks/f4_cpu_trials/run.py')
f4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(f4)


class TrialsTest(unittest.TestCase):
    def test_ranking_order_and_mask(self):
        y = torch.tensor([[1.], [0.], [1.]])
        mask = torch.tensor([[True], [True], [False]])
        good = torch.tensor([[3.], [-3.], [-1000.]], requires_grad=True)
        bad = -good.detach()
        self.assertLess(f4.ranking_loss(good, y, mask), f4.ranking_loss(bad, y, mask))
        f4.ranking_loss(good, y, mask).backward()
        self.assertEqual(good.grad[2].item(), 0)
        self.assertTrue(torch.isfinite(good.grad).all())

    def test_no_supported_pairs(self):
        z = torch.ones(3, 2, requires_grad=True)
        loss = f4.ranking_loss(z, torch.ones_like(z), torch.ones_like(z, dtype=torch.bool))
        self.assertEqual(loss.item(), 0)
        loss.backward()

    def test_dropout_retains_whole_available_series(self):
        valid = torch.ones(20, 6, 8, dtype=torch.bool)
        valid[:, 2] = False
        a = f4.drop_series(valid, 1., torch.Generator().manual_seed(1))
        b = f4.drop_series(valid, 1., torch.Generator().manual_seed(1))
        self.assertTrue(torch.equal(a, b))
        self.assertTrue((a.sum((1, 2)) == 8).all())
        self.assertFalse(a[:, 2].any())
        self.assertTrue(torch.equal(valid, f4.drop_series(valid, 0., torch.Generator())))


if __name__ == '__main__':
    unittest.main()
