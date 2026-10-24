"""Test the actual generated blend function without model, data or label access."""
import ast
import json
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from scipy.special import ndtr

ROOT = Path(__file__).resolve().parents[1]


class GaussianRankTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        nb = json.loads((ROOT / 'notebooks/f6_gaussian_rank/f6.ipynb').read_text())
        tree = ast.parse(''.join(nb['cells'][12]['source']))
        funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'pct']
        scope = {'np': np, 'pd': pd}
        exec(compile(ast.Module(body=funcs, type_ignores=[]), '<F6 generated function>', 'exec'), scope)
        cls.transform = staticmethod(scope['pct'])

    def test_endpoints_ties_and_singleton(self):
        self.assertEqual(self.transform([2])[0], 0.)
        np.testing.assert_array_equal(self.transform([2, 2, 2]), np.zeros(3))
        z = self.transform(np.arange(10000))
        self.assertTrue(np.isfinite(z).all())
        self.assertTrue((np.diff(z) > 0).all())
        np.testing.assert_allclose(z, -z[::-1], atol=1e-12)

    def test_monotonic_input_and_row_permutation(self):
        x = np.array([4., 1., 3., 3., 5.])
        np.testing.assert_allclose(self.transform(x), self.transform(np.exp(x)))
        order = np.array([3, 1, 4, 0, 2])
        np.testing.assert_allclose(self.transform(x)[order], self.transform(x[order]))

    def test_changes_ensemble_order_and_stays_bounded(self):
        rng = np.random.default_rng(0)
        a, b = rng.normal(size=(2, 100))
        old = .5 * pd.Series(a).rank().to_numpy() + .5 * pd.Series(b).rank().to_numpy()
        new = ndtr(.5 * self.transform(a) + .5 * self.transform(b))
        self.assertFalse(np.array_equal(np.argsort(old), np.argsort(new)))
        self.assertTrue(((new > 0) & (new < 1)).all())


if __name__ == '__main__':
    unittest.main()
