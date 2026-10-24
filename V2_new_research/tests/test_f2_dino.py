"""Synthetic CPU contracts for F2; run with the existing V2 .venv Python."""
import csv
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from f2_data import TARGETS, centers_geometry, cohort, select
from f2_model import Head, masked_bce


def row(uid="1", plane="Sagittal", fluid="1", fat="1"):
    return {"series": uid, "plane": plane, "fluid": fluid, "fatsat": fat,
            "ok": "1", "bad_slices": "0", "length": "5", "positions": "0;2;5;9;14",
            "plane_geom": plane}


class DataContracts(unittest.TestCase):
    def test_selection_distinct_and_ranked_without_fallback(self):
        rows = [row("3", fat="0"), row("2"), row("1"), row("4", fluid="0")]
        selected = select(rows)
        self.assertEqual([selected[i]["series"] for i in (0, 1)], ["1", "4"])
        selected[0]["ok"] = "0"
        with self.assertRaisesRegex(ValueError, "QC"):
            centers_geometry(selected[0], 8)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            select([row(), row()])

    def test_geometry_uses_real_gaps_and_no_wrapped_neighbors(self):
        centers, geometry = centers_geometry(row(), 8)
        np.testing.assert_array_equal(centers, [1, 2, 3])
        np.testing.assert_allclose(geometry[:, 1], [.2, .3, .4])
        np.testing.assert_allclose(geometry[:, 2], [.3, .4, .5])
        moved = row()
        moved["positions"] = "100;102;105;109;114"
        np.testing.assert_allclose(centers_geometry(moved, 8)[1], geometry)

    def test_invalid_geometry_rejected(self):
        for positions in ("0;2;2;9;14", "0;2;nan;9;14", "0;2;5"):
            r = row()
            r["positions"] = positions
            with self.assertRaises(ValueError):
                centers_geometry(r, 8)

    def test_excluded_values_never_parsed_and_groups_isolated(self):
        with tempfile.TemporaryDirectory(prefix="f2_test_", dir=ROOT / "tests") as td:
            split, labels = Path(td) / "split.csv", Path(td) / "labels.csv"
            split.write_text("StudyInstanceUID,group_id,partition\na,g1,candidate_train\nb,g2,candidate_validation\nx,g3,excluded\n")
            with labels.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["StudyInstanceUID", *TARGETS])
                writer.writerow(["a", *([.9] * 12)])
                writer.writerow(["b", *([.1] * 12)])
                writer.writerow(["x", *(["DO_NOT_PARSE"] * 12)])
            uids, _, values = cohort(split, labels, strict=False)
            self.assertEqual(uids, ["a", "b"])
            self.assertNotIn("x", values)
            split.write_text(split.read_text().replace("b,g2", "b,g1"))
            with self.assertRaisesRegex(ValueError, "group crosses"):
                cohort(split, labels, strict=False)


class ModelContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def inputs(self):
        torch.manual_seed(0)
        features = torch.randn(2, 6, 4, 384)
        valid = torch.zeros(2, 6, 4, dtype=torch.bool)
        valid[:, :2, :3] = True
        geom = torch.randn(2, 6, 4, 5)
        return features, valid, geom

    def test_both_heads_finite_backward_and_padding_invariant(self):
        for geo in (False, True):
            model = Head(width=32, geometry=geo).eval()
            features, valid, geometry = self.inputs()
            z = model(features, valid, geometry)
            self.assertEqual(z.shape, (2, 12))
            self.assertTrue(torch.isfinite(z).all())
            modified = features.clone()
            modified[~valid] = 1000
            g2 = geometry.clone()
            g2[~valid] = 1000
            torch.testing.assert_close(z, model(modified, valid, g2))
            loss = masked_bce(z, torch.zeros_like(z), torch.ones_like(z, dtype=torch.bool))
            loss.backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_A_ignores_geometry_and_instance_order(self):
        model = Head(width=32).eval()
        f, m, g = self.inputs()
        torch.testing.assert_close(model(f, m, g), model(f.flip(2), m.flip(2), g.flip(2) * 7))

    def test_B_uses_geometry(self):
        model = Head(width=32, geometry=True).eval()
        f, m, g = self.inputs()
        self.assertFalse(torch.allclose(model(f, m, g), model(f, m, g * 3), atol=1e-7))

    def test_empty_study_rejected_and_unobserved_targets_ignored(self):
        f, m, g = self.inputs()
        with self.assertRaisesRegex(ValueError, "empty study"):
            Head(width=32)(f, m & False, g)
        z = torch.tensor([[.2, .5]], requires_grad=True)
        obs = torch.tensor([[True, False]])
        l1 = masked_bce(z, torch.tensor([[0., 0.]]), obs)
        l2 = masked_bce(z, torch.tensor([[0., 1.]]), obs)
        torch.testing.assert_close(l1, l2)

    def test_config_has_no_kaggle_launch_and_bounded_local_budget(self):
        config = json.loads((ROOT / "notebooks/f2_dino_geometry/config.json").read_text())
        self.assertLessEqual(config["hard_wall_seconds"], 7200)
        self.assertLess(config["extraction_budget_seconds"], config["hard_wall_seconds"])
        self.assertGreaterEqual(config["save_reserve_seconds"], 300)

    def test_paired_training_checkpoint_and_comparison_end_to_end(self):
        """Small synthetic CPU integration, including selection and group bootstrap."""
        import time
        spec = importlib.util.spec_from_file_location("f2_runner", ROOT / "notebooks/f2_dino_geometry/run.py")
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        with tempfile.TemporaryDirectory(prefix="f2_test_", dir=ROOT / "tests") as td:
            out = Path(td)
            (out / "encoder.json").write_text('{"kind": "synthetic test only"}')
            split = out / "split.csv"
            uids = [f"synthetic_{i}" for i in range(24)]
            partition = {u: "candidate_train" if i < 16 else "candidate_validation" for i, u in enumerate(uids)}
            split.write_text("StudyInstanceUID,group_id,partition\n" + "".join(
                f"{u},group{i},{partition[u]}\n" for i, u in enumerate(uids)))
            values = {u: (np.full(12, i % 2, np.float32), np.ones(12, bool)) for i, u in enumerate(uids)}
            rng = np.random.default_rng(0)
            features = (rng.normal(size=(24, 6, 2, 384)).astype(np.float16),
                        np.ones((24, 6, 2), bool), np.zeros((24, 6, 2, 5), np.float32))
            cfg = dict(seed=0, head_width=16, head_batch=8, epochs=2, learning_rate=3e-4,
                       weight_decay=.01, hard_wall_seconds=60, save_reserve_seconds=5,
                       bootstrap_draws=12, split=str(split.relative_to(ROOT)))
            prepared = (uids, partition, values, None, None, {"kind": "synthetic only"})
            with contextlib.redirect_stdout(io.StringIO()):
                result = runner.train(cfg, out, prepared, features, time.monotonic())
            for arm in ("A", "B"):
                self.assertTrue((out / f"f2_{arm}_best.pt").is_file())
                self.assertEqual(np.load(out / f"f2_{arm}_val_probabilities.npy").shape, (8, 12))
            self.assertEqual(result["kaggle_gpu_hours_used"], 0)
            self.assertTrue(np.isfinite(result["B_minus_A_macro_auc"]))
            history = json.loads((out / "history.json").read_text())
            self.assertEqual([len(history[a]) for a in ("A", "B")], [2, 2])


if __name__ == "__main__":
    unittest.main(verbosity=2)
