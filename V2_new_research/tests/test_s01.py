"""Synthetic CPU checks for S01 multi-series coverage. Run: python -m pytest tests/test_s01.py -q"""
import ast
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

torch = pytest.importorskip("torch")
import m00_data as D  # noqa: E402
import s01_data as S  # noqa: E402


def row(uid, plane, fluid=0, fat=0):
    return {"SeriesInstanceUID": uid, "Anatomical_Plane": plane, "Fluid_Sensitive": fluid, "Fat_Suppression": fat}


def vol(n=10, seed=0):
    return np.random.default_rng(seed).random((n, 32, 32), dtype=np.float32), np.ones(n, bool)


def test_selection_ranks_dedupes_and_matches_m00_first_choice():
    rows = [row("3", "sagittal"), row("2", "Sagittal", fat=1), row("9", "Sagittal", fluid=1),
            row("9", "Sagittal", fluid=1), row("5", "Coronal"), row("7", "Other")]
    sel = S.select_series_multi(rows)
    assert sel == {"Sagittal": ["9", "2"], "Coronal": ["5"], "Axial": []}
    assert {r: (v[0] if v else None) for r, v in sel.items()} == D.select_series(rows)


def test_uid_across_planes_fails_study():
    with pytest.raises(D.StudyError):
        S.select_series_multi([row("1", "Sagittal"), row("1", "Axial")])


def test_budget_split():
    assert S.split_budget(4, 2) == [2, 2] and S.split_budget(16, 2) == [8, 8]
    assert S.split_budget(4, 1) == [4] and S.split_budget(16, 1) == [16] and S.split_budget(4, 0) == []


@pytest.mark.parametrize("train", [True, False])
def test_shapes_unchanged_and_windows_from_each_series(train):
    a, b = vol(seed=1), vol(seed=2)
    rng = np.random.default_rng(0) if train else None
    x, m = S.build_study_multi({"Sagittal": [a, b], "Coronal": [a]}, train, rng)
    n = D.TRAIN_WINDOWS if train else D.INFER_WINDOWS
    assert x.shape == (3, n, 3, D.SIZE, D.SIZE) and m.shape == (3, n)
    half = n // 2
    xa, _ = D.series_windows(*a, half, np.random.default_rng(0) if train else None)
    assert torch.equal(x[0, :half], xa) and not torch.equal(x[0, half:], xa)
    assert m[0].all() and m[1].all() and not m[2].any()


def test_validation_deterministic_and_single_series_matches_m00():
    a, b = vol(seed=1), vol(seed=2)
    x1, _ = S.build_study_multi({"Axial": [a, b]}, False)
    x2, _ = S.build_study_multi({"Axial": [a, b]}, False)
    assert torch.equal(x1, x2)
    xs, ms = S.build_study_multi({"Axial": [a]}, False)
    xo, mo = D.build_study({"Axial": a}, False)
    assert torch.equal(xs, xo) and torch.equal(ms, mo)


def test_train_seeded():
    a, b = vol(30, 1), vol(30, 2)
    run = lambda s: S.build_study_multi({"Sagittal": [a, b]}, True, np.random.default_rng(s))[0]
    assert torch.equal(run(5), run(5))


def test_no_valid_window_raises():
    with pytest.raises(D.StudyError):
        S.build_study_multi({"Sagittal": []}, False)


def test_generated_sources_compile_and_isolated():
    names = ("s01_full_train/train.py", "s01_full_train/train_template.py",
             "s01_cpu_check/check.py", "s01_gpu_pilot/pilot.py")
    for name in names:
        path = ROOT / "notebooks" / name
        if not path.exists():
            pytest.skip("run src/research/build_s01.py first")
        text = path.read_text(encoding="utf-8")
        ast.parse(text)
        assert "p00_best.pt" not in text and "p00_full_train_summary" not in text
        assert "p00_gpu_pilot_summary" not in text and "train.csv\")" not in text
    train = (ROOT / "notebooks/s01_full_train/train.py").read_text(encoding="utf-8")
    assert "select_series_multi(self.by_study" in train and "build_study_multi(roles" in train
    assert "HARD_CAP_S = 8 * 3600.0" in train and "MAX_EPOCHS, LR = 0, 4, 2, 1, 3, 3e-4" in train
