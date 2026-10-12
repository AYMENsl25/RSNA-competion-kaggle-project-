"""Synthetic CPU smoke for M00. Run: python -m pytest tests -q  (needs torch, torchvision, numpy, pydicom)."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.append(str(ROOT / ".tools"))  # vendored pydicom fallback

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
pydicom = pytest.importorskip("pydicom")
from pydicom.dataset import Dataset, FileMetaDataset  # noqa: E402
from pydicom.uid import ExplicitVRLittleEndian, generate_uid  # noqa: E402

import m00_data as D  # noqa: E402
import m00_model as M  # noqa: E402


def make_ds(arr, ipp=None, iop=(1, 0, 0, 0, 1, 0), inst=1, mono1=False, slope=None, icpt=None):
    ds = Dataset()
    ds.file_meta = FileMetaDataset()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.SOPInstanceUID = generate_uid()
    ds.is_little_endian, ds.is_implicit_VR = True, False
    ds.InstanceNumber = inst
    ds.Rows, ds.Columns = arr.shape
    ds.PixelSpacing = [0.5, 0.5]
    ds.SamplesPerPixel, ds.BitsAllocated, ds.BitsStored, ds.HighBit = 1, 16, 16, 15
    ds.PixelRepresentation = 0
    ds.PhotometricInterpretation = "MONOCHROME1" if mono1 else "MONOCHROME2"
    if ipp is not None:
        ds.ImagePositionPatient, ds.ImageOrientationPatient = list(ipp), list(iop)
    if slope is not None:
        ds.RescaleSlope, ds.RescaleIntercept = slope, icpt
    ds.PixelData = arr.astype(np.uint16).tobytes()
    return ds


def vol_datasets(n=8, hw=(40, 32), seed=0, **kw):
    rng = np.random.default_rng(seed)
    return [make_ds(rng.integers(0, 1000, hw), ipp=(0, 0, float(i)), inst=i + 1, **kw) for i in range(n)]


# ---------- series selection ----------
def test_select_series_deterministic_missing_multiple():
    rows = [
        dict(SeriesInstanceUID="b", Fluid_Sensitive=0, Fat_Suppression=1, Anatomical_Plane="Sagittal"),
        dict(SeriesInstanceUID="c", Fluid_Sensitive=1, Fat_Suppression=0, Anatomical_Plane="Sagittal"),
        dict(SeriesInstanceUID="a", Fluid_Sensitive=1, Fat_Suppression=0, Anatomical_Plane="Sagittal"),
        dict(SeriesInstanceUID="z", Fluid_Sensitive=1, Fat_Suppression=1, Anatomical_Plane="Axial"),
        dict(SeriesInstanceUID="o", Fluid_Sensitive=1, Fat_Suppression=1, Anatomical_Plane="Oblique"),
    ]
    out = D.select_series(rows)
    assert out == {"Sagittal": "a", "Coronal": None, "Axial": "z"}
    assert D.select_series(rows[::-1]) == out


# ---------- geometry ----------
def test_order_uses_geometry_not_instance_or_filename():
    dss = vol_datasets(6)
    perm = [3, 0, 5, 1, 4, 2]
    shuffled = [dss[i] for i in perm]
    for d in shuffled:  # make InstanceNumber misleading (reverse of anatomy)
        d.InstanceNumber = 100 - int(d.ImagePositionPatient[2])
    got = [float(d.ImagePositionPatient[2]) for d in D.order_slices(shuffled)]
    assert got == sorted(got)


def test_order_tilted_normal():
    # axial-like orientation: row=y, col=z -> normal = y x z = +x ; position must project on x
    iop = (0, 1, 0, 0, 0, 1)
    dss = [make_ds(np.ones((8, 8)) * (i + 1), ipp=(10 - i, 5, 5), iop=iop, inst=i) for i in range(5)]
    got = [d.ImagePositionPatient[0] for d in D.order_slices(dss)]
    assert got == sorted(got)


def test_fallback_only_when_all_tags_missing():
    a = [make_ds(np.ones((4, 4)) * i, inst=i + 1) for i in (3, 1, 2)]
    assert [int(d.InstanceNumber) for d in D.order_slices(a)] == [2, 3, 4]
    mixed = vol_datasets(3)
    del mixed[1].ImagePositionPatient
    with pytest.raises(D.SeriesError):
        D.order_slices(mixed)
    dup = [make_ds(np.ones((4, 4)), inst=1), make_ds(np.ones((4, 4)), inst=1)]
    with pytest.raises(D.SeriesError):
        D.order_slices(dup)


def test_duplicate_position_and_bad_orientation_rejected():
    dss = vol_datasets(4)
    dss[2].ImagePositionPatient = [0, 0, 1.0]
    with pytest.raises(D.SeriesError):
        D.order_slices(dss)
    bad = vol_datasets(3)
    bad[0].ImageOrientationPatient = [1, 0, 0, 1, 0, 0]
    with pytest.raises(D.SeriesError):
        D.order_slices(bad)


def test_irregular_gap_spacing_and_in_plane_rotation_rejected():
    gap = vol_datasets(5)
    gap[-1].ImagePositionPatient = [0, 0, 10]
    with pytest.raises(D.SeriesError, match="irregular slice gap"):
        D.order_slices(gap)
    spacing = vol_datasets(5)
    spacing[-1].PixelSpacing = [0.6, 0.5]
    with pytest.raises(D.SeriesError, match="pixel spacing"):
        D.order_slices(spacing)
    rotated = vol_datasets(5)
    rotated[-1].ImageOrientationPatient = [0, 1, 0, -1, 0, 0]
    with pytest.raises(D.SeriesError, match="in-plane orientation"):
        D.order_slices(rotated)


# ---------- pixels ----------
def test_rescale_and_monochrome1_and_scaling_range():
    base = np.arange(64).reshape(8, 8) * 10
    d2 = make_ds(base, ipp=(0, 0, 0), slope=2.0, icpt=-5.0)
    np.testing.assert_allclose(D.slice_pixels(d2), base * 2.0 - 5.0)
    d1 = make_ds(base, ipp=(0, 0, 0), mono1=True)
    np.testing.assert_allclose(D.slice_pixels(d1), -base.astype(np.float32))
    vol, ok = D.load_series(vol_datasets(6))
    assert ok.all() and vol.min() >= 0 and vol.max() <= 1 and vol.dtype == np.float32


def test_constant_and_nonfinite_slices_flagged(monkeypatch):
    dss = vol_datasets(7)
    dss[2].PixelData = np.full((40, 32), 7, np.uint16).tobytes()  # constant
    vol, ok = D.load_series(dss)
    assert not ok[2] and ok.sum() == 6 and (vol[2] == 0).all()
    centers = D.window_centers(ok, 16)
    assert not {2, 3}.intersection(centers.tolist()) and 1 not in centers  # windows touching slice 2
    # nonfinite pixels in one slice
    dss = vol_datasets(5)
    real = D.slice_pixels
    nan_for = id(dss[1])
    monkeypatch.setattr(D, "slice_pixels", lambda d: (lambda a: (a.__setitem__((0, 0), np.nan), a)[1])(real(d)) if id(d) == nan_for else real(d))
    assert not D.load_series(dss)[1][1]
    # too few usable slices
    with pytest.raises(D.SeriesError):
        D.load_series(vol_datasets(2))


def test_windows_are_adjacent_triplets_and_deterministic():
    n = 20
    vol = np.stack([np.full((10, 10), i / n, np.float32) for i in range(n)])
    ok = np.ones(n, bool)
    x, m = D.series_windows(vol, ok, 16)
    assert x.shape == (16, 3, 224, 224) and m.sum() == 16
    for i in range(16):
        v = x[i, :, 0, 0] * n
        assert torch.allclose(v[1] - v[0], torch.tensor(1.0), atol=1e-3) and torch.allclose(v[2] - v[1], torch.tensor(1.0), atol=1e-3)
    x2, _ = D.series_windows(vol, ok, 16)
    assert torch.equal(x, x2)
    # short series: padding masked
    _, m = D.series_windows(vol[:5], ok[:5], 16)
    assert m.sum() == 3 and not m[3:].any()
    # training sampling: seeded, distinct
    a = D.window_centers(ok, 4, np.random.default_rng(1))
    assert len(set(a.tolist())) == 4 and (a == D.window_centers(ok, 4, np.random.default_rng(1))).all()


# ---------- study assembly + model ----------
def fake_study(roles=("Sagittal", "Coronal", "Axial"), n=6):
    out = {}
    for r in roles:
        out[r] = D.load_series(vol_datasets(n, seed=hash(r) % 100))
    return out


def test_study_all_roles_absent_fails():
    with pytest.raises(D.StudyError):
        D.build_study({}, train=False)
    short = {"Axial": D.load_series(vol_datasets(3))}
    short["Axial"][1][:] = False  # no usable slices -> no windows
    with pytest.raises(D.StudyError):
        D.build_study(short, train=True, rng=np.random.default_rng(0))


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return M.M00().eval()


def test_model_shapes_finite_and_missing_role(model):
    x, m = D.build_study(fake_study(("Sagittal", "Axial")), train=False)
    assert x.shape == (3, 16, 3, 224, 224) and not m[1].any() and m[0].any()
    p = M.predict_proba(model, x[None], m[None])
    assert p.shape == (1, 12) and torch.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    xt, mt = D.build_study(fake_study(), train=True, rng=np.random.default_rng(0))
    assert xt.shape == (3, 4, 3, 224, 224)


def test_absent_and_padded_windows_never_affect_output(model):
    x, m = D.build_study(fake_study(("Sagittal", "Coronal")), train=False)
    x, m = x[None], m[None]
    ref = M.predict_proba(model, x, m)
    x2 = x.clone()
    x2[~m] = 1e6                       # garbage (even huge) in masked slots
    x2[0, 2] = torch.randn_like(x2[0, 2]) * 100   # whole absent role filled with noise
    assert torch.equal(ref, M.predict_proba(model, x2, m))
    x3 = x2.clone()
    x3[0, 2, 0, 0, 0, 0] = float("nan")  # nonfinite in masked slot is ignored too
    assert torch.equal(ref, M.predict_proba(model, x3, m))


def test_max_pool_is_over_valid_windows_and_batch_independent(model):
    a = D.build_study(fake_study(("Sagittal",), 5), train=False)
    b = D.build_study(fake_study(("Axial", "Coronal"), 7), train=False)
    x, m = torch.stack([a[0], b[0]]), torch.stack([a[1], b[1]])
    both = M.predict_proba(model, x, m)
    solo = torch.cat([M.predict_proba(model, a[0][None], a[1][None]), M.predict_proba(model, b[0][None], b[1][None])])
    assert torch.allclose(both, solo, atol=1e-5)
    # manual: feature max over valid windows == pooled input of head
    with torch.no_grad():
        f = model.encoder(a[0][a[1]])
        expect = torch.sigmoid(model.head(f.max(0).values))
    assert torch.allclose(expect, solo[0], atol=1e-5)


def test_model_rejects_bad_inputs(model):
    x = torch.zeros(2, 3, 4, 3, 224, 224)
    m = torch.zeros(2, 3, 4, dtype=torch.bool)
    m[0, 0, 0] = True
    with pytest.raises(ValueError, match="no valid window"):
        model(x, m)
    m[1, 1, 1] = True
    x[1, 1, 1, 0, 0, 0] = float("inf")
    with pytest.raises(ValueError, match="nonfinite"):
        model(x, m)
    with pytest.raises(ValueError):
        model(x, m.int())


def test_masked_bce_and_train_step(model):
    logits = torch.tensor([[0.0, 100.0], [0.0, -100.0]])
    tgt = torch.tensor([[1, 0], [0, 1]])
    obs = torch.tensor([[True, False], [True, False]])
    assert abs(M.masked_bce(logits, tgt, obs).item() - np.log(2)) < 1e-6   # wrong 2nd col ignored
    tgt_nan = tgt.float()
    tgt_nan[:, 1] = float("nan")
    assert abs(M.masked_bce(logits, tgt_nan, obs).item() - np.log(2)) < 1e-6
    with pytest.raises(ValueError):
        M.masked_bce(logits, tgt, torch.zeros_like(obs))
    net = M.M00().train()
    x, m = D.build_study(fake_study(), train=True, rng=np.random.default_rng(0))
    loss = M.masked_bce(net(x[None], m[None]), torch.ones(1, 12), torch.ones(1, 12, dtype=torch.bool))
    loss.backward()
    assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)
