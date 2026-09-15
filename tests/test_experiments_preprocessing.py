"""Synthetic cache contract checks; no real-data/model validation implied."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types

import numpy as np
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid
import pytest


@pytest.fixture
def prep(monkeypatch):
    """Use production common.py when present; scoped stand-in otherwise."""
    source = Path(__file__).resolve().parents[1] / "src" / "rsna_experiments"
    package = types.ModuleType("_preprocessing_contract_test")
    package.__path__ = [str(source)]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    if not (source / "common.py").exists():
        common = types.ModuleType(package.__name__ + ".common")
        common.UID = "StudyInstanceUID"
        common.sha256 = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
        common.fingerprint = lambda obj: hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()

        def write_json(path, obj):
            Path(path).write_text(json.dumps(obj, allow_nan=False), encoding="utf-8")

        def locate(filename, roots=None, override=None):
            if override:
                return Path(override)
            return next(Path(root) / filename for root in roots if (Path(root) / filename).exists())

        common.write_json, common.locate = write_json, locate
        monkeypatch.setitem(sys.modules, common.__name__, common)
    name = package.__name__ + ".preprocessing"
    spec = importlib.util.spec_from_file_location(name, source / "preprocessing.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def _csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _series(comp, uid, positions=(4, 0, 2), plane="AX", spacing=(1, 1), shape=(16, 16), description="", series=None):
    series = series or generate_uid()
    folder = comp / "train_series" / uid / series
    folder.mkdir(parents=True)
    frame = generate_uid()
    orientation = {"AX": [1, 0, 0, 0, 1, 0], "SAG": [0, 1, 0, 0, 0, 1], "COR": [0, 0, 1, 1, 0, 0]}[plane]
    for index, position in enumerate(positions):
        meta = FileMetaDataset()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        meta.MediaStorageSOPClassUID = MRImageStorage
        meta.MediaStorageSOPInstanceUID = generate_uid()
        ds = FileDataset(str(folder / f"{index}.dcm"), {}, file_meta=meta, preamble=b"\0" * 128)
        ds.SOPInstanceUID, ds.SOPClassUID = meta.MediaStorageSOPInstanceUID, MRImageStorage
        ds.StudyInstanceUID, ds.SeriesInstanceUID, ds.FrameOfReferenceUID = uid, series, frame
        ds.ImageOrientationPatient = orientation
        ds.ImagePositionPatient = [position if axis == {"SAG": 0, "COR": 1, "AX": 2}[plane] else 0 for axis in range(3)]
        ds.InstanceNumber = 100 - index  # deliberately unrelated to spatial order
        if spacing is not None:
            ds.PixelSpacing = list(spacing)
        ds.Rows, ds.Columns = shape
        ds.SeriesDescription = description
        ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
        ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 0
        pixels = (np.indices(shape)[0] + 100 * position).astype(np.uint16)
        ds.PixelData = pixels.tobytes()
        ds.save_as(folder / f"{index}.dcm", enforce_file_format=True)
    return folder


@pytest.fixture
def case(tmp_path):
    comp = tmp_path / "competition"
    comp.mkdir()
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    uid = generate_uid()
    manifest = prepared / "experiment_manifest.csv"
    _csv(manifest, [{"StudyInstanceUID": uid, "fold": "0", "y_acl": "0.3", "w_acl": "1", "cv_group_id": "001"}])
    (prepared / "preparation_meta.json").write_text(json.dumps(dict(cohort_id="cohort", supervision_id="supervision-1", fold=0, complete=True)), encoding="utf-8")
    return uid, comp, dict(comp_dir=str(comp), manifest_path=str(manifest), out_dir=str(tmp_path / "cache"), slices=5, size=8)


def _archive(out):
    with (out / "cache_index.csv").open(newline="", encoding="utf-8") as stream:
        row = next(csv.DictReader(stream))
    return out / row["file"]


def test_geometric_order_masks_and_provenance(prep, case):
    uid, comp, cfg = case
    _series(comp, uid)
    out = prep.build_cache(cfg)
    with np.load(_archive(out), allow_pickle=False) as archive:
        images, mask = archive["images"], archive["mask"]
        assert images.shape == (3, 5, 8, 8) and images.dtype == np.uint8
        assert mask.dtype == bool and mask.tolist() == [[False]*5, [False]*5, [True]*3 + [False]*2]
        assert not images[~mask].any()
        assert np.all(np.diff(images[2, :3].mean(axis=(1, 2))) > 0)
    audit = _read(out / "cache_audit.json")[0]
    assert audit["selected"]["AX"]["positions"] == [0., 2., 4.]
    meta = _read(out / "cache_meta.json")
    assert meta["schema"] == "rsna-experiments-cache-v1" and meta["complete"]
    assert meta["fold"] == 0 and meta["cohort_id"] == "cohort"
    assert meta["index_sha256"] == prep.sha256(out / "cache_index.csv")
    assert meta["manifest_sha256"] == prep.sha256(out / "cache_manifest.csv")
    assert prep._read_csv(out / "cache_manifest.csv") == prep._read_csv(cfg["manifest_path"])


def test_only_selected_slices_decode_and_explicit_preference(prep, case, monkeypatch):
    uid, comp, cfg = case
    _series(comp, uid, positions=range(20), description="ax non-fs")
    preferred = _series(comp, uid, positions=range(20), description="ax PD FS")
    _series(comp, uid, description="ax localizer PD FS")
    cfg["slices"] = 3
    reads = []
    original = prep.pydicom.dcmread

    def recording(path, *args, **kwargs):
        if not kwargs.get("stop_before_pixels"):
            reads.append(Path(path))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(prep.pydicom, "dcmread", recording)
    out = prep.build_cache(cfg)
    assert len(reads) == 3 and all(path.parent == preferred for path in reads)
    assert _read(out / "cache_audit.json")[0]["selected"]["AX"]["series"] == preferred.name


@pytest.mark.parametrize("malformation,error", [("duplicate", "duplicate_slice_positions"), ("orientation", "inconsistent_orientation"), ("position", "invalid_ImagePositionPatient"), ("gap", "nonuniform_slice_spacing"), ("origin", "inconsistent_inplane_origins")])
def test_malformed_geometry_retains_failed_study(prep, case, malformation, error):
    uid, comp, cfg = case
    folder = _series(comp, uid, positions=(0, 2, 2) if malformation == "duplicate" else (0, 2, 8) if malformation == "gap" else (0, 2, 4))
    path = folder / "1.dcm"
    ds = pydicom.dcmread(path)
    if malformation == "orientation":
        ds.ImageOrientationPatient = [-1, 0, 0, 0, -1, 0]
    elif malformation == "position":
        del ds.ImagePositionPatient
    elif malformation == "origin":
        ds.ImagePositionPatient = [1, 0, 2]
    ds.save_as(path, enforce_file_format=True)
    out = prep.build_cache(cfg)
    assert not _read(out / "cache_meta.json")["complete"]
    assert error in json.dumps(_read(out / "cache_audit.json"))
    with np.load(_archive(out)) as archive:
        assert not archive["mask"].any() and not archive["images"].any()
    assert prep._read_csv(out / "cache_manifest.csv") == prep._read_csv(cfg["manifest_path"])


@pytest.mark.parametrize("spacing,shape,error", [(None, (160, 160), "invalid_PixelSpacing"), ((-1, 1), (160, 160), "invalid_PixelSpacing"), ((1, 1), (16, 16), "physical_crop_exceeds_fov")])
def test_physical_crop_failures(prep, case, spacing, shape, error):
    uid, comp, cfg = case
    _series(comp, uid, spacing=spacing, shape=shape)
    out = prep.build_cache({**cfg, "recipe": "p1"})
    assert not _read(out / "cache_meta.json")["complete"]
    assert error in json.dumps(_read(out / "cache_audit.json"))


def test_recipes_change_one_operation_and_have_separate_directories(prep, case):
    uid, comp, cfg = case
    _series(comp, uid, positions=range(20), shape=(160, 180))
    outputs = []
    for recipe, operations in prep.RECIPES.items():
        assert sum(operations[key] != prep.RECIPES["p0"][key] for key in operations) == int(recipe != "p0")
        out = prep.build_cache({**cfg, "recipe": recipe})
        assert out.name == recipe and _read(out / "cache_meta.json")["complete"]
        outputs.append(out)
    assert len(set(outputs)) == 6
    assert _read(outputs[0] / "cache_audit.json")[0]["selected"]["AX"]["positions"] != _read(outputs[5] / "cache_audit.json")[0]["selected"]["AX"]["positions"]


def test_resume_checks_corruption_source_changes_and_label_independence(prep, case):
    uid, comp, cfg = case
    folder = _series(comp, uid)
    out = prep.build_cache(cfg)
    archive = _archive(out)
    original_hash, original_time = prep.sha256(archive), archive.stat().st_mtime_ns
    old_id = _read(out / "cache_meta.json")["cache_id"]
    assert prep.build_cache(cfg) == out
    assert archive.stat().st_mtime_ns == original_time
    assert _read(out / "cache_meta.json")["counts"]["resumed"] == 1
    # Supervision can change without invalidating label-free imaging work.
    manifest = Path(cfg["manifest_path"])
    manifest.write_text(manifest.read_text().replace("0.3", "0.8"), encoding="utf-8")
    meta_path = manifest.with_name("preparation_meta.json")
    meta_path.write_text(json.dumps(dict(cohort_id="cohort", supervision_id="supervision-2", fold=0)), encoding="utf-8")
    prep.build_cache(cfg)
    assert archive.stat().st_mtime_ns == original_time
    assert _read(out / "cache_meta.json")["supervision_id"] == "supervision-2"
    assert _read(out / "cache_meta.json")["cache_id"] != old_id
    archive.write_bytes(b"corrupt NPZ")
    prep.build_cache(cfg)
    assert prep.sha256(archive) == original_hash
    assert _read(out / "cache_meta.json")["counts"]["resumed"] == 0
    # Source contents change even with the same path; no stale NPZ reuse.
    path = folder / "0.dcm"
    ds = pydicom.dcmread(path)
    ds.PixelData = np.zeros((16, 16), np.uint16).tobytes()
    ds.save_as(path, enforce_file_format=True)
    prep.build_cache(cfg)
    assert prep.sha256(archive) != original_hash
    with pytest.raises(ValueError, match="mismatch"):
        prep.build_cache({**cfg, "size": 9})
    assert not (out / ".build.lock").exists()


def test_missing_and_bounded_studies_preserve_exact_rows(prep, case):
    uid, comp, cfg = case
    _series(comp, uid)
    other = generate_uid()
    manifest = Path(cfg["manifest_path"])
    rows, _ = prep._read_csv(manifest)
    rows.append({**rows[0], prep.UID: other})
    _csv(manifest, rows)
    out = prep.build_cache({**cfg, "max_studies": 1})
    meta = _read(out / "cache_meta.json")
    assert not meta["complete"] and meta["counts"]["unprocessed"] == 1
    assert prep._read_csv(out / "cache_manifest.csv")[0] == rows
    assert [r[prep.UID] for r in prep._read_csv(out / "cache_index.csv")[0]] == [uid, other]
    out = prep.build_cache({**cfg, "out_dir": str(out.parent / "full")})
    assert not _read(out / "cache_meta.json")["complete"]
    assert _read(out / "cache_meta.json")["counts"]["failed"] == 1
    assert len(list((out / "studies").glob("*.npz"))) == 2


def test_disk_budget_and_unowned_output_are_refused(prep, case):
    _, _, cfg = case
    with pytest.raises(OSError, match="max_cache_gb"):
        prep.build_cache({**cfg, "max_cache_gb": 1e-10})
    out = Path(cfg["out_dir"]) / "p0"
    (out / "unowned.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(ValueError, match="Nonempty"):
        prep.build_cache(cfg)
    assert (out / "unowned.txt").read_text() == "keep"


def test_metadata_plane_and_no_t1_inference(prep, case):
    uid, comp, cfg = case
    folder = _series(comp, uid, plane="SAG")
    _csv(comp / "train_series.csv", [{prep.UID: uid, "SeriesInstanceUID": folder.name, "Anatomical_Plane": "Sagittal", "SeriesDescription": "PD fat sat"}])
    out = prep.build_cache(cfg)
    assert "SAG" in _read(out / "cache_audit.json")[0]["selected"]
    header = pydicom.dcmread(folder / "0.dcm", stop_before_pixels=True)
    assert prep._semantics({"SeriesDescription": "sag non-FS"}, header)[1] == (0, 0, 0)


def test_zscore_is_per_selected_series_not_per_slice(prep, case):
    uid, comp, cfg = case
    _series(comp, uid)
    out = prep.build_cache({**cfg, "recipe": "p3"})
    with np.load(_archive(out)) as archive:
        means = archive["images"][2, :3].mean(axis=(1, 2))
        assert np.all(np.diff(means) > 20)


def test_physical_crop_uses_row_column_spacing(prep, case):
    from PIL import Image

    uid, comp, cfg = case
    folder = _series(comp, uid, positions=(0,), spacing=(1, 2), shape=(160, 180))
    path = folder / "0.dcm"
    ds = pydicom.dcmread(path)
    raw = (np.indices((160, 180))[0] * 10 + np.indices((160, 180))[1]).astype(np.uint16)
    ds.PixelData = raw.tobytes()
    ds.save_as(path, enforce_file_format=True)
    out = prep.build_cache({**cfg, "recipe": "p1"})
    lower, upper = np.percentile(raw.astype(np.float32), [1, 99])
    normalized = np.rint(np.clip((raw.astype(np.float32)-lower)/(upper-lower), 0, 1) * 255).astype(np.uint8)
    # 130 mm / [1,2] mm per pixel = [130,65] pixels, centered independently.
    expected = np.asarray(Image.fromarray(normalized[15:145, 57:122]).resize((8, 8), Image.Resampling.BOX))
    with np.load(_archive(out)) as archive:
        assert np.array_equal(archive["images"][2, 0], expected)
        assert archive["mask"].sum() == 1


def test_selected_decode_failure_is_explicit_and_masks_entire_series(prep, case):
    uid, comp, cfg = case
    folder = _series(comp, uid)
    path = folder / "0.dcm"
    ds = pydicom.dcmread(path)
    ds.PixelData = b"\0\0"
    ds.save_as(path, enforce_file_format=True)
    out = prep.build_cache(cfg)
    assert not _read(out / "cache_meta.json")["complete"]
    assert any(error["stage"] == "decode" for error in _read(out / "cache_audit.json")[0]["errors"])
    with np.load(_archive(out)) as archive:
        assert not archive["mask"].any()


def test_resume_rejects_wrong_array_contract_even_with_matching_checksum(prep, case):
    uid, comp, cfg = case
    _series(comp, uid)
    out = prep.build_cache(cfg)
    path = _archive(out)
    sidecar = path.with_name(path.name + ".json")
    np.savez_compressed(path, images=np.zeros((3, 5, 8, 8), np.float32), mask=np.ones((3, 5), bool))
    info = _read(sidecar)
    info["sha256"] = prep.sha256(path)
    prep.write_json(sidecar, info)
    prep.build_cache(cfg)
    assert _read(out / "cache_meta.json")["counts"]["resumed"] == 0
    with np.load(path) as archive:
        assert archive["images"].dtype == np.uint8
        assert archive["mask"].sum() == 3


def test_memory_bound_rejects_before_decoding(prep, monkeypatch):
    def must_not_decode(*args, **kwargs):
        raise AssertionError("pixel decoding must not start")

    monkeypatch.setattr(prep.pydicom, "dcmread", must_not_decode)
    with pytest.raises(ValueError, match="memory_bound"):
        prep._series_stack([{"shape": (4096, 4096)}] * 4, dict(recipe="p0", slices=4, size=8))


def test_single_value_localizer_image_type_is_excluded(prep, case):
    uid, comp, cfg = case
    folder = _series(comp, uid)
    for path in folder.glob("*.dcm"):
        ds = pydicom.dcmread(path)
        ds.ImageType = "LOCALIZER"
        ds.save_as(path, enforce_file_format=True)
    out = prep.build_cache(cfg)
    assert not _read(out / "cache_meta.json")["complete"]
    assert "localizer_excluded" in json.dumps(_read(out / "cache_audit.json"))
