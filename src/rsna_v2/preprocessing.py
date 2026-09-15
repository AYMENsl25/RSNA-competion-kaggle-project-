"""Deterministic, label-free MRI cache construction (not a B2B1 reproduction).

Only selected slices are retained in memory. Every source instance is decoded to
audit failures, including instances outside the sampling band. Pixel data and
free-text DICOM fields are never logged. Synthetic tests do not qualify real data.
"""
from pathlib import Path
import csv
import json
import logging
import math
import re
import shutil

import numpy as np
import pydicom
from PIL import Image

from .common import SLOTS, fingerprint, hash_file, resolve_comp_dir, resolve_unique

LOG = logging.getLogger(__name__)
VERSION = "rsna-v2-preprocess-1"
DEFAULTS = dict(n_slices=24, size=224, crop_mm=130.0,
                normalization="percentile", slice_band=[.05, .95],
                laterality="off", force_dicom=False, reject_multiframe=True,
                orientation_tolerance_deg=2.0, slot_strategy="legacy", max_decode_failure_fraction=0.1)
INDEX_FIELDS = ["UID", "SeriesUID", "slot", "path", "cv_group_id",
                "training_role", "n_valid", "preprocessing_id"]


def _config(config):
    config = dict(config)
    for alias, key in (("image_size", "size"), ("norm", "normalization")):
        if alias in config:
            if key in config and config[key] != config[alias]:
                raise ValueError(f"Conflicting {alias}/{key}")
            config[key] = config[alias]
    cfg = {**DEFAULTS, **{k: v for k, v in config.items() if k in DEFAULTS},
           **config.get("cache", {})}
    for key in ("n_slices", "size"):
        if isinstance(cfg[key], bool) or int(cfg[key]) != cfg[key] or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
        cfg[key] = int(cfg[key])
    if cfg["normalization"] not in {"percentile", "clahe", "zscore"}:
        raise ValueError("Unknown normalization")
    if cfg["laterality"] not in {"off", "geometry-canonical"}:
        raise ValueError("Unknown laterality policy")
    if cfg["slot_strategy"] not in {"legacy", "semantic"}:
        raise ValueError("Unknown slot strategy")
    lo, hi = cfg["slice_band"]
    if not 0 <= lo < hi <= 1:
        raise ValueError("slice_band must satisfy 0 <= low < high <= 1")
    if cfg["crop_mm"] is not None and not (math.isfinite(cfg["crop_mm"]) and cfg["crop_mm"] > 0):
        raise ValueError("crop_mm must be positive or None")
    if not 0 < cfg["orientation_tolerance_deg"] <= 10:
        raise ValueError("orientation_tolerance_deg must be in (0, 10]")
    if not 0 <= cfg['max_decode_failure_fraction'] <= 1:
        raise ValueError('max_decode_failure_fraction must be in [0,1]')
    for key in ("force_dicom", "reject_multiframe"):
        if not isinstance(cfg[key], bool):
            raise ValueError(f"{key} must be boolean")
    if not cfg["reject_multiframe"]:
        raise ValueError("Multiframe processing is not implemented; explicit rejection is required")
    cfg["image_size"] = cfg["size"]
    cfg["norm"] = cfg["normalization"]
    return cfg


def _vector(ds, key, length):
    try:
        arr = np.asarray(getattr(ds, key), dtype=np.float64)
        return arr if arr.shape == (length,) and np.isfinite(arr).all() else None
    except (AttributeError, TypeError, ValueError):
        return None


def _axes(ds):
    v = _vector(ds, "ImageOrientationPatient", 6)
    if v is None:
        return None
    x, y = v[:3], v[3:]
    if abs(np.linalg.norm(x) - 1) > .01 or abs(np.linalg.norm(y) - 1) > .01 or abs(x @ y) > .01:
        return None
    x, y = x / np.linalg.norm(x), y / np.linalg.norm(y)
    z = np.cross(x, y)
    return np.stack([x, y, z / np.linalg.norm(z)])


def _atomic_json(path, obj):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _csv(path, rows, fields):
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _normalize(arr, ds, cfg):
    x = np.asarray(arr, dtype=np.float32)
    slope, intercept = float(getattr(ds, "RescaleSlope", 1)), float(getattr(ds, "RescaleIntercept", 0))
    if not math.isfinite(slope) or not math.isfinite(intercept) or slope == 0:
        raise ValueError("invalid_rescale")
    x = x * slope + intercept
    if not np.isfinite(x).all():
        raise ValueError("nonfinite_pixels")
    photo = str(getattr(ds, "PhotometricInterpretation", ""))
    if photo not in {"MONOCHROME1", "MONOCHROME2"}:
        raise ValueError("unsupported_photometric")
    if photo == "MONOCHROME1":
        x = x.min() + x.max() - x
    if cfg["normalization"] == "zscore":
        std = float(x.std())
        x = np.zeros_like(x) if std < 1e-6 else np.clip(((x - x.mean()) / std + 3) / 6, 0, 1)
    else:
        lo, hi = np.percentile(x, [1, 99])
        if hi - lo < 1e-6:
            lo, hi = float(x.min()), float(x.max())
        x = np.zeros_like(x) if hi - lo < 1e-6 else np.clip((x - lo) / (hi - lo), 0, 1)
    x = np.rint(x * 255).astype(np.uint8)
    if cfg["normalization"] == "clahe":
        import cv2
        x = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(x)
    return x


def _crop_resize(x, ds, cfg):
    audit = "off"
    if cfg["crop_mm"] is not None:
        spacing = _vector(ds, "PixelSpacing", 2)
        if spacing is None or (spacing <= 0).any():
            raise ValueError("invalid_pixel_spacing_for_crop")
        wanted = np.rint(cfg["crop_mm"] / spacing)
        h, w = [max(1, int(min(want, actual))) for want, actual in zip(wanted, x.shape)]
        audit = "clipped_to_image" if any(wanted > np.asarray(x.shape)) else "physical_center"
        top, left = (x.shape[0] - h) // 2, (x.shape[1] - w) // 2
        x = x[top:top+h, left:left+w]
    return np.asarray(Image.fromarray(x).resize((cfg["size"], cfg["size"]), Image.Resampling.BOX)), audit


def preprocess_one_series(series_path, config):
    """Return uint8 [N,H,W] and metadata containing positions/valid lists.

    Invalid or padded positions are zero with valid=False. Physical positions
    are millimetres only when position_kind='mm'; otherwise they are ordinal.
    Geometry-canonical makes cardinal in-plane axes positive in DICOM LPS;
    this is orientation standardization, not an inferred left/right knee label.
    """
    cfg = _config(config)
    folder = Path(series_path)
    files = sorted(p for p in folder.iterdir() if p.is_file()) if folder.is_dir() else []
    records, failures = [], []
    for index, path in enumerate(files):
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True, force=cfg["force_dicom"])
            if int(getattr(ds, "NumberOfFrames", 1)) != 1:
                raise ValueError("multiframe_rejected")
            records.append(dict(path=path, ds=ds, axes=_axes(ds), ipp=_vector(ds, "ImagePositionPatient", 3), index=index))
        except Exception as exc:
            failures.append(dict(file_index=index, stage="header", reason=type(exc).__name__, code=str(exc) if str(exc) == "multiframe_rejected" else "header_read_failed"))
    flags = []
    axes = next((r["axes"] for r in records if r["axes"] is not None), None)
    consistent = axes is not None and all(r["axes"] is not None and np.all(np.sum(r["axes"] * axes, axis=1) >= math.cos(math.radians(cfg["orientation_tolerance_deg"]))) for r in records)
    spatial = bool(records) and consistent and all(r["ipp"] is not None for r in records)
    if spatial:
        normal = axes[2].copy()
        if cfg["laterality"] == "geometry-canonical" and normal[np.argmax(abs(normal))] < 0:
            normal *= -1
        for r in records:
            r["position"] = float(r["ipp"] @ normal)
        order_method, position_kind = "reference_iop_projection", "mm"
        records.sort(key=lambda r: (r["position"], r["path"].name))
    else:
        flags.append("missing_or_inconsistent_geometry")
        try:
            values = [float(r["ds"].InstanceNumber) for r in records]
            if not np.isfinite(values).all() or len(set(values)) != len(values):
                raise ValueError("ambiguous_instance_numbers")
            for r, value in zip(records, values):
                r["position"] = value
            records.sort(key=lambda r: (r["position"], r["path"].name))
            order_method = "instance_number_fallback"
        except (AttributeError, ValueError, TypeError):
            order_method = "filename_fallback"
        position_kind = "ordinal"
        for index, r in enumerate(records):
            r["position"] = float(index)
        flags.append(order_method)
    geometry_rejected = False
    if not consistent and any(r["axes"] is not None for r in records) and all(r["axes"] is not None for r in records):
        flags.append("inconsistent_orientation_rejected")
        geometry_rejected = True
    flips = (False, False)
    if cfg["laterality"] == "geometry-canonical":
        cardinal = consistent and len(set(np.argmax(abs(axes), axis=1))) == 3 and np.all(np.max(abs(axes), axis=1) >= math.cos(math.radians(cfg["orientation_tolerance_deg"])))
        if not spatial or not cardinal:
            flags.append("unsafe_canonical_geometry_rejected")
            geometry_rejected = True
        else:
            flips = (bool(axes[1, np.argmax(abs(axes[1]))] < 0), bool(axes[0, np.argmax(abs(axes[0]))] < 0))
    if spatial and len(records) > 1:
        gaps = np.diff([r["position"] for r in records])
        if (gaps <= 1e-4).any():
            flags.append("duplicate_positions_rejected")
            geometry_rejected = True
        if (gaps > 0).any() and gaps.max() > 2 * np.median(gaps[gaps > 0]):
            flags.append("large_spacing_gap")
    for key in ("StudyInstanceUID", "SeriesInstanceUID", "FrameOfReferenceUID"):
        values = {str(getattr(r["ds"], key, "")) for r in records}
        if len(values) > 1:
            flags.append(f"mixed_{key}_rejected")
            geometry_rejected = True
    sop = [str(getattr(r["ds"], "SOPInstanceUID", "")) for r in records]
    if len([s for s in sop if s]) != len(set(s for s in sop if s)):
        flags.append("duplicate_sop_rejected")
        geometry_rejected = True
    n, size = cfg["n_slices"], cfg["size"]
    images = np.zeros((n, size, size), np.uint8)
    valid = np.zeros(n, bool)
    positions = np.zeros(n, np.float32)
    if records:
        lo = int(np.floor(cfg["slice_band"][0] * (len(records)-1)))
        hi = int(np.ceil(cfg["slice_band"][1] * (len(records)-1)))
        selected = np.rint(np.linspace(lo, hi, min(n, hi-lo+1))).astype(int)
    else:
        selected = []
    selected_map = {int(r): i for i, r in enumerate(selected)}
    crop_modes, decoded = set(), 0
    for index, r in enumerate(records):
        try:
            ds = pydicom.dcmread(r["path"], force=cfg["force_dicom"])
            raw = ds.pixel_array
            if raw.ndim != 2 or int(getattr(ds, "SamplesPerPixel", 1)) != 1:
                raise ValueError("not_single_grayscale_frame")
            x = _normalize(raw, ds, cfg)
            x, crop_mode = _crop_resize(x, ds, cfg)
            crop_modes.add(crop_mode)
            decoded += 1
            if index in selected_map and not geometry_rejected:
                out_index = selected_map[index]
                if flips[0]:
                    x = x[::-1, :]
                if flips[1]:
                    x = x[:, ::-1]
                images[out_index] = x
                positions[out_index] = r["position"]
                valid[out_index] = True
        except Exception as exc:
            safe_codes = {"invalid_rescale", "nonfinite_pixels", "unsupported_photometric", "invalid_pixel_spacing_for_crop", "not_single_grayscale_frame"}
            failures.append(dict(file_index=r["index"], stage="decode", reason=type(exc).__name__, code=str(exc) if str(exc) in safe_codes else "pixel_decode_or_transform_failed"))
    metadata = dict(n_files=len(files), n_decoded=decoded, n_valid=int(valid.sum()),
                    n_selected=len(selected), positions=positions.tolist(), valid=valid.tolist(),
                    position_kind=position_kind, order_method=order_method,
                    geometry_flags=flags, geometry_rejected=geometry_rejected,
                    crop_modes=sorted(crop_modes), canonical_flips=list(flips),
                    decode_failures=failures, complete=bool(records) and len(failures)/max(len(files),1) <= cfg['max_decode_failure_fraction'] and not geometry_rejected and bool(valid.any()))
    return images, metadata


def _binary(value):
    value = str(value).strip().lower()
    if value in {"1", "1.0", "true", "yes", "fs", "fatsat"}:
        return 1
    if value in {"0", "0.0", "false", "no", "nonfs", "non-fs"}:
        return 0
    return None


def _semantic_slot(row):
    """Use explicit plane/FS metadata or unambiguous description tokens only."""
    plane_map = {"sagittal": "SAG", "sag": "SAG", "coronal": "COR", "cor": "COR", "axial": "AX", "ax": "AX"}
    desc = str(row.get("first_SeriesDescription", row.get("SeriesDescription", ""))).lower()
    tokens = set(re.findall(r"[a-z0-9]+", desc))
    planes = {plane_map[t] for t in tokens if t in plane_map}
    explicit = plane_map.get(str(row.get("Anatomical_Plane", "")).strip().lower())
    if explicit:
        planes.add(explicit)
    fs = _binary(row.get("Fat_Suppression", row.get("fat_bin", "")))
    has_nonfs = bool(re.search(r"non[ _-]?(?:fs|fat[ _-]?sat)", desc))
    has_fs = bool(tokens & {"fs", "fatsat", "stir", "spair"}) and not has_nonfs
    inferred = 0 if has_nonfs else (1 if has_fs else None)
    if len(planes) != 1 or (fs is not None and inferred is not None and fs != inferred):
        return "UNKNOWN", "ambiguous_semantics"
    fs = fs if fs is not None else inferred
    if fs is None:
        return "UNKNOWN", "unknown_fat_suppression"
    return f"{next(iter(planes))}_{'FS' if fs else 'NONFS'}", "metadata_or_description"


def _component(value):
    if not value or not re.fullmatch(r"[A-Za-z0-9_.-]+", value) or value in {".", ".."}:
        raise ValueError("Unsafe or empty source identifier")
    return value


def _series_path(comp, row):
    uid, series = _component(row["StudyInstanceUID"]), _component(row["SeriesInstanceUID"])
    split = _component(row.get("split") or "train")
    candidates = [comp / f"{split}_series" / uid / series, comp / f"{split}_images" / uid / series]
    supplied = row.get("series_path", "")
    if supplied:
        p = Path(supplied)
        p = p if p.is_absolute() else comp / p
        if p.resolve().is_relative_to(comp.resolve()):
            candidates.append(p)
    hits = sorted({p.resolve() for p in candidates if p.is_dir()})
    if len(hits) > 1:
        raise ValueError("Ambiguous source series directories")
    return hits[0] if hits else candidates[0]


def _source_signature(path):
    return fingerprint([(p.name, p.stat().st_size, p.stat().st_mtime_ns)
                        for p in sorted(path.iterdir()) if p.is_file()]) if path.is_dir() else "missing"


def _valid_cache(path, sidecar, preprocessing_id, source_signature, cfg):
    try:
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
        if meta["preprocessing_id"] != preprocessing_id or meta["source_signature"] != source_signature or meta["sha256"] != hash_file(path):
            return None
        with np.load(path, allow_pickle=False) as z:
            images, valid, positions = z["images"], z["valid"], z["positions"]
            if images.shape != (cfg["n_slices"], cfg["size"], cfg["size"]) or images.dtype != np.uint8 or valid.shape != (cfg["n_slices"],) or valid.dtype != bool or positions.shape != valid.shape or positions.dtype != np.float32 or not np.isfinite(positions).all():
                return None
            if int(valid.sum()) != meta["series_meta"]["n_valid"]:
                return None
        return meta["series_meta"]
    except (OSError, ValueError, KeyError, EOFError):
        return None


def run_preprocessing(config):
    """Build or resume a cache. Production raises after writing incomplete audits.

    Completeness means all planned series succeeded; absent legacy slots are
    reported, not invented. smoke=True always implies complete=False globally.
    Existing NPZs require both matching provenance and SHA256 before reuse.
    """
    cfg = _config(config)
    comp = Path(resolve_comp_dir(config.get("comp_dir")))
    roots = config.get("input_roots", ["/kaggle/input"])
    plan = Path(resolve_unique("rsna_phase4b2a_six_slot_plan_v1.csv", roots, override=config.get("slot_plan_path")))
    with plan.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        required = {"StudyInstanceUID", "SeriesInstanceUID", "slot", "cv_group_id", "training_role"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Slot plan missing columns: {sorted(required - set(reader.fieldnames or []))}")
        rows = list(reader)
    if not rows:
        raise ValueError("Empty slot plan")
    all_studies = sorted({r["StudyInstanceUID"] for r in rows})
    smoke_count = config.get("smoke_studies", 0)
    if isinstance(smoke_count, bool) or int(smoke_count) != smoke_count or smoke_count < 0:
        raise ValueError("smoke_studies must be a nonnegative integer")
    smoke = smoke_count > 0
    included = set(all_studies[:int(smoke_count)] if smoke else all_studies)
    rows = sorted([r for r in rows if r["StudyInstanceUID"] in included], key=lambda r: (r["StudyInstanceUID"], r["slot"], r["SeriesInstanceUID"]))
    seen, study_roles = set(), {}
    for row in rows:
        uid = row["StudyInstanceUID"]
        key = (uid, row["slot"])
        if key in seen:
            raise ValueError("Duplicate study-slot rows require an explicit upstream selection")
        seen.add(key)
        role = (row["cv_group_id"], row["training_role"])
        if not all(role) or (uid in study_roles and study_roles[uid] != role):
            raise ValueError("Missing or inconsistent study grouping/role")
        study_roles[uid] = role
    identity = dict(version=VERSION, config=cfg, slot_plan_sha256=hash_file(plan),
                    smoke_studies=int(smoke_count), source_root=str(comp.resolve()))
    preprocessing_id = fingerprint(identity)
    out = Path(config.get("out_dir", "/kaggle/working/rsna_v2_cache"))
    out.mkdir(parents=True, exist_ok=True)
    config_path = out / "preprocess_config.json"
    if config_path.exists():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if previous.get("preprocessing_id") != preprocessing_id:
            raise ValueError("Existing cache configuration/provenance differs; choose a new out_dir")
    elif any(out.iterdir()):
        raise ValueError("Nonempty output has no preprocessing identity; choose a new out_dir")
    per_series = cfg["n_slices"] * cfg["size"] ** 2 + cfg["n_slices"] * 5 + 16384
    estimate = int(len(rows) * per_series * 1.05)
    if estimate > float(config.get("max_cache_gb", 30)) * 1e9:
        raise OSError("Estimated cache exceeds max_cache_gb; reduce size/slices or increase explicit budget")
    existing_bytes = sum(p.stat().st_size for p in out.rglob("*.npz"))
    required_free = max(per_series * 2, estimate - existing_bytes) + int(config.get("disk_reserve_bytes", 256 * 1024**2))
    if shutil.disk_usage(out).free < required_free:
        raise OSError("Insufficient free disk space for estimated cache and reserve")
    _atomic_json(config_path, {**identity, "preprocessing_id": preprocessing_id})
    meta = dict(preprocessing_id=preprocessing_id, config=cfg, complete=False,
                smoke=smoke, smoke_studies=int(smoke_count), planned_series=len(rows),
                planned_studies=len(included), total_plan_studies=len(all_studies), estimated_bytes=estimate)
    _atomic_json(out / "cache_meta.json", meta)
    index, geometry, decode, slots = [], [], [], []
    failed = 0
    cache = out / "series"
    cache.mkdir(exist_ok=True)
    for number, row in enumerate(rows):
        uid, series = row["StudyInstanceUID"], row["SeriesInstanceUID"]
        path = _series_path(comp, row)
        slot, slot_reason = row["slot"], "legacy_plan"
        if cfg["slot_strategy"] == "semantic":
            semantic_row = dict(row)
            if not semantic_row.get("first_SeriesDescription") and path.is_dir():
                for source_file in sorted(path.iterdir()):
                    if not source_file.is_file():
                        continue
                    try:
                        header = pydicom.dcmread(source_file, stop_before_pixels=True, force=cfg["force_dicom"], specific_tags=["SeriesDescription"])
                        semantic_row["first_SeriesDescription"] = str(getattr(header, "SeriesDescription", ""))
                        break
                    except Exception:
                        continue
            slot, slot_reason = _semantic_slot(semantic_row)
        slot_ok = slot in SLOTS and not any(r["UID"] == uid and r["slot"] == slot for r in index)
        slots.append(dict(UID=uid, SeriesUID=series, legacy_slot=row["slot"], slot=slot,
                          status=slot_reason if slot_ok else "unknown_or_ambiguous_slot"))
        if not slot_ok:
            failed += 1
            continue
        name = fingerprint([uid, series, slot]) + ".npz"
        dest, sidecar = cache / name, cache / (name + ".json")
        signature = _source_signature(path)
        series_meta = _valid_cache(dest, sidecar, preprocessing_id, signature, cfg) if dest.exists() and sidecar.exists() else None
        if series_meta is None:
            images, series_meta = preprocess_one_series(path, cfg)
            temporary = dest.with_suffix(".npz.tmp")
            with temporary.open("wb") as f:
                np.savez_compressed(f, images=images, positions=np.asarray(series_meta["positions"], np.float32), valid=np.asarray(series_meta["valid"], bool))
            temporary.replace(dest)
            _atomic_json(sidecar, dict(preprocessing_id=preprocessing_id, source_signature=signature,
                                      sha256=hash_file(dest), series_meta=series_meta))
        failed += int(not series_meta["complete"])
        index.append(dict(UID=uid, SeriesUID=series, slot=slot, path=dest.relative_to(out).as_posix(),
                          cv_group_id=row["cv_group_id"], training_role=row["training_role"],
                          n_valid=series_meta["n_valid"], preprocessing_id=preprocessing_id))
        geometry.append(dict(UID=uid, SeriesUID=series, order_method=series_meta["order_method"],
                             position_kind=series_meta["position_kind"], n_files=series_meta["n_files"],
                             n_decoded=series_meta["n_decoded"], n_valid=series_meta["n_valid"],
                             flags="|".join(series_meta["geometry_flags"]), crop_modes="|".join(series_meta["crop_modes"]),
                             complete=series_meta["complete"]))
        decode.extend(dict(UID=uid, SeriesUID=series, **failure) for failure in series_meta["decode_failures"])
        if number % 100 == 0:
            LOG.info("Preprocessing %d/%d series; incomplete=%d", number+1, len(rows), failed)
    coverage = [dict(slot=slot, planned_studies=len(included),
                     n_studies_present=len({r["UID"] for r in index if r["slot"] == slot and r["n_valid"] > 0})) for slot in SLOTS]
    public_index = [{('StudyInstanceUID' if k == 'UID' else 'SeriesInstanceUID' if k == 'SeriesUID' else k): v for k,v in row.items()} for row in index]
    public_fields = ['StudyInstanceUID' if k == 'UID' else 'SeriesInstanceUID' if k == 'SeriesUID' else k for k in INDEX_FIELDS]
    _csv(out / "cache_index.csv", public_index, public_fields)
    _csv(out / "geometry_audit.csv", geometry, ["UID", "SeriesUID", "order_method", "position_kind", "n_files", "n_decoded", "n_valid", "flags", "crop_modes", "complete"])
    _csv(out / "decode_audit.csv", decode, ["UID", "SeriesUID", "file_index", "stage", "reason", "code"])
    _csv(out / "slot_audit.csv", slots, ["UID", "SeriesUID", "legacy_slot", "slot", "status"])
    _csv(out / "slot_coverage.csv", coverage, ["slot", "planned_studies", "n_studies_present"])
    meta.update(complete=not smoke and failed == 0 and len(index) == len(rows),
                subset_complete=failed == 0 and len(index) == len(rows), failed_series=failed,
                cached_series=len(index), cache_index_sha256=hash_file(out / "cache_index.csv"))
    meta["index_sha256"] = meta["cache_index_sha256"]
    _atomic_json(out / "cache_meta.json", meta)
    if not smoke and not meta["complete"]:
        raise RuntimeError(f"Production cache incomplete: {failed} failed series; inspect audit CSVs")
    return out
