"""Bounded, label-independent MRI slice stacks; no canonical/isotropic claim.

Plane order is SAG, COR, AX. Depth sampling is ordinal after geometric ordering;
padding is zero and invalid. Only selected instances are pixel-decoded. Source
SHA256 auditing reads compressed bytes, including unselected instances. Recipes
change exactly one baseline operation. Percentiles/z-score use all pixels of the
selected stack, not the entire acquisition (which would require extra decoding).
"""
from pathlib import Path
import csv
import json
import logging
import math
import platform
import re
import shutil
import time
import zipfile

import numpy as np
import pydicom
from PIL import Image

from .common import UID, fingerprint, locate, sha256, write_json

LOG = logging.getLogger(__name__)
SCHEMA = "rsna-experiments-cache-v1"
VERSION = "bounded-mri-2"
PLANES = ("SAG", "COR", "AX")
RECIPES = {
    "p0": dict(band=(.05, .95), crop_mm=None, normalization="percentile", laterality="none"),
    "p1": dict(band=(.05, .95), crop_mm=130., normalization="percentile", laterality="none"),
    "p2": dict(band=(.05, .95), crop_mm=150., normalization="percentile", laterality="none"),
    "p3": dict(band=(.05, .95), crop_mm=None, normalization="zscore", laterality="none"),
    "p4": dict(band=(.05, .95), crop_mm=None, normalization="clahe", laterality="none"),
    "p5": dict(band=(.12, .88), crop_mm=None, normalization="percentile", laterality="none"),
    # Canonicalize right knees to the left-knee orientation used by the public
    # reference pipeline. This is deliberately its own one-factor ablation.
    "p6": dict(band=(.05, .95), crop_mm=None, normalization="percentile", laterality="canonical"),
}


def _read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        return list(reader), reader.fieldnames or []


def _csv(path, rows, fields):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _component(value):
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", value or "") or value in {".", ".."}:
        raise ValueError("Unsafe or empty study/series identifier")
    return value


def _vector(ds, key, length):
    try:
        value = np.asarray(getattr(ds, key), dtype=np.float64)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("invalid_" + key) from exc
    if value.shape != (length,) or not np.isfinite(value).all():
        raise ValueError("invalid_" + key)
    return value


def _axes(ds):
    values = _vector(ds, "ImageOrientationPatient", 6).reshape(2, 3)
    if not np.allclose(np.linalg.norm(values, axis=1), 1, atol=.001) or abs(values[0] @ values[1]) > .001:
        raise ValueError("invalid_orientation_axes")
    values = values / np.linalg.norm(values, axis=1)[:, None]
    return values, np.cross(values[0], values[1])


def _semantics(row, ds):
    """Explicit metadata only: absence of FS never implies T1."""
    description = " ".join([str(row.get("SeriesDescription", "")), str(getattr(ds, "SeriesDescription", ""))]).lower()
    raw_type = getattr(ds, "ImageType", [])
    image_type = raw_type if isinstance(raw_type, str) else " ".join(str(x) for x in raw_type)
    if re.search(r"locali[sz]er|scout|survey|\bloc\b", description + " " + image_type, re.I):
        raise ValueError("localizer_excluded")
    aliases = {"sag": "SAG", "sagittal": "SAG", "cor": "COR", "coronal": "COR", "ax": "AX", "axial": "AX", "transverse": "AX", "tra": "AX"}
    explicit = aliases.get(str(row.get("Anatomical_Plane", "")).strip().lower())
    tokens = set(re.findall(r"[a-z0-9]+", description))
    hints = {aliases[t] for t in tokens if t in aliases}
    if len(hints) > 1 or (explicit and hints and explicit not in hints):
        raise ValueError("conflicting_plane_metadata")
    _, normal = _axes(ds)
    axis = int(np.argmax(abs(normal)))
    if abs(normal[axis]) < math.cos(math.radians(30)):
        raise ValueError("ambiguous_oblique_plane")
    geometric = ("SAG", "COR", "AX")[axis]
    plane = explicit or next(iter(hints), None) or geometric
    if plane != geometric:
        raise ValueError("plane_metadata_geometry_conflict")
    non_fs = bool(re.search(r"non[ _-]?(?:fs|fat[ _-]?sat)|without\s+fat", description))
    fat = str(row.get("Fat_Suppression", "")).strip().lower()
    fs = fat in {"1", "1.0", "true", "yes", "fs"} or (not non_fs and bool(tokens & {"fs", "fatsat", "stir", "spair"} or re.search(r"fat[ _-]?(?:sat|supp)", description)))
    if fat in {"0", "0.0", "false", "no"} or non_fs:
        fs = False
    fluid = bool(tokens & {"t2", "t2w", "pd", "pdw", "stir"})
    return plane, (int(fluid and fs), int(fs), int(fluid))


def _laterality(records):
    """Resolve L/R conservatively from tags, then patient-space volume center."""
    tagged = set()
    for item in records:
        for key in ("ImageLaterality", "Laterality"):
            value = str(getattr(item["ds"], key, "")).strip().upper()
            if value in {"L", "LEFT"}: tagged.add("L")
            elif value in {"R", "RIGHT"}: tagged.add("R")
    if len(tagged) > 1:
        raise ValueError("conflicting_laterality_tags")
    centers = []
    for item in records:
        spacing = item.get("spacing")
        if spacing is None:
            continue
        rows, columns = item["shape"]
        axes = item["axes"]
        center = (item["ipp"] + axes[0] * ((columns - 1) * spacing[1] / 2)
                  + axes[1] * ((rows - 1) * spacing[0] / 2))
        centers.append(float(center[0]))
    geometric = None
    if centers:
        x = float(np.median(centers))
        geometric = "L" if x > 5 else "R" if x < -5 else None
    tag = next(iter(tagged), None)
    if tag and geometric and tag != geometric:
        raise ValueError("laterality_tag_geometry_conflict")
    if tag:
        return tag, "dicom_tag"
    if geometric:
        return geometric, "patient_geometry"
    raise ValueError("laterality_unresolved")


def _headers(folder, row, uid, series, recipe):
    records = []
    files = sorted(p for p in folder.iterdir() if p.is_file())
    if not files:
        raise ValueError("empty_series")
    for path in files:
        ds = pydicom.dcmread(path, stop_before_pixels=True)
        axes, normal = _axes(ds)
        position = _vector(ds, "ImagePositionPatient", 3)
        if int(getattr(ds, "NumberOfFrames", 1)) != 1:
            raise ValueError("multiframe_rejected")
        if str(getattr(ds, "StudyInstanceUID", "")) != uid or str(getattr(ds, "SeriesInstanceUID", "")) != series:
            raise ValueError("source_uid_mismatch")
        if int(getattr(ds, "SamplesPerPixel", 0)) != 1 or str(getattr(ds, "PhotometricInterpretation", "")) not in {"MONOCHROME1", "MONOCHROME2"}:
            raise ValueError("unsupported_photometric")
        shape = (int(ds.Rows), int(ds.Columns))
        if min(shape) < 1 or max(shape) > 4096 or shape[0] * shape[1] > 4096**2:
            raise ValueError("source_dimensions_out_of_bounds")
        spacing = _vector(ds, "PixelSpacing", 2) if hasattr(ds, "PixelSpacing") or recipe["crop_mm"] or recipe["laterality"] == "canonical" else None
        if spacing is not None and (spacing <= 0).any():
            raise ValueError("invalid_PixelSpacing")
        plane, preference = _semantics(row, ds)
        records.append(dict(path=path, ds=ds, axes=axes, normal=normal, ipp=position,
                            shape=shape, spacing=spacing, plane=plane, preference=preference))
    first = records[0]
    sop = set()
    for item in records:
        if not np.allclose(item["axes"], first["axes"], atol=1e-4, rtol=0):
            raise ValueError("inconsistent_orientation")
        if item["shape"] != first["shape"] or item["plane"] != first["plane"]:
            raise ValueError("inconsistent_dimensions_or_plane")
        a, b = item["spacing"], first["spacing"]
        if (a is None) != (b is None) or (a is not None and not np.allclose(a, b, atol=1e-4, rtol=0)):
            raise ValueError("inconsistent_PixelSpacing")
        if str(getattr(item["ds"], "FrameOfReferenceUID", "")) != str(getattr(first["ds"], "FrameOfReferenceUID", "")):
            raise ValueError("mixed_frame_of_reference")
        instance = str(getattr(item["ds"], "SOPInstanceUID", ""))
        if not instance or instance in sop:
            raise ValueError("missing_or_duplicate_sop")
        sop.add(instance)
        offset = item["ipp"] - first["ipp"]
        if np.linalg.norm(offset - (offset @ first["normal"]) * first["normal"]) > .1:
            raise ValueError("inconsistent_inplane_origins")
        item["position"] = float(item["ipp"] @ first["normal"])
    records.sort(key=lambda item: item["position"])
    gaps = np.diff([item["position"] for item in records])
    if len(gaps) and (gaps <= 1e-3).any():
        raise ValueError("duplicate_slice_positions")
    # Missing/nonuniform slice intervals are rejected, not interpolated silently.
    if len(gaps) > 1 and not np.allclose(gaps, np.median(gaps), rtol=.1, atol=.1):
        raise ValueError("nonuniform_slice_spacing")
    if recipe["crop_mm"]:
        wanted = np.rint(recipe["crop_mm"] / first["spacing"]).astype(int)
        if (wanted < 1).any() or (wanted > first["shape"]).any():
            raise ValueError("physical_crop_exceeds_fov")
    if recipe["laterality"] == "canonical":
        laterality, source = _laterality(records)
        for item in records:
            item["laterality"], item["laterality_source"] = laterality, source
    return records


def _series_stack(records, cfg):
    recipe = RECIPES[cfg["recipe"]]
    low, high = recipe["band"]
    # Slice-center quantiles give meaningful bands even for short acquisitions.
    centers = (np.arange(len(records)) + .5) / len(records)
    candidates = np.flatnonzero((centers >= low) & (centers <= high))
    selected = candidates[np.rint(np.linspace(0, len(candidates)-1, min(cfg["slices"], len(candidates)))).astype(int)]
    if len(selected) * math.prod(records[0]["shape"]) > 32 * 1024**2:
        raise ValueError("selected_source_stack_exceeds_memory_bound")
    decoded = []
    for index in selected:
        record = records[int(index)]
        ds = pydicom.dcmread(record["path"])
        pixels = np.asarray(ds.pixel_array, dtype=np.float32)
        if pixels.shape != record["shape"]:
            raise ValueError("decoded_shape_mismatch")
        slope, intercept = float(getattr(ds, "RescaleSlope", 1)), float(getattr(ds, "RescaleIntercept", 0))
        if not math.isfinite(slope) or slope == 0 or not math.isfinite(intercept):
            raise ValueError("invalid_rescale")
        pixels = pixels * slope + intercept
        if not np.isfinite(pixels).all():
            raise ValueError("nonfinite_pixels")
        if ds.PhotometricInterpretation == "MONOCHROME1":
            pixels = pixels.min() + pixels.max() - pixels
        decoded.append(pixels)
    stack = np.stack(decoded)
    # Normalize before crop so physical-crop recipes change only the FOV.
    if recipe["normalization"] == "zscore":
        deviation = float(stack.std(dtype=np.float64))
        stack = np.full_like(stack, .5) if deviation < 1e-6 else np.clip((stack - stack.mean(dtype=np.float64)) / deviation, -3, 3) / 6 + .5
    else:
        lower, upper = np.percentile(stack, [1, 99])
        stack = np.zeros_like(stack) if upper - lower < 1e-6 else np.clip((stack - lower) / (upper - lower), 0, 1)
    stack = np.rint(stack * 255).astype(np.uint8)
    if recipe["crop_mm"]:
        h, w = np.rint(recipe["crop_mm"] / records[0]["spacing"]).astype(int)
        top, left = (stack.shape[1] - h) // 2, (stack.shape[2] - w) // 2
        stack = stack[:, top:top+h, left:left+w]
    if recipe["laterality"] == "canonical" and records[0]["laterality"] == "R":
        if records[0]["plane"] == "SAG":
            stack = stack[::-1]
            selected = selected[::-1]
        else:
            stack = stack[:, :, ::-1]
    result = np.zeros((cfg["slices"], cfg["size"], cfg["size"]), np.uint8)
    mask = np.zeros(cfg["slices"], bool)
    clahe = None
    if recipe["normalization"] == "clahe":
        import cv2
        clahe = cv2.createCLAHE(clipLimit=2., tileGridSize=(8, 8))
    for i, pixels in enumerate(stack):
        pixels = np.asarray(Image.fromarray(pixels).resize((cfg["size"], cfg["size"]), Image.Resampling.BOX))
        result[i] = clahe.apply(pixels) if clahe is not None else pixels
        mask[i] = True
    return result, mask, [records[int(i)]["position"] for i in selected]


def _folders(comp, uid):
    studies = [comp / root / uid for root in ("train_series", "train_images")]
    hits = [path for path in studies if path.is_dir()]
    if len(hits) > 1:
        raise ValueError("ambiguous_study_directories")
    if not hits:
        return []
    if not hits[0].resolve().is_relative_to(comp):
        raise ValueError("source_path_escapes_competition")
    result = sorted(p for p in hits[0].iterdir() if p.is_dir())
    for path in result:
        _component(path.name)
        if not path.resolve().is_relative_to(comp):
            raise ValueError("source_path_escapes_competition")
    return result


def _source_hash(folders, comp):
    records = []
    for folder in folders:
        records.append([folder.relative_to(comp).as_posix(), "directory"])
        for path in sorted(folder.iterdir()):
            if path.is_file():
                if not path.resolve().is_relative_to(comp):
                    raise ValueError("source_path_escapes_competition")
                records.append([path.relative_to(comp).as_posix(), sha256(path)])
    return fingerprint(records)


def _reuse(path, sidecar, image_id, source_hash, cfg):
    try:
        info = json.loads(sidecar.read_text(encoding="utf-8"))
        if info["image_id"] != image_id or info["source_hash"] != source_hash or not info["success"] or sha256(path) != info["sha256"]:
            return None
        with np.load(path, allow_pickle=False) as archive:
            images, mask = archive["images"], archive["mask"]
            if images.dtype != np.uint8 or images.shape != (3, cfg["slices"], cfg["size"], cfg["size"]) or mask.dtype != bool or mask.shape != (3, cfg["slices"]):
                return None
            if not mask.any() or np.any(images[~mask]) or int(mask.sum()) != info["valid_slices"]:
                return None
        return info
    except (OSError, ValueError, KeyError, TypeError, EOFError, zipfile.BadZipFile):
        return None


def _study(folders, metadata, uid, cfg):
    images = np.zeros((3, cfg["slices"], cfg["size"], cfg["size"]), np.uint8)
    mask = np.zeros((3, cfg["slices"]), bool)
    choices, errors, chosen = {}, [], {}
    for folder in folders:
        try:
            records = _headers(folder, metadata.get((uid, folder.name), {}), uid, folder.name, RECIPES[cfg["recipe"]])
            first = records[0]
            rank = tuple(-n for n in first["preference"]) + (-len(records), folder.name)
            choices.setdefault(first["plane"], []).append((rank, folder.name, records))
        except Exception as exc:
            errors.append(dict(series=folder.name, stage="header", error=f"{type(exc).__name__}: {exc}"))
    for slot, plane in enumerate(PLANES):
        if plane not in choices:
            continue
        _, series, records = min(choices[plane], key=lambda item: item[0])
        chosen[plane] = dict(series=series, positions=[], success=False,
                             laterality=records[0].get("laterality"),
                             laterality_source=records[0].get("laterality_source"))
        try:
            stack, valid, positions = _series_stack(records, cfg)
            images[slot], mask[slot] = stack, valid
            chosen[plane].update(positions=positions, success=True)
        except Exception as exc:
            errors.append(dict(series=series, stage="decode", error=f"{type(exc).__name__}: {exc}"))
    if not mask.any():
        errors.append(dict(series="", stage="study", error="no_usable_series"))
    return images, mask, dict(success=bool(mask.any()), valid_slices=int(mask.sum()), selected=chosen, errors=errors)


def build_cache(config) -> Path:
    """Build ``out_dir/recipe`` and return that directory, including on failure.

    Check cache_meta.json complete before training. All manifest rows survive;
    max_studies processes the first N rows and leaves explicit unprocessed index
    entries, making a truncated run incomplete. Relabeling identical study rows
    refreshes provenance without touching valid label-free NPZs. Different image
    configuration/cohort requires a new output directory. Single writer only.
    """
    started = time.monotonic()
    cfg = {key: config.get(key, default) for key, default in dict(recipe="p0", slices=16, size=192, max_studies=0).items()}
    if cfg["recipe"] not in RECIPES:
        raise ValueError("recipe must be p0 through p6")
    for key in ("slices", "size", "max_studies"):
        if isinstance(cfg[key], bool) or not isinstance(cfg[key], (int, np.integer)) or cfg[key] < (0 if key == "max_studies" else 1):
            raise ValueError(f"{key} must be a {'nonnegative' if key == 'max_studies' else 'positive'} integer")
        cfg[key] = int(cfg[key])
    if cfg["slices"] > 256 or cfg["size"] > 1024:
        raise ValueError("Bounded cache requires slices <= 256 and size <= 1024")
    comp = Path(config["comp_dir"]).resolve(strict=True)
    manifest = Path(locate("experiment_manifest.csv", roots=config.get("input_roots"), override=config.get("manifest_path"))).resolve()
    preparation_path = manifest.with_name("preparation_meta.json")
    preparation = json.loads(preparation_path.read_text(encoding="utf-8-sig"))
    if preparation.get("complete") is False:
        raise ValueError("Preparation is incomplete")
    for key in ("cohort_id", "supervision_id"):
        if not preparation.get(key):
            raise ValueError("Preparation missing " + key)
    fold = preparation.get("fold", preparation.get("config", {}).get("fold"))
    if isinstance(fold, bool) or not isinstance(fold, int) or fold < 0:
        raise ValueError("Preparation must specify a nonnegative integer fold")
    manifest_hash = sha256(manifest)
    expected = preparation.get("manifest_sha256", preparation.get("experiment_manifest_sha256"))
    if expected is not None and expected != manifest_hash:
        raise ValueError("Preparation manifest SHA256 mismatch")
    rows, fields = _read_csv(manifest)
    if UID not in fields or not rows:
        raise ValueError("Prepared manifest must contain study rows and " + UID)
    uids = [_component(row[UID]) for row in rows]
    if len(set(uids)) != len(uids):
        raise ValueError("Duplicate study in prepared manifest")
    metadata, series_hash = {}, None
    if (comp / "train_series.csv").exists():
        series_rows, series_fields = _read_csv(comp / "train_series.csv")
        if not {UID, "SeriesInstanceUID"}.issubset(series_fields):
            raise ValueError("train_series.csv is missing UID columns")
        for row in series_rows:
            key = (row[UID], row["SeriesInstanceUID"])
            if key in metadata and metadata[key] != row:
                raise ValueError("Conflicting train_series.csv rows")
            metadata[key] = row
        series_hash = sha256(comp / "train_series.csv")
    image_config = {**cfg, "comp_dir": str(comp), "version": VERSION}
    identity = dict(config=image_config, study_ids=uids, series_metadata_sha256=series_hash)
    image_id = fingerprint(identity)
    out = Path(config["out_dir"]).resolve() / cfg["recipe"]
    if out.is_relative_to(comp) or manifest.is_relative_to(out):
        raise ValueError("Output must be separate from competition/preparation inputs")
    out.mkdir(parents=True, exist_ok=True)
    lock = out / ".build.lock"
    try:
        handle = lock.open("x")
    except FileExistsError as exc:
        raise RuntimeError("Cache already has a writer; inspect .build.lock before retrying") from exc
    try:
        with handle:
            return _build(out, comp, manifest, preparation_path, preparation, rows, fields, uids,
                          metadata, cfg, config, image_config, identity, image_id, manifest_hash,
                          series_hash, started)
    finally:
        lock.unlink()


def _build(out, comp, manifest, preparation_path, preparation, rows, fields, uids,
           metadata, cfg, config, image_config, identity, image_id, manifest_hash,
           series_hash, started):
    identity_path = out / "image_identity.json"
    if identity_path.exists():
        if json.loads(identity_path.read_text(encoding="utf-8")).get("image_id") != image_id:
            raise ValueError("Cache config/cohort mismatch; choose a new out_dir")
    elif any(p.name != ".build.lock" for p in out.iterdir()):
        raise ValueError("Nonempty cache has no verifiable identity")
    count = min(len(rows), cfg["max_studies"] or len(rows))
    per_study = 3 * cfg["slices"] * (cfg["size"]**2 + 1) + 65536
    estimate = math.ceil(per_study * count * 1.05)
    budget = float(config.get("max_cache_gb", 30))
    if not math.isfinite(budget) or budget <= 0 or estimate > budget * 1e9:
        raise OSError("Estimated cache exceeds max_cache_gb")
    existing = sum(p.stat().st_size for p in (out / "studies").glob("*.npz"))
    if shutil.disk_usage(out).free < max(per_study * 2, estimate - existing) + 64 * 1024**2:
        raise OSError("Insufficient free disk for bounded estimate and working reserve")
    write_json(identity_path, {**identity, "image_id": image_id})
    meta = dict(schema=SCHEMA, cache_id=None, image_id=image_id, cohort_id=preparation["cohort_id"],
                supervision_id=preparation["supervision_id"], fold=preparation.get("fold", preparation.get("config", {}).get("fold")),
                config={**image_config, "manifest_path": str(manifest), "out_dir": str(out)},
                complete=False, estimated_bytes=estimate, source_manifest_sha256=manifest_hash,
                preparation_sha256=sha256(preparation_path), planes=list(PLANES),
                geometry="native in-plane orientation; geometrically ordered sampled slice stack; not isotropic",
                normalization_scope="selected series stack", environment=dict(python=platform.python_version(), pydicom=pydicom.__version__, numpy=np.__version__))
    write_json(out / "cache_meta.json", meta)
    _csv(out / "cache_manifest.csv", rows, fields)
    cache = out / "studies"
    cache.mkdir(exist_ok=True)
    index, audits, source_hashes = [], [], {}
    resumed = failed = 0
    for number, uid in enumerate(uids):
        if number >= count:
            index.append({UID: uid, "file": "", "sha256": ""})
            audits.append({UID: uid, "status": "unprocessed", "errors": []})
            continue
        tick = time.monotonic()
        name = fingerprint(uid) + ".npz"
        path, sidecar = cache / name, cache / (name + ".json")
        try:
            folders = _folders(comp, uid)
            signature = _source_hash(folders, comp)
            source_hashes[uid] = signature
            info = _reuse(path, sidecar, image_id, signature, cfg)
            if info is not None:
                resumed += 1
            else:
                images, mask, info = _study(folders, metadata, uid, cfg)
                temporary = path.with_suffix(".npz.tmp")
                with temporary.open("wb") as stream:
                    np.savez_compressed(stream, images=images, mask=mask)
                temporary.replace(path)
                info.update(image_id=image_id, source_hash=signature, sha256=sha256(path))
                write_json(sidecar, info)
            index.append({UID: uid, "file": path.relative_to(out).as_posix(), "sha256": info["sha256"]})
            audits.append({UID: uid, "status": "ok" if info["success"] else "failed", **info})
            failed += int(not info["success"])
        except Exception as exc:
            failed += 1
            index.append({UID: uid, "file": "", "sha256": ""})
            audits.append({UID: uid, "status": "failed", "errors": [f"{type(exc).__name__}: {exc}"]})
        if audits[-1]["status"] == "failed":
            LOG.error("Study %s failed: %s", uid, audits[-1]["errors"])
        elif audits[-1].get("errors"):
            LOG.warning("Study %s series audit: %s", uid, audits[-1]["errors"])
        if number == 0 or (number + 1) % 25 == 0 or number + 1 == count:
            LOG.info("MRI cache %s %d/%d; failed=%d resumed=%d study=%.2fs elapsed=%.1fs", cfg["recipe"], number + 1, count, failed, resumed, time.monotonic()-tick, time.monotonic()-started)
    _csv(out / "cache_index.csv", index, [UID, "file", "sha256"])
    write_json(out / "cache_audit.json", audits)
    meta.update(manifest_sha256=sha256(out / "cache_manifest.csv"), index_sha256=sha256(out / "cache_index.csv"),
                complete=failed == 0 and count == len(rows),
                counts=dict(studies=len(rows), processed=count, failed=failed, unprocessed=len(rows)-count, resumed=resumed),
                seconds=round(time.monotonic()-started, 3), source_hashes=source_hashes)
    meta["cache_id"] = fingerprint(dict(config=image_config, source_hashes=source_hashes,
                                       series_metadata_sha256=series_hash, manifest_sha256=manifest_hash,
                                       preparation_sha256=meta["preparation_sha256"]))
    write_json(out / "cache_meta.json", meta)
    return out
