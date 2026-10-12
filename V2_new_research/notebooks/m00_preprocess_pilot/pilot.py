"""Bounded CPU-only real DICOM preprocessing pilot; no labels, fitting or submission."""

import csv
import hashlib
import json
import statistics
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pydicom
import torch

# Embedded M00 data source SHA256: 84190f34e7f4f25518f2f0855104c1c8514bb6e0f4789c1e6e9d12d4803811c5
"""M00 data path: series selection, DICOM ordering/scaling, window sampling, study tensor.

Offline, deterministic, no flips. Failures raise SeriesError (series-level) or StudyError
(study-level) so callers can count them; nothing is silently substituted.
"""
import numpy as np
import torch
import torch.nn.functional as F

ROLES = ("Sagittal", "Coronal", "Axial")
TRAIN_WINDOWS, INFER_WINDOWS, SIZE = 4, 16, 224
TOL_MM = 1e-3


class SeriesError(ValueError):
    pass


class StudyError(ValueError):
    pass


def select_series(rows):
    """rows: iterable of dicts with SeriesInstanceUID, Fluid_Sensitive, Fat_Suppression,
    Anatomical_Plane (official train_series.csv fields) for ONE study.
    Returns {role: SeriesInstanceUID or None}. Deterministic.
    ponytail: preference fluid-sensitive > fat-suppressed > smallest UID is MY choice, not a
    clinical rule; revisit after the A05 role study."""
    best = {r: None for r in ROLES}
    for row in rows:
        role = str(row["Anatomical_Plane"]).strip().capitalize()
        if role not in best:
            continue
        key = (-int(row["Fluid_Sensitive"] or 0), -int(row["Fat_Suppression"] or 0),
               str(row["SeriesInstanceUID"]))
        if best[role] is None or key < best[role][0]:
            best[role] = (key, str(row["SeriesInstanceUID"]))
    return {r: (v[1] if v else None) for r, v in best.items()}


def slice_position(ds):
    """Projection of ImagePositionPatient on row x col normal; None if tags missing."""
    iop, ipp = getattr(ds, "ImageOrientationPatient", None), getattr(ds, "ImagePositionPatient", None)
    if iop is None or ipp is None:
        return None
    iop, ipp = np.asarray(iop, float), np.asarray(ipp, float)
    if iop.shape != (6,) or ipp.shape != (3,) or not (np.isfinite(iop).all() and np.isfinite(ipp).all()):
        raise SeriesError("malformed orientation/position tags")
    r, c = iop[:3], iop[3:]
    if abs(np.linalg.norm(r) - 1) > 1e-2 or abs(np.linalg.norm(c) - 1) > 1e-2 or abs(r @ c) > 1e-2:
        raise SeriesError("ImageOrientationPatient not orthonormal")
    return float(ipp @ np.cross(r, c)), tuple(np.round(np.cross(r, c), 3))


def order_slices(datasets):
    """Sort datasets by physical position. Fallback to InstanceNumber ONLY if every slice lacks
    orientation/position tags; mixed presence, duplicates, irregular gaps or in-plane changes raise."""
    if not datasets:
        raise SeriesError("empty series")
    spacings = []
    for ds in datasets:
        raw = getattr(ds, "PixelSpacing", None)
        if raw is None:
            raise SeriesError("missing PixelSpacing")
        spacing = np.asarray(raw, dtype=float)
        if spacing.shape != (2,) or not np.isfinite(spacing).all() or np.any(spacing <= 0):
            raise SeriesError("invalid PixelSpacing")
        spacings.append(spacing)
    if any(np.max(np.abs(spacing - spacings[0])) > 1e-3 for spacing in spacings[1:]):
        raise SeriesError("pixel spacing differs within series")
    pos = [slice_position(d) for d in datasets]
    if all(p is None for p in pos):
        nums = [getattr(d, "InstanceNumber", None) for d in datasets]
        if any(n is None for n in nums) or len(set(int(n) for n in nums)) != len(nums):
            raise SeriesError("no geometry tags and InstanceNumber missing/duplicated")
        keys = [float(n) for n in nums]
    elif any(p is None for p in pos):
        raise SeriesError("geometry tags present on only some slices")
    else:
        if len({p[1] for p in pos}) != 1:
            raise SeriesError("slice normal differs within series")
        orientation = np.asarray(datasets[0].ImageOrientationPatient, dtype=float)
        if any(np.max(np.abs(np.asarray(ds.ImageOrientationPatient, dtype=float) - orientation)) > 1e-3
               for ds in datasets[1:]):
            raise SeriesError("in-plane orientation differs within series")
        keys = [p[0] for p in pos]
    order = np.argsort(keys, kind="stable")
    s = np.asarray(keys)[order]
    if len(s) > 1 and np.diff(s).min() < TOL_MM:
        raise SeriesError("duplicate slice positions")
    if len(s) > 2:
        gaps = np.diff(s)
        if gaps.max() > 1.5 * np.median(gaps) + 0.01:
            raise SeriesError("irregular slice gap")
    return [datasets[i] for i in order]


def slice_pixels(ds):
    """float32 pixels after rescale and MONOCHROME1 inversion (negation; scaling is affine)."""
    a = ds.pixel_array.astype(np.float32)
    if a.ndim != 2:
        raise SeriesError("non-2D pixel array")
    a = a * float(getattr(ds, "RescaleSlope", 1.0)) + float(getattr(ds, "RescaleIntercept", 0.0))
    if str(getattr(ds, "PhotometricInterpretation", "MONOCHROME2")) == "MONOCHROME1":
        a = -a
    return a


def load_series(datasets, lo_q=1.0, hi_q=99.0):
    """-> (volume [N,H,W] float32 in [0,1], slice_ok [N] bool). Slices with nonfinite values or
    constant content are flagged not-ok (zero-filled, never used in a window)."""
    ds = order_slices(datasets)
    arrs = [slice_pixels(d) for d in ds]
    if len({a.shape for a in arrs}) != 1:
        raise SeriesError("inconsistent slice shapes")
    ok = np.array([np.isfinite(a).all() and a.max() > a.min() for a in arrs])
    if ok.sum() < 3:
        raise SeriesError("fewer than 3 usable slices")
    # Bound percentile scratch memory for large series while sampling each slice
    # deterministically. Full decoded slices are still used for final scaling.
    vals = np.concatenate([
        a.ravel()[::max(1, a.size // 16384)] for a, k in zip(arrs, ok) if k
    ])
    lo, hi = np.percentile(vals, [lo_q, hi_q])
    if not hi > lo:
        raise SeriesError("degenerate intensity range")
    vol = np.stack([np.clip((a - lo) / (hi - lo), 0, 1) if k else np.zeros_like(a)
                    for a, k in zip(arrs, ok)]).astype(np.float32)
    return vol, ok


def window_centers(ok, n, rng=None):
    """Centers c such that slices c-1,c,c+1 are all usable (physically adjacent in sorted order).
    rng None -> deterministic evenly spaced (inference); else random without replacement (train)."""
    valid = np.flatnonzero(ok[:-2] & ok[1:-1] & ok[2:]) + 1
    if len(valid) == 0:
        return valid
    if rng is None:
        return np.unique(valid[np.round(np.linspace(0, len(valid) - 1, n)).astype(int)])
    return np.sort(rng.choice(valid, size=min(n, len(valid)), replace=False))


def series_windows(vol, ok, n, rng=None):
    """-> (x [n,3,224,224], mask [n]); unused rows zero with mask False."""
    x = torch.zeros(n, 3, SIZE, SIZE)
    m = torch.zeros(n, dtype=torch.bool)
    for i, c in enumerate(window_centers(ok, n, rng)):
        w = torch.from_numpy(vol[c - 1:c + 2])[None]
        x[i] = F.interpolate(w, size=(SIZE, SIZE), mode="bilinear", align_corners=False)[0]
        m[i] = True
    return x, m


def build_study(role_series, train, rng=None):
    """role_series: {role: (vol, ok) or None}. -> (x [3,W,3,224,224], mask [3,W]).
    Raises StudyError if no role yields a valid window."""
    n = TRAIN_WINDOWS if train else INFER_WINDOWS
    xs, ms = [], []
    for r in ROLES:
        s = role_series.get(r)
        x, m = series_windows(*s, n, rng if train else None) if s else (
            torch.zeros(n, 3, SIZE, SIZE), torch.zeros(n, dtype=torch.bool))
        xs.append(x), ms.append(m)
    mask = torch.stack(ms)
    if not mask.any():
        raise StudyError("study has no valid window in any role")
    return torch.stack(xs), mask


ROOTS = (
    Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"),
    Path("/kaggle/input/rsna-knee-abnormality-detection"),
)
DATA = next((p for p in ROOTS if (p / "train.csv").exists()), None)
if DATA is None:
    raise FileNotFoundError("Official competition input missing")
OUT = Path("/kaggle/working/preprocess_pilot_summary.json")
MAX_STUDIES = 20
MAX_SCRIPT_SECONDS = 900
start = time.perf_counter()


def read_rows(name):
    with (DATA / name).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


train, series = read_rows("train.csv"), read_rows("train_series.csv")
target_names = [key for key in train[0] if key not in {"StudyInstanceUID", "Report"}]
experts = {r["StudyInstanceUID"] for r in train if all(r[t] in {"0", "1"} for t in target_names)}
expert_reports = {" ".join(r["Report"].casefold().split()) for r in train if r["StudyInstanceUID"] in experts}
eligible = {r["StudyInstanceUID"] for r in train if r["StudyInstanceUID"] not in experts and " ".join(r["Report"].casefold().split()) not in expert_reports}
series_by_study = {}
for row in series:
    series_by_study.setdefault(row["StudyInstanceUID"], []).append(row)
selected = sorted(eligible, key=lambda uid: hashlib.sha256(uid.encode()).hexdigest())[:MAX_STUDIES]

timings, role_counts, gap_ratios, max_gap_mm = [], Counter(), [], []
failures = Counter()
total_slices = 0
processed = 0
for uid in selected:
    if time.perf_counter() - start > MAX_SCRIPT_SECONDS:
        failures["time_budget_stop"] += MAX_STUDIES - processed
        break
    t0 = time.perf_counter()
    roles = select_series(series_by_study[uid])
    prepared = {}
    try:
        for role, sid in roles.items():
            if sid is None:
                continue
            paths = sorted((DATA / "train_series" / uid / sid).glob("*.dcm"))
            if not paths:
                raise SeriesError("empty selected series")
            datasets = [pydicom.dcmread(path) for path in paths]
            total_slices += len(datasets)
            ordered = order_slices(datasets)
            positions = [slice_position(ds) for ds in ordered]
            if all(p is not None for p in positions) and len(positions) > 1:
                gaps = np.diff([p[0] for p in positions])
                median_gap = float(np.median(gaps))
                if median_gap > 0:
                    gap_ratios.append(float(gaps.max() / median_gap))
                    max_gap_mm.append(float(gaps.max()))
            prepared[role] = load_series(datasets)
            role_counts[role] += 1
        x, mask = build_study(prepared, train=False)
        if x.shape != (3, 16, 3, 224, 224) or not mask.any() or not torch.isfinite(x[mask]).all():
            raise ValueError("invalid M00 study tensor")
        timings.append(time.perf_counter() - t0)
    except Exception as exc:
        failures[type(exc).__name__] += 1
    processed += 1

summary = {
    "scope": "20 deterministic nonexpert studies; full selected-series DICOM decode and M00 48-window CPU tensor build; no model or labels",
    "selected_studies": len(selected),
    "processed_studies": processed,
    "successful_studies": len(timings),
    "failed_studies": dict(failures),
    "total_decoded_slices_attempted": total_slices,
    "roles_built": dict(role_counts),
    "seconds_per_successful_study_min_median_max": [min(timings), statistics.median(timings), max(timings)] if timings else [],
    "max_to_median_slice_gap_ratio_max": max(gap_ratios) if gap_ratios else None,
    "series_with_gap_ratio_gt_1p5": sum(r > 1.5 for r in gap_ratios),
    "series_with_measured_gaps": len(gap_ratios),
    "max_slice_gap_mm": max(max_gap_mm) if max_gap_mm else None,
    "elapsed_seconds": round(time.perf_counter() - start, 2),
    "engineering_only": True,
}
OUT.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2))
