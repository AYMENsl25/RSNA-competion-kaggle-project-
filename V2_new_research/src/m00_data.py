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
