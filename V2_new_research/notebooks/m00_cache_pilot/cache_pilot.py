"""CPU-only M00 window-cache feasibility pilot; no labels, fitting or submission.

Two fixed seeded training draws per study alter epoch-wise random-window sampling relative to
dynamic M00. That is a pretraining input-policy choice requiring review, NOT exact equivalence.
Cache lives in /kaggle/working scratch and is deleted in `finally`. Output: one aggregate JSON.
"""

import collections
import csv
import hashlib
import json
import multiprocessing as mp
import os
import resource
import shutil
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor, wait
from pathlib import Path

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
WORKING = Path("/kaggle/working")
CACHE = WORKING / "cache_pilot_scratch"
OUT = WORKING / "cache_pilot_summary.json"
WORKERS, N_STUDIES, N_INFER, N_VERIFY, DRAWS, INFER_N = 4, 100, 10, 10, 2, 48
TRAIN_STUDIES, VAL_STUDIES, HEADROOM, EPOCHS = 3457, 885, 1.5, 8
TOTAL_BUDGET_S = 1140  # cooperative phase-start bound; running workers can exceed it
PHASE_CAP_S = {"build": 420, "infer": 180, "verify": 120, "read": 120, "dicom": 300}
MAX_CACHE_GIB = 5.0

DATA = next((p for p in ROOTS if (p / "train.csv").exists()), None)
if DATA is None:
    raise FileNotFoundError("Official competition input missing")


def read_rows(name):
    with (DATA / name).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def name_of(uid):
    return hashlib.sha256(uid.encode()).hexdigest()[:20]


def draw_rng(uid, k):
    return np.random.default_rng(int(hashlib.sha256(f"{uid}|draw{k}".encode()).hexdigest()[:16], 16))


def decode(uid):
    role_data = {}
    for role, sid in select_series(SERIES_BY_STUDY[uid]).items():
        if sid is None:
            continue
        paths = sorted((DATA / "train_series" / uid / sid).glob("*.dcm"))
        role_data[role] = load_series([pydicom.dcmread(p) for p in paths])
    return role_data


def train_arrays(role_data, uid, k):
    x, m = build_study(role_data, train=True, rng=draw_rng(uid, k))
    assert tuple(x.shape) == (3, 4, 3, SIZE, SIZE) and tuple(m.shape) == (3, 4)
    return {f"x{k}": x.numpy(), f"m{k}": m.numpy()}


def infer_arrays(role_data):
    xs, ms = [], []
    for r in ROLES:
        s = role_data.get(r)
        x, m = series_windows(*s, INFER_N) if s else (
            torch.zeros(INFER_N, 3, SIZE, SIZE), torch.zeros(INFER_N, dtype=torch.bool))
        xs.append(x), ms.append(m)
    mask = torch.stack(ms)
    if not mask.any():
        raise StudyError("study has no valid window in any role")
    return {"xi": torch.stack(xs).numpy(), "mi": mask.numpy()}


def rss_mib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # Linux: KiB


def fail(exc, t0):
    return {"ok": False, "error": type(exc).__name__, "seconds": time.perf_counter() - t0}


def build_train(uid):
    torch.set_num_threads(1)
    t0, c0 = time.perf_counter(), time.process_time()
    try:
        role_data = decode(uid)
        arrays = {}
        for k in range(DRAWS):
            arrays.update(train_arrays(role_data, uid, k))
        path = CACHE / f"{name_of(uid)}_train.npz"
        np.savez_compressed(path, **arrays)
        return {"ok": True, "bytes": os.path.getsize(path), "seconds": time.perf_counter() - t0,
                "cpu": time.process_time() - c0, "rss_mib": rss_mib()}
    except Exception as exc:
        return fail(exc, t0)


def build_infer(uid):
    torch.set_num_threads(1)
    t0, c0 = time.perf_counter(), time.process_time()
    try:
        arrays = infer_arrays(decode(uid))
        path = CACHE / f"{name_of(uid)}_infer.npz"
        np.savez_compressed(path, **arrays)
        return {"ok": True, "bytes": os.path.getsize(path), "seconds": time.perf_counter() - t0,
                "cpu": time.process_time() - c0, "rss_mib": rss_mib()}
    except Exception as exc:
        return fail(exc, t0)


def verify(uid):
    """Recompute everything from DICOM and compare bitwise with the cached arrays."""
    torch.set_num_threads(1)
    t0 = time.perf_counter()
    try:
        role_data = decode(uid)
        fresh = {"train": {}, "infer": infer_arrays(role_data)}
        for k in range(DRAWS):
            fresh["train"].update(train_arrays(role_data, uid, k))
        digest, equal, count = hashlib.sha256(), True, 0
        for kind, arrays in (("train", fresh["train"]), ("infer", fresh["infer"])):
            with np.load(CACHE / f"{name_of(uid)}_{kind}.npz") as z:
                if set(z.files) != set(arrays):
                    equal = False
                for key in sorted(arrays):
                    a = arrays[key]
                    b = z[key]
                    digest.update(key.encode() + str(a.dtype).encode() + str(a.shape).encode() + a.tobytes())
                    equal &= (a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes())
                    count += 1
        return {"ok": True, "equal": bool(equal), "digest": digest.hexdigest(), "arrays": count,
                "seconds": time.perf_counter() - t0}
    except Exception as exc:
        return fail(exc, t0)


def read_cached(arg):
    uid, k = arg
    t0, c0 = time.perf_counter(), time.process_time()
    try:
        with np.load(CACHE / f"{name_of(uid)}_train.npz") as z:
            x, m = z[f"x{k}"], z[f"m{k}"]
        assert x.shape == (3, 4, 3, SIZE, SIZE) and m.shape == (3, 4)
        return {"ok": True, "seconds": time.perf_counter() - t0, "cpu": time.process_time() - c0,
                "rss_mib": rss_mib()}
    except Exception as exc:
        return fail(exc, t0)


def dicom_on_demand(arg):
    uid, k = arg
    torch.set_num_threads(1)
    t0, c0 = time.perf_counter(), time.process_time()
    try:
        train_arrays(decode(uid), uid, k)
        return {"ok": True, "seconds": time.perf_counter() - t0, "cpu": time.process_time() - c0,
                "rss_mib": rss_mib()}
    except Exception as exc:
        return fail(exc, t0)


def run_pool(fn, items, phase, start):
    """Four forked workers; cancel queued work after the budget, then wait for running jobs."""
    budget = min(PHASE_CAP_S[phase], TOTAL_BUDGET_S - (time.perf_counter() - start))
    if budget <= 0:
        return [None] * len(items), 0.0
    t0 = time.perf_counter()
    pool = ProcessPoolExecutor(max_workers=WORKERS, mp_context=mp.get_context("fork"))
    futures = [pool.submit(fn, item) for item in items]
    wait(futures, timeout=budget)
    pool.shutdown(wait=True, cancel_futures=True)
    wall = time.perf_counter() - t0
    out = []
    for f in futures:
        if f.cancelled() or not f.done():
            out.append(None)
            continue
        try:
            out.append(f.result())
        except Exception as exc:
            out.append({"ok": False, "error": type(exc).__name__, "seconds": 0.0})
    return out, wall


def pct95(values):
    return sorted(values)[int(0.95 * (len(values) - 1))]


def summarize(outcomes, wall):
    done = [o for o in outcomes if o is not None]
    good = [o for o in done if o["ok"]]
    s = {"workers": WORKERS, "requested": len(outcomes), "not_run": len(outcomes) - len(done),
         "succeeded": len(good), "failed": len(done) - len(good), "wall_seconds": wall,
         "failures_by_exception_type": dict(collections.Counter(o["error"] for o in done if not o["ok"]))}
    if good:
        s["wall_seconds_per_study"] = wall / len(good)
        s["studies_per_hour"] = len(good) * 3600 / wall
        s["median_study_seconds"] = statistics.median(o["seconds"] for o in good)
        if "cpu" in good[0]:
            s["cpu_seconds_per_study_mean"] = statistics.mean(o["cpu"] for o in good)
        if "rss_mib" in good[0]:
            s["max_worker_peak_rss_mib"] = max(o["rss_mib"] for o in good)  # fork may inherit parent RSS
    return s


def size_stats(outcomes):
    sizes = [o["bytes"] for o in outcomes if o and o["ok"]]
    if not sizes:
        return None
    gib = lambda b: b / 2**30
    stats = {"median": statistics.median(sizes), "p95": pct95(sizes), "max": max(sizes),
             "mean": statistics.mean(sizes)}
    return {
        "bytes_per_study": stats,
        "projected_gib": {key: {"train_3457": gib(v * TRAIN_STUDIES), "validation_885": gib(v * VAL_STUDIES)}
                          for key, v in stats.items()},
    }


def cache_gib():
    return sum(p.stat().st_size for p in CACHE.glob("*.npz")) / 2**30


start = time.perf_counter()
output = {
    "scope": ("CPU-only cache feasibility pilot: 100 fixed nonexpert studies (same selection as loader probe), "
              "all selected series decoded; 2 fixed seeded training draws/study ([3,4,3,224,224] float32, "
              "mask [3,4]) saved via np.savez_compressed; 48-window inference arrays for 10 studies sized "
              "separately; four workers"),
    "policy_warning": ("Two fixed draws per study alter epoch-wise random-window sampling; this is a "
                       "pretraining input-policy choice requiring review and is NOT exactly equivalent to "
                       "dynamic M00."),
    "cpu_count": os.cpu_count(),
    "python": sys.version.split()[0],
    "working_disk_free_gib_start": shutil.disk_usage(WORKING).free / 2**30,
    "estimates_provisional": True,
    "gpu_runtime_note": "No GPU runtime estimate can be derived from this CPU-only measurement.",
}
try:
    train, series = read_rows("train.csv"), read_rows("train_series.csv")
    targets = [key for key in train[0] if key not in {"StudyInstanceUID", "Report"}]
    expert = {r["StudyInstanceUID"] for r in train if all(r[t] in {"0", "1"} for t in targets)}
    norm = lambda text: " ".join(text.casefold().split())
    expert_reports = {norm(r["Report"]) for r in train if r["StudyInstanceUID"] in expert}
    eligible = {r["StudyInstanceUID"] for r in train
                if r["StudyInstanceUID"] not in expert and norm(r["Report"]) not in expert_reports}
    del train
    SERIES_BY_STUDY = {}
    for row in series:
        SERIES_BY_STUDY.setdefault(row["StudyInstanceUID"], []).append(row)
    selected = sorted(eligible, key=lambda uid: hashlib.sha256(uid.encode()).hexdigest())[:N_STUDIES]
    CACHE.mkdir(parents=True, exist_ok=True)

    outcomes, wall = run_pool(build_train, selected, "build", start)
    output["cache_build_train"] = summarize(outcomes, wall)
    output["cache_build_train"]["size"] = size_stats(outcomes)
    built = [u for u, o in zip(selected, outcomes) if o and o["ok"]]
    output["cache_disk_gib_after_train_build"] = cache_gib()

    infer_set = built[:N_INFER]
    if cache_gib() < MAX_CACHE_GIB:
        out_i, wall_i = run_pool(build_infer, infer_set, "infer", start)
        output["cache_build_infer_48_windows"] = summarize(out_i, wall_i)
        output["cache_build_infer_48_windows"]["size"] = size_stats(out_i)
        output["cache_build_infer_48_windows"]["note"] = (
            "10 studies over 4 workers: tail effects inflate wall per study; separate decode from train build")
    else:
        out_i = []
        output["cache_build_infer_48_windows"] = {"skipped": "cache disk guard"}
    output["cache_disk_gib_peak"] = cache_gib()

    infer_ok = [u for u, o in zip(infer_set, out_i) if o and o["ok"]]
    out_v, wall_v = run_pool(verify, infer_ok[:N_VERIFY], "verify", start)
    ver = summarize(out_v, wall_v)
    good_v = [o for o in out_v if o and o["ok"]]
    ver.update({
        "studies_compared": len(good_v),
        "all_bitwise_equal": bool(good_v) and all(o["equal"] for o in good_v),
        "arrays_compared": sum(o["arrays"] for o in good_v),
        "aggregate_sha256": hashlib.sha256("".join(sorted(o["digest"] for o in good_v)).encode()).hexdigest(),
        "meets_minimum_10_studies": len(good_v) >= 10,
    })
    output["bitwise_verification"] = ver

    items = [(u, i % DRAWS) for i, u in enumerate(built)]
    out_r, wall_r = run_pool(read_cached, items, "read", start)
    output["compressed_read_one_draw_per_study"] = summarize(out_r, wall_r)
    output["compressed_read_one_draw_per_study"]["note"] = (
        "files just written, so page cache is likely warm: optimistic vs a cold-cache epoch")
    out_d, wall_d = run_pool(dicom_on_demand, items, "dicom", start)
    output["dicom_on_demand_one_draw_per_study"] = summarize(out_d, wall_d)

    est = {"provisional": True, "headroom": HEADROOM}
    b, i = output["cache_build_train"], output["cache_build_infer_48_windows"]
    if "wall_seconds_per_study" in b and "wall_seconds_per_study" in i:
        est["full_build_hours_with_headroom"] = HEADROOM * (
            TRAIN_STUDIES * b["wall_seconds_per_study"] + VAL_STUDIES * i["wall_seconds_per_study"]) / 3600
    for key, label in (("compressed_read_one_draw_per_study", "cached"),
                       ("dicom_on_demand_one_draw_per_study", "dicom")):
        s = output[key]
        if "wall_seconds_per_study" in s:
            est[f"{label}_eight_epoch_read_hours_with_headroom"] = (
                HEADROOM * EPOCHS * TRAIN_STUDIES * s["wall_seconds_per_study"] / 3600)
    output["estimates"] = est
except Exception as exc:
    output["fatal_error_type"] = type(exc).__name__  # type only: messages may carry paths/UIDs
finally:
    shutil.rmtree(CACHE, ignore_errors=True)
    output["cache_deleted"] = not CACHE.exists()
    output["elapsed_seconds"] = time.perf_counter() - start
    output["parent_peak_rss_mib"] = rss_mib()
    OUT.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))
if "fatal_error_type" in output or not output["cache_deleted"]:
    sys.exit(1)
