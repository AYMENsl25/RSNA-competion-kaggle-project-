"""CPU-only audit of two M00 cache designs; no labels, fitting or submission.

Design A (float32_volume): per-role resized float32 volumes + slice_ok, lossless. Windows are still
  drawn at read time, so dynamic window sampling is preserved.
Design B (uint16_two_draws): two FIXED seeded training draws quantized to uint16. LOSSY input storage and
  a policy change (no dynamic sampling); never exact parity.
Cache lives in /kaggle/working scratch and is deleted in `finally`. Output: one aggregate JSON.
Time limits are cooperative (phase-start budgets): running workers are not killed, so the 20-minute
target can be overrun; it is not a hard wall cap.
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

# __M00_DATA_SOURCE__

ROOTS = (
    Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"),
    Path("/kaggle/input/rsna-knee-abnormality-detection"),
)
WORKING = Path("/kaggle/working")
CACHE = WORKING / "cache_alternatives_scratch"
OUT = WORKING / "cache_alternatives_summary.json"
WORKERS, N_STUDIES, N_PARITY, DRAWS = 4, 100, 10, 2
TRAIN_STUDIES, VAL_STUDIES, HEADROOM, EPOCHS = 3457, 885, 1.5, 8
FREE_GIB, RESERVE_GIB = 19.5, 4.0
TOTAL_BUDGET_S = 1100  # cooperative phase-start bound; running workers can exceed it
PHASE_CAP_S = {"build": 330, "parity": 100, "read": 80}
MAX_CACHE_GIB = 5.0  # per design; the first design's files are deleted before the second is built
Q = 65535
ERR_BOUND = 0.5 / Q + 1e-6
BINS, BIN_W = 2001, 1e-5 / 2000  # error histogram (exact aggregate p99 to one bin width)

DATA = next((p for p in ROOTS if (p / "train.csv").exists()), None)
if DATA is None:
    raise FileNotFoundError("Official competition input missing")
DIR = CACHE  # per-design cache directory; reassigned before each pool is forked


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


def rss_mib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # Linux: KiB


def fail(exc, t0):
    return {"ok": False, "error": type(exc).__name__, "seconds": time.perf_counter() - t0}


def meta(t0, c0):
    return {"seconds": time.perf_counter() - t0, "cpu": time.process_time() - c0,
            "rss_mib": rss_mib(), "pid": os.getpid()}


def dir_gib(path):
    return sum(p.stat().st_size for p in path.glob("*.npz")) / 2**30


def guard_and_save(path, arrays):
    if dir_gib(DIR) > MAX_CACHE_GIB:
        raise MemoryError("cache disk guard")
    np.savez_compressed(path, **arrays)
    return os.path.getsize(path)


# ---- Design A: resized float32 volumes -----------------------------------------------------
def resize_volume(vol, ok):
    out = np.zeros((vol.shape[0], SIZE, SIZE), np.float32)
    idx = np.flatnonzero(ok)
    for s in range(0, len(idx), 16):
        sel = idx[s:s + 16]
        t = F.interpolate(torch.from_numpy(vol[sel])[:, None], size=(SIZE, SIZE),
                          mode="bilinear", align_corners=False)
        out[sel] = t[:, 0].numpy()
    return out


def volume_arrays(role_data):
    arrays = {}
    for role, (vol, ok) in role_data.items():
        arrays[f"v_{role}"], arrays[f"ok_{role}"] = resize_volume(vol, ok), ok
    return arrays


def volumes_from_npz(z):
    return {r: (z[f"v_{r}"], z[f"ok_{r}"]) for r in ROLES if f"v_{r}" in z.files}


def recon_study(volumes, train, rng=None):
    """Mirror of build_study using already-resized volumes (no DICOM, no interpolation)."""
    n = TRAIN_WINDOWS if train else INFER_WINDOWS
    xs, ms = [], []
    for r in ROLES:
        x = torch.zeros(n, 3, SIZE, SIZE)
        m = torch.zeros(n, dtype=torch.bool)
        if r in volumes:
            res, ok = volumes[r]
            for i, c in enumerate(window_centers(ok, n, rng if train else None)):
                x[i] = torch.from_numpy(res[c - 1:c + 2])
                m[i] = True
        xs.append(x), ms.append(m)
    mask = torch.stack(ms)
    if not mask.any():
        raise StudyError("study has no valid window in any role")
    return torch.stack(xs), mask


def build_volume(uid):
    torch.set_num_threads(1)
    t0, c0 = time.perf_counter(), time.process_time()
    try:
        size = guard_and_save(DIR / f"{name_of(uid)}.npz", volume_arrays(decode(uid)))
        return {"ok": True, "bytes": size, **meta(t0, c0)}
    except Exception as exc:
        return fail(exc, t0)


def parity_volume(uid):
    torch.set_num_threads(1)
    t0 = time.perf_counter()
    try:
        role_data = decode(uid)
        with np.load(DIR / f"{name_of(uid)}.npz") as z:
            volumes = volumes_from_npz(z)
        equal, max_err, count = True, 0.0, 0
        for train, k in ((True, 0), (True, 1), (False, None)):
            xd, md = build_study(role_data, train, draw_rng(uid, k) if train else None)
            xr, mr = recon_study(volumes, train, draw_rng(uid, k) if train else None)
            equal &= bool(torch.equal(xd, xr) and torch.equal(md, mr))
            max_err = max(max_err, float((xd - xr).abs().max()))
            count += 1
        return {"ok": True, "equal": equal, "max_abs_error": max_err, "tensors": count,
                "seconds": time.perf_counter() - t0, "pid": os.getpid(), "rss_mib": rss_mib()}
    except Exception as exc:
        return fail(exc, t0)


def read_volume(arg):
    uid, k = arg
    torch.set_num_threads(1)
    t0, c0 = time.perf_counter(), time.process_time()
    try:
        with np.load(DIR / f"{name_of(uid)}.npz") as z:
            volumes = volumes_from_npz(z)
        x, m = recon_study(volumes, True, draw_rng(uid, k))
        assert tuple(x.shape) == (3, TRAIN_WINDOWS, 3, SIZE, SIZE)
        return {"ok": True, **meta(t0, c0)}
    except Exception as exc:
        return fail(exc, t0)


# ---- Design B: two fixed uint16 draws -------------------------------------------------------
def train_arrays(role_data, uid, k):
    x, m = build_study(role_data, train=True, rng=draw_rng(uid, k))
    assert tuple(x.shape) == (3, TRAIN_WINDOWS, 3, SIZE, SIZE) and tuple(m.shape) == (3, TRAIN_WINDOWS)
    return x.numpy(), m.numpy()


def quantize(x):
    return np.rint(np.clip(x.astype(np.float64) * Q, 0, Q)).astype(np.uint16)


def dequantize(q):
    return q.astype(np.float32) / Q


def build_uint16(uid):
    torch.set_num_threads(1)
    t0, c0 = time.perf_counter(), time.process_time()
    try:
        role_data, arrays = decode(uid), {}
        for k in range(DRAWS):
            x, m = train_arrays(role_data, uid, k)
            arrays[f"q{k}"], arrays[f"m{k}"] = quantize(x), m
        size = guard_and_save(DIR / f"{name_of(uid)}.npz", arrays)
        return {"ok": True, "bytes": size, **meta(t0, c0)}
    except Exception as exc:
        return fail(exc, t0)


def parity_uint16(uid):
    """Error of x=q/65535 vs the original float32, over valid (mask-true) windows only."""
    torch.set_num_threads(1)
    t0 = time.perf_counter()
    try:
        role_data = decode(uid)
        hist, total, count, max_err, masks_equal = np.zeros(BINS, np.int64), 0.0, 0, 0.0, True
        with np.load(DIR / f"{name_of(uid)}.npz") as z:
            for k in range(DRAWS):
                x, m = train_arrays(role_data, uid, k)
                masks_equal &= bool(np.array_equal(m, z[f"m{k}"]))
                err = np.abs(dequantize(z[f"q{k}"])[m].astype(np.float64) - x[m].astype(np.float64))
                hist += np.bincount(np.minimum((err / BIN_W).astype(np.int64).ravel(), BINS - 1),
                                    minlength=BINS)
                total, count, max_err = total + float(err.sum()), count + err.size, max(max_err, float(err.max()))
        return {"ok": True, "masks_equal": masks_equal, "hist": hist.tolist(), "sum": total,
                "count": count, "max": max_err, "seconds": time.perf_counter() - t0,
                "pid": os.getpid(), "rss_mib": rss_mib()}
    except Exception as exc:
        return fail(exc, t0)


def read_uint16(arg):
    uid, k = arg
    torch.set_num_threads(1)
    t0, c0 = time.perf_counter(), time.process_time()
    try:
        with np.load(DIR / f"{name_of(uid)}.npz") as z:
            x, m = torch.from_numpy(dequantize(z[f"q{k}"])), torch.from_numpy(z[f"m{k}"])
        assert tuple(x.shape) == (3, TRAIN_WINDOWS, 3, SIZE, SIZE) and tuple(m.shape) == (3, TRAIN_WINDOWS)
        return {"ok": True, **meta(t0, c0)}
    except Exception as exc:
        return fail(exc, t0)


# ---- harness -------------------------------------------------------------------------------
def run_pool(fn, items, phase, start):
    """Four forked workers; cancel queued work after the budget, then wait for running jobs."""
    budget = min(PHASE_CAP_S[phase], TOTAL_BUDGET_S - (time.perf_counter() - start))
    if budget <= 0:
        return [None] * len(items), 0.0
    t0 = time.perf_counter()
    pool = ProcessPoolExecutor(max_workers=WORKERS, mp_context=mp.get_context("fork"))
    futures = [pool.submit(fn, item) for item in items]
    wait(futures, timeout=budget)
    pool.shutdown(wait=True, cancel_futures=True)  # running jobs finish: cooperative, not a hard cap
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
            peak = {}
            for o in good:
                peak[o["pid"]] = max(peak.get(o["pid"], 0.0), o["rss_mib"])
            s["max_worker_peak_rss_mib"] = max(peak.values())
            # sum of per-worker peaks: upper bound (peaks need not coincide); fork may inherit parent RSS
            s["aggregate_worker_peak_rss_mib_upper_bound"] = sum(peak.values())
            s["worker_processes_seen"] = len(peak)
    return s


def size_stats(outcomes):
    sizes = [o["bytes"] for o in outcomes if o and o["ok"]]
    if not sizes:
        return None
    gib = lambda b: b / 2**30
    stats = {"median": statistics.median(sizes), "p95": pct95(sizes), "max": max(sizes),
             "mean": statistics.mean(sizes)}
    avail = FREE_GIB - RESERVE_GIB
    projected = {key: gib(v * TRAIN_STUDIES) for key, v in stats.items()}
    return {
        "bytes_per_study": stats,
        "projected_train_3457_gib": projected,
        "available_gib_after_reserve": avail,
        "fits_19_5gib_free_with_4gib_reserve": {key: v <= avail for key, v in projected.items()},
        "validation_not_cached": "validation stays on-demand (overlapping 48 windows are not cached)",
    }


def evict(path):
    """Best-effort drop of this directory's pages from the OS page cache (no root needed)."""
    try:
        os.sync()
        for p in path.glob("*.npz"):
            fd = os.open(p, os.O_RDONLY)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            os.close(fd)
        return True
    except Exception:
        return False


def run_design(tag, build, parity, read, selected, start):
    global DIR
    DIR = CACHE / tag
    DIR.mkdir(parents=True, exist_ok=True)
    result = {}
    try:
        outcomes, wall = run_pool(build, selected, "build", start)
        b = result["build"] = summarize(outcomes, wall)
        b["size"] = size_stats(outcomes)
        result["cache_disk_gib_after_build"] = dir_gib(DIR)
        built = [u for u, o in zip(selected, outcomes) if o and o["ok"]]
        out_p, wall_p = run_pool(parity, built[:N_PARITY], "parity", start)
        result["parity_raw"] = (summarize(out_p, wall_p), [o for o in out_p if o and o["ok"]])
        result["page_cache_eviction_attempted"] = evict(DIR)
        items = [(u, i % DRAWS) for i, u in enumerate(built)]
        out_r, wall_r = run_pool(read, items, "read", start)
        result["read_one_draw_per_study"] = summarize(out_r, wall_r)
        result["read_one_draw_per_study"]["note"] = (
            "read + window construction, one draw/study; page-cache eviction is best-effort, so "
            "numbers may still be warm-cache optimistic")
        est = {"provisional": True, "headroom": HEADROOM}
        if "wall_seconds_per_study" in b:
            est["full_train_build_hours_with_headroom"] = HEADROOM * TRAIN_STUDIES * b["wall_seconds_per_study"] / 3600
        r = result["read_one_draw_per_study"]
        if "wall_seconds_per_study" in r:
            est["eight_epoch_cpu_read_hours_with_headroom"] = (
                HEADROOM * EPOCHS * TRAIN_STUDIES * r["wall_seconds_per_study"] / 3600)
        est["gpu_note"] = "CPU-only; no GPU runtime estimate"
        result["estimates"] = est
    finally:
        shutil.rmtree(DIR, ignore_errors=True)
        result["cache_deleted"] = not DIR.exists()
    return result


def finish_volume(res):
    summary, good = res.pop("parity_raw")
    summary.update({
        "studies_compared": len(good), "tensors_compared": sum(o["tensors"] for o in good),
        "all_bitwise_equal": bool(good) and all(o["equal"] for o in good),
        "max_abs_error": max((o["max_abs_error"] for o in good), default=None),
        "meets_minimum_10_studies": len(good) >= N_PARITY,
        "note": ("2 seeded training draws (12 windows) + 48-window inference vs direct build_study; "
                 "if not bitwise equal, max_abs_error is the reported deviation"),
    })
    res["parity"] = summary


def finish_uint16(res):
    summary, good = res.pop("parity_raw")
    count = sum(o["count"] for o in good)
    hist = np.sum([o["hist"] for o in good], axis=0) if good else np.zeros(BINS, np.int64)
    p99 = None
    if count:
        p99 = float((np.searchsorted(np.cumsum(hist), 0.99 * count) + 1) * BIN_W)  # upper bin edge
    max_err = max((o["max"] for o in good), default=None)
    summary.update({
        "studies_compared": len(good), "elements_compared": count,
        "masks_exactly_equal": bool(good) and all(o["masks_equal"] for o in good),
        "max_abs_error": max_err,
        "mean_abs_error": sum(o["sum"] for o in good) / count if count else None,
        "p99_abs_error_upper_bin_edge": p99, "p99_resolution": BIN_W,
        "error_bound": ERR_BOUND,
        "max_error_within_bound": max_err is not None and max_err <= ERR_BOUND,
        "note": "valid (mask-true) windows only; LOSSY storage, never exact parity",
        "meets_minimum_10_studies": len(good) >= N_PARITY,
    })
    res["error"] = summary


start = time.perf_counter()
output = {
    "scope": ("CPU-only audit: 100 fixed nonexpert studies (same selection as loader/cache pilots), two "
              "cache designs built from the same DICOM, four workers; validation stays on-demand"),
    "design_float32_volume": ("Compressed resized float32 volumes + slice_ok (lossless); windows drawn at "
                              "read time, so dynamic window sampling is PRESERVED"),
    "design_uint16_two_draws": ("Two FIXED seeded draws quantized to uint16 (round(x*65535)); LOSSY and "
                                "NOT dynamic sampling: a policy/input change, never exact parity"),
    "time_limit_note": ("Cooperative phase-start budgets (~20 min target); running workers are not killed, "
                        "so this is not a hard wall cap"),
    "cpu_count": os.cpu_count(),
    "python": sys.version.split()[0],
    "working_disk_free_gib_start": shutil.disk_usage(WORKING).free / 2**30,
    "estimates_provisional": True,
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
    for key, tag, funcs, finish in (
            ("float32_volume", "volume", (build_volume, parity_volume, read_volume), finish_volume),
            ("uint16_two_draws", "uint16", (build_uint16, parity_uint16, read_uint16), finish_uint16)):
        try:
            res = run_design(tag, *funcs, selected, start)
            finish(res)
            output[key] = res
        except Exception as exc:
            output[key] = {"fatal_error_type": type(exc).__name__}
except Exception as exc:
    output["fatal_error_type"] = type(exc).__name__  # type only: messages may carry paths/UIDs
finally:
    shutil.rmtree(CACHE, ignore_errors=True)
    output["cache_deleted"] = not CACHE.exists()
    output["elapsed_seconds"] = time.perf_counter() - start
    output["parent_peak_rss_mib"] = rss_mib()
    OUT.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))
if ("fatal_error_type" in output or not output["cache_deleted"]
        or any("fatal_error_type" in output.get(k, {}) for k in ("float32_volume", "uint16_two_draws"))):
    sys.exit(1)
