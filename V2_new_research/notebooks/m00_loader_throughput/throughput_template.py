"""CPU-only parallel DICOM loader probe; no labels, fitting or submission."""

import csv
import hashlib
import json
import multiprocessing as mp
import os
import shutil
import statistics
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pydicom
import torch

# __M00_DATA_SOURCE__

ROOTS = (
    Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"),
    Path("/kaggle/input/rsna-knee-abnormality-detection"),
)
DATA = next((p for p in ROOTS if (p / "train.csv").exists()), None)
if DATA is None:
    raise FileNotFoundError("Official competition input missing")


def read_rows(name):
    with (DATA / name).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def prepare(uid):
    torch.set_num_threads(1)
    start = time.perf_counter()
    role_ids = select_series(SERIES_BY_STUDY[uid])
    role_data = {}
    try:
        for role, sid in role_ids.items():
            if sid is None:
                continue
            paths = sorted((DATA / "train_series" / uid / sid).glob("*.dcm"))
            datasets = [pydicom.dcmread(path) for path in paths]
            role_data[role] = load_series(datasets)
        rng = np.random.default_rng(int(hashlib.sha256(uid.encode()).hexdigest()[:16], 16))
        train_x, train_mask = build_study(role_data, train=True, rng=rng)
        infer_x, infer_mask = build_study(role_data, train=False)
        assert train_x.shape[:2] == (3, 4) and infer_x.shape[:2] == (3, 16)
        digest = hashlib.sha256(infer_x.numpy().tobytes() + infer_mask.numpy().tobytes()).hexdigest()
        return {"ok": True, "digest": digest, "seconds": time.perf_counter() - start,
                "train_valid_windows": int(train_mask.sum()), "infer_valid_windows": int(infer_mask.sum())}
    except Exception as exc:
        return {"ok": False, "error": type(exc).__name__, "seconds": time.perf_counter() - start}


def run_mode(workers, selected):
    t0 = time.perf_counter()
    if workers == 1:
        outcomes = [prepare(uid) for uid in selected]
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("fork")) as pool:
            outcomes = list(pool.map(prepare, selected))
    wall = time.perf_counter() - t0
    return {
        "workers": workers,
        "wall_seconds": wall,
        "successful_studies": sum(o["ok"] for o in outcomes),
        "failed_studies": sum(not o["ok"] for o in outcomes),
        "studies_per_hour": len(outcomes) * 3600 / wall,
        "median_study_seconds": statistics.median(o["seconds"] for o in outcomes),
        "p95_study_seconds": sorted(o["seconds"] for o in outcomes)[int(0.95 * (len(outcomes) - 1))],
        "minimum_train_valid_windows": min((o.get("train_valid_windows", 0) for o in outcomes), default=0),
    }, outcomes


start = time.perf_counter()
train, series = read_rows("train.csv"), read_rows("train_series.csv")
targets = [key for key in train[0] if key not in {"StudyInstanceUID", "Report"}]
expert = {r["StudyInstanceUID"] for r in train if all(r[t] in {"0", "1"} for t in targets)}
expert_reports = {" ".join(r["Report"].casefold().split()) for r in train if r["StudyInstanceUID"] in expert}
eligible = {r["StudyInstanceUID"] for r in train if r["StudyInstanceUID"] not in expert and " ".join(r["Report"].casefold().split()) not in expert_reports}
SERIES_BY_STUDY = {}
for row in series:
    SERIES_BY_STUDY.setdefault(row["StudyInstanceUID"], []).append(row)
selected = sorted(eligible, key=lambda uid: hashlib.sha256(uid.encode()).hexdigest())[:100]
mode_summaries, digests_by_mode = [], []
for workers in (4,):
    summary, outcomes = run_mode(workers, selected)
    mode_summaries.append(summary)
    digests_by_mode.append([o.get("digest") for o in outcomes])
    print(summary, flush=True)
    if time.perf_counter() - start > 900:
        break

output = {
    "scope": "100 fixed nonexpert studies, all selected sagittal/coronal/axial series, full decode and both 12-window train/48-window inference tensors; four CPU workers",
    "cpu_count": os.cpu_count(),
    "working_disk_free_gib": shutil.disk_usage("/kaggle/working").free / 2**30,
    "mode_summaries": mode_summaries,
    "window_tensor_parity_across_modes": None,
    "elapsed_seconds": time.perf_counter() - start,
    "engineering_only": True,
}
Path("/kaggle/working/loader_throughput_summary.json").write_text(json.dumps(output, indent=2) + "\n")
print(json.dumps(output, indent=2))
