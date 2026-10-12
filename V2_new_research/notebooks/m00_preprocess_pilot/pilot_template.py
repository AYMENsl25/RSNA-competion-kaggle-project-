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

# __M00_DATA_SOURCE__

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
