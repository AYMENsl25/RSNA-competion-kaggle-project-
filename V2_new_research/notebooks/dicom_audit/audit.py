"""CPU-only, read-only audit of official RSNA Knee DICOMs; no model fitting."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

import pydicom


DATA = Path("/kaggle/input/competitions/rsna-knee-abnormality-detection")
OUT = Path("/kaggle/working")
TARGETS = (
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
    "Contusion", "Fracture",
)


def rows(name: str) -> list[dict[str, str]]:
    with (DATA / name).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def key(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def numbers(values: object, expected: int) -> tuple[float, ...]:
    converted = tuple(float(value) for value in values)
    if len(converted) != expected or not all(math.isfinite(value) for value in converted):
        raise ValueError(f"Expected {expected} finite values")
    return converted


def geometry(ds: pydicom.dataset.FileDataset) -> tuple[float, tuple[float, float], tuple[int, int]]:
    o = numbers(ds.ImageOrientationPatient, 6)
    p = numbers(ds.ImagePositionPatient, 3)
    s = numbers(ds.PixelSpacing, 2)
    if min(s) <= 0:
        raise ValueError("Nonpositive pixel spacing")
    a, b = o[:3], o[3:]
    dot = sum(x * y for x, y in zip(a, b))
    if abs(sum(x * x for x in a) - 1) > 2e-3 or abs(sum(x * x for x in b) - 1) > 2e-3 or abs(dot) > 1e-3:
        raise ValueError("Invalid orientation cosines")
    n = (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])
    return sum(x * y for x, y in zip(p, n)), (round(s[0], 5), round(s[1], 5)), (int(ds.Rows), int(ds.Columns))


start = time.perf_counter()
mounts = sorted(path.name for path in Path("/kaggle/input").iterdir())
print("Mounted input directories:", mounts)
if not (DATA / "train.csv").exists():
    fallback = Path("/kaggle/input/rsna-knee-abnormality-detection")
    if not (fallback / "train.csv").exists():
        raise FileNotFoundError(f"Official train.csv absent at both expected mount paths; input roots: {mounts}")
    DATA = fallback
print("Using mounted competition folder:", DATA)
train = rows("train.csv")
series = rows("train_series.csv")
expert = {row["StudyInstanceUID"] for row in train if all(row[t] in {"0", "1"} for t in TARGETS)}
report_groups = defaultdict(set)
for row in train:
    report_groups[key(" ".join(row["Report"].casefold().split()))].add(row["StudyInstanceUID"])
excluded = set(expert)
for members in report_groups.values():
    if members & expert:
        excluded |= members

# One header per study. This is an integrity scan, not training or model selection.
first_series = {}
for row in series:
    first_series.setdefault(row["StudyInstanceUID"], row["SeriesInstanceUID"])
subject_groups = defaultdict(set)
header_failures = Counter()
transfer_syntax = Counter()
patient_missing = 0
patient_manifest = []
for uid in sorted(first_series):
    folder = DATA / "train_series" / uid / first_series[uid]
    paths = sorted(folder.glob("*.dcm"))
    if not paths:
        header_failures["missing_first_series_file"] += 1
        continue
    try:
        ds = pydicom.dcmread(paths[0], stop_before_pixels=True)
        if str(ds.StudyInstanceUID) != uid or str(ds.SeriesInstanceUID) != first_series[uid]:
            raise ValueError("UID/path mismatch")
        pid = str(getattr(ds, "PatientID", "")).strip()
        if not pid:
            patient_missing += 1
        else:
            subject_groups[key(pid)].add(uid)
        transfer_syntax[str(ds.file_meta.TransferSyntaxUID)] += 1
        patient_manifest.append((uid, key(pid) if pid else ""))
    except Exception as exc:
        header_failures[type(exc).__name__] += 1

# Broader transfer-syntax and subject-consistency census: one header per series.
patient_by_study = dict(patient_manifest)
all_series_header_failures = Counter()
all_series_transfer_syntax = Counter()
patient_id_conflicts_across_series = 0
syntax_representative = {}
for row in series:
    uid, sid = row["StudyInstanceUID"], row["SeriesInstanceUID"]
    paths = sorted((DATA / "train_series" / uid / sid).glob("*.dcm"))
    if not paths:
        all_series_header_failures["missing_series_folder_or_file"] += 1
        continue
    try:
        ds = pydicom.dcmread(paths[0], stop_before_pixels=True)
        if str(ds.StudyInstanceUID) != uid or str(ds.SeriesInstanceUID) != sid:
            raise ValueError("UID/path mismatch")
        pid = str(getattr(ds, "PatientID", "")).strip()
        if patient_by_study.get(uid, "") != (key(pid) if pid else ""):
            patient_id_conflicts_across_series += 1
        ts = str(ds.file_meta.TransferSyntaxUID)
        all_series_transfer_syntax[ts] += 1
        if uid not in excluded:
            syntax_representative.setdefault(ts, row)
    except Exception as exc:
        all_series_header_failures[type(exc).__name__] += 1

expert_subject_keys = {key_ for key_, members in subject_groups.items() if members & expert}
subject_linked_nonexpert = set().union(*(subject_groups[k] - expert for k in expert_subject_keys)) if expert_subject_keys else set()
excluded |= subject_linked_nonexpert

# Deterministic study/series sample across the six plane-by-fluid strata.
by_stratum = defaultdict(list)
for row in series:
    if row["StudyInstanceUID"] not in excluded:
        stratum = f'{row["Anatomical_Plane"]}|fluid={row["Fluid_Sensitive"]}'
        by_stratum[stratum].append(row)
selected = []
for stratum, entries in sorted(by_stratum.items()):
    entries.sort(key=lambda row: key(row["StudyInstanceUID"] + "/" + row["SeriesInstanceUID"]))
    selected.extend(entries[:10])
selected_keys = {(row["StudyInstanceUID"], row["SeriesInstanceUID"]) for row in selected}
for row in syntax_representative.values():
    pair = (row["StudyInstanceUID"], row["SeriesInstanceUID"])
    if pair not in selected_keys and row["StudyInstanceUID"] not in excluded:
        selected.append(row)
        selected_keys.add(pair)

series_failures = Counter()
pixel_failures = Counter()
sampled_transfer = Counter()
sampled_slices = 0
decoded_slices = 0
geometry_mismatch = 0
filename_order_mismatch = 0
duplicate_positions = 0
dimensions = Counter()
spacing = Counter()
all_series_slices = []
for row in selected:
    uid, sid = row["StudyInstanceUID"], row["SeriesInstanceUID"]
    folder = DATA / "train_series" / uid / sid
    paths = sorted(folder.glob("*.dcm"))
    if not paths:
        series_failures["empty_folder"] += 1
        continue
    all_series_slices.append(len(paths))
    projected = []
    known_shapes = set()
    known_spacing = set()
    for path in paths:
        sampled_slices += 1
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True)
            if str(ds.StudyInstanceUID) != uid or str(ds.SeriesInstanceUID) != sid or str(ds.SOPInstanceUID) != path.stem:
                raise ValueError("UID/path mismatch")
            z, ps, shape = geometry(ds)
            projected.append((path.name, z))
            known_shapes.add(shape)
            known_spacing.add(ps)
            dimensions[str(shape)] += 1
            spacing[str(ps)] += 1
            sampled_transfer[str(ds.file_meta.TransferSyntaxUID)] += 1
        except Exception as exc:
            series_failures[type(exc).__name__] += 1
    if len(known_shapes) > 1 or len(known_spacing) > 1:
        geometry_mismatch += 1
    spatial = sorted(projected, key=lambda item: item[1])
    if [item[0] for item in projected] != [item[0] for item in spatial]:
        filename_order_mismatch += 1
    duplicate_positions += sum(b[1] - a[1] <= 0.01 for a, b in zip(spatial, spatial[1:]))
    # Decode a reproducible first, middle and last slice for each series.
    for index in sorted({0, len(paths) // 2, len(paths) - 1}):
        try:
            ds = pydicom.dcmread(paths[index])
            pixels = ds.pixel_array
            if pixels.ndim != 2 or pixels.shape != (int(ds.Rows), int(ds.Columns)):
                raise ValueError("Pixel shape mismatch")
            decoded_slices += 1
        except Exception as exc:
            pixel_failures[type(exc).__name__] += 1

with (OUT / "patient_groups.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.writer(handle)
    writer.writerow(["StudyInstanceUID", "patient_id_sha256", "expert_or_linked_excluded"])
    for uid, patient_hash in patient_manifest:
        writer.writerow([uid, patient_hash, int(uid in excluded)])

summary = {
    "scope": "CPU header census by first series per study plus deterministic 10-series-per-stratum geometry sample; no training",
    "train_studies": len(train),
    "expert_studies": len(expert),
    "expert_or_exact_report_excluded": len({uid for members in report_groups.values() if members & expert for uid in members}),
    "nonexpert_patient_linked_to_expert": len(subject_linked_nonexpert),
    "excluded_union": len(excluded),
    "patient_id_missing_in_first_series": patient_missing,
    "patient_groups_with_multiple_studies": sum(len(members) > 1 for members in subject_groups.values()),
    "first_series_header_failures": dict(header_failures),
    "first_series_transfer_syntax": dict(transfer_syntax),
    "all_series_header_count": len(series),
    "all_series_header_failures": dict(all_series_header_failures),
    "all_series_transfer_syntax": dict(all_series_transfer_syntax),
    "patient_id_conflicts_across_series": patient_id_conflicts_across_series,
    "sampled_series_by_stratum": dict(Counter(f'{r["Anatomical_Plane"]}|fluid={r["Fluid_Sensitive"]}' for r in selected)),
    "sampled_series": len(selected),
    "sampled_slices": sampled_slices,
    "series_header_failures": dict(series_failures),
    "decoded_slices": decoded_slices,
    "pixel_decode_failures": dict(pixel_failures),
    "sampled_transfer_syntax": dict(sampled_transfer),
    "series_with_mixed_shape_or_spacing": geometry_mismatch,
    "series_filename_order_differs_from_geometry": filename_order_mismatch,
    "duplicate_projected_position_pairs": duplicate_positions,
    "series_slice_count_min_median_max": [min(all_series_slices), statistics.median(all_series_slices), max(all_series_slices)] if all_series_slices else [],
    "sampled_shape_counts": dict(dimensions),
    "sampled_pixel_spacing_counts": dict(spacing),
    "elapsed_seconds": round(time.perf_counter() - start, 2),
}
(OUT / "audit_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps({k: summary[k] for k in ("train_studies", "sampled_series", "sampled_slices", "decoded_slices", "elapsed_seconds")}))
