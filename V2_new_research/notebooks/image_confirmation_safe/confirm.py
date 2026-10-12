"""Private bounded CPU image comparison using only Kaggle-mounted inputs."""
import csv
import hashlib
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pydicom
from PIL import Image
from skimage.metrics import structural_similarity

MAX_SECONDS = 1500
MAX_PIXEL_DECODES = 3800
QUANTILES = (0.15, 0.325, 0.5, 0.675, 0.85)
start = time.perf_counter()
DATA = next((root for root in (Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"), Path("/kaggle/input/rsna-knee-abnormality-detection")) if (root / "train_series.csv").exists()), None)
if DATA is None:
    raise FileNotFoundError("Competition mount missing")


def read_rows(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


# The preceding private Kaggle screen is attached as a kernel source. This
# script uploads no study UID, series UID, report, label, or image payload.
sources = sorted(Path("/kaggle/input").rglob("near_image_pairs.csv"))
if len(sources) != 1:
    raise FileNotFoundError("Expected one attached private near_image_pairs.csv")
near_source = sources[0]
candidate_rows = read_rows(near_source)
if len(candidate_rows) != 272:
    raise ValueError("Unexpected candidate count from preceding screen")
series_rows = read_rows(DATA / "train_series.csv")
first_series = {}
for row in series_rows:
    first_series.setdefault(row["StudyInstanceUID"], row)
train_rows = read_rows(DATA / "train.csv")
targets = [key for key in train_rows[0] if key not in {"StudyInstanceUID", "Report"}]
# Only completeness is inspected to enforce the 58-study holdout; no expert
# target values are used or exported.
expert_uids = {row["StudyInstanceUID"] for row in train_rows if all(row[key] in {"0", "1"} for key in targets)}
if len(expert_uids) != 58:
    raise ValueError("Unexpected protected expert cohort size")
candidate_pairs = []
candidate_set = set()
for row in candidate_rows:
    a, b = row["study_uid_a"], row["study_uid_b"]
    pair = tuple(sorted((a, b)))
    candidate_set.add(pair)
    candidate_pairs.append([pair[0], pair[1], "candidate", pair[0] in expert_uids or pair[1] in expert_uids, False])
if len(candidate_set) != 272 or sum(pair[3] for pair in candidate_pairs) != 1:
    raise ValueError("Unexpected candidate or expert-linked pair")
eligible = sorted(uid for uid in first_series if uid not in expert_uids)
rng = random.Random(20261009)
controls = set()
for attempt in range(100000):
    if len(controls) == 100:
        break
    a, b = sorted(rng.sample(eligible, 2))
    if (a, b) not in candidate_set and all(first_series[a][key] == first_series[b][key]
                                            for key in ("Anatomical_Plane", "Fluid_Sensitive", "Fat_Suppression")):
        controls.add((a, b))
if len(controls) != 100:
    raise ValueError("Could not sample 100 same-sequence controls")
candidate_pairs.sort(key=lambda pair: (not pair[3], pair[0], pair[1]))
candidate_pairs.extend([a, b, "control", False, False] for a, b in sorted(controls))
INPUT = {
    "pairs": candidate_pairs,
    "series": first_series,
    "source_sha256": {
        "near_image_pairs": hashlib.sha256(near_source.read_bytes()).hexdigest(),
        "train_series": hashlib.sha256((DATA / "train_series.csv").read_bytes()).hexdigest(),
    },
}
QUARANTINED_UIDS = expert_uids
headers = {}
arrays = {}
decoded = 0
header_reads = 0
failures = Counter()


def bounded():
    if time.perf_counter() - start > MAX_SECONDS or decoded >= MAX_PIXEL_DECODES:
        raise TimeoutError("Prespecified audit bound reached")


def geometry(uid):
    global header_reads
    if uid in headers:
        return headers[uid]
    series = INPUT["series"][uid]
    files = sorted((DATA / "train_series" / uid / series["SeriesInstanceUID"]).glob("*.dcm"))
    if len(files) < 5 or len(files) > 500:
        raise ValueError("Unexpected series size")
    records = []
    for path in files:
        bounded()
        ds = pydicom.dcmread(path, stop_before_pixels=True)
        header_reads += 1
        if str(ds.StudyInstanceUID) != uid or str(ds.SeriesInstanceUID) != series["SeriesInstanceUID"]:
            raise ValueError("Header/path UID mismatch")
        orientation = np.asarray(ds.ImageOrientationPatient, dtype=float)
        normal = np.cross(orientation[:3], orientation[3:])
        normal /= np.linalg.norm(normal)
        position = float(np.dot(np.asarray(ds.ImagePositionPatient, dtype=float), normal))
        shape = (int(ds.Rows), int(ds.Columns))
        spacing = np.asarray(ds.PixelSpacing, dtype=float)
        if not np.isfinite(orientation).all() or not np.isfinite(position) or not np.isfinite(spacing).all() or min(spacing) <= 0:
            raise ValueError("Invalid geometry")
        records.append((position, path, orientation, shape, spacing))
    records.sort(key=lambda item: item[0])
    reference = records[0]
    if any(rec[3] != reference[3] or not np.allclose(rec[4], reference[4], rtol=0.001) or not np.allclose(rec[2], reference[2], atol=0.001) for rec in records):
        raise ValueError("Within-series geometry mismatch")
    if min(np.diff([rec[0] for rec in records])) <= 1e-4:
        raise ValueError("Duplicate projected positions")
    headers[uid] = records
    return records


def normalized_slices(uid, records):
    global decoded
    if uid in QUARANTINED_UIDS:
        raise ValueError("Expert-linked UID pixel quarantine")
    if uid in arrays:
        return arrays[uid]
    result = []
    indices = [round(q * (len(records) - 1)) for q in QUANTILES]
    if len(set(indices)) != 5:
        raise ValueError("Non-distinct quantile slices")
    for index in indices:
        bounded()
        ds = pydicom.dcmread(records[index][1])
        arr = np.asarray(ds.pixel_array, dtype=np.float32)
        decoded += 1
        if arr.ndim != 2 or not np.isfinite(arr).all():
            raise ValueError("Invalid pixels")
        lo, hi = np.percentile(arr, [1, 99])
        if hi <= lo:
            raise ValueError("Flat image")
        arr = np.clip((arr - lo) / (hi - lo), 0, 1)
        # Physical aspect-preserving resample and fixed central crop; metadata
        # comparability is checked first. No learned registration or tuning.
        fov = np.asarray(arr.shape) * records[index][4]
        scale = 320 / min(fov)
        if max(fov) * scale > 2048:
            raise ValueError("Unexpected physical aspect ratio")
        image = Image.fromarray(arr).resize((max(320, round(fov[1] * scale)), max(320, round(fov[0] * scale))), Image.Resampling.BILINEAR)
        width, height = image.size
        arr = np.asarray(image.crop(((width - 320)//2, (height - 320)//2, (width - 320)//2 + 320, (height - 320)//2 + 320)), dtype=np.float32)
        result.append(arr)
    arrays[uid] = result
    return result


def compare(a, b, expert):
    left, right = geometry(a), geometry(b)
    if expert:
        return "expert_quarantined_headers_only", []
    sa, sb = INPUT["series"][a], INPUT["series"][b]
    if any(sa[key] != sb[key] for key in ("Anatomical_Plane", "Fluid_Sensitive", "Fat_Suppression")):
        return "unresolved_sequence_metadata", []
    la, lb = left[0], right[0]
    if min(np.dot(la[2][:3], lb[2][:3]), np.dot(la[2][3:], lb[2][3:])) < 0.99:
        return "unresolved_orientation", []
    fov_a, fov_b = np.asarray(la[3]) * la[4], np.asarray(lb[3]) * lb[4]
    if np.max(np.abs(fov_a/fov_b - 1)) > 0.10:
        return "unresolved_field_of_view", []
    span_a, span_b = left[-1][0] - left[0][0], right[-1][0] - right[0][0]
    if max(span_a, span_b) / min(span_a, span_b) > 1.20:
        return "unresolved_coverage", []
    metrics = []
    for x, y in zip(normalized_slices(a, left), normalized_slices(b, right)):
        ncc = float(np.corrcoef(x.ravel(), y.ravel())[0, 1])
        rmse = float(np.sqrt(np.mean((x-y)**2)))
        ssim = float(structural_similarity(x, y, data_range=1.0))
        if not np.isfinite((ncc, rmse, ssim)).all():
            raise ValueError("Nonfinite similarity")
        metrics.append((ncc, rmse, ssim))
    strong = sum(ncc >= 0.99 and rmse <= 0.035 and ssim >= 0.97 for ncc, rmse, ssim in metrics)
    distinct = sum(ncc < 0.90 and ssim < 0.75 for ncc, rmse, ssim in metrics)
    return ("high_similarity_requires_manual_confirmation" if strong >= 3 else ("sampled_slices_dissimilar" if distinct >= 3 else "unresolved_similarity")), metrics


decisions = defaultdict(Counter)
values = defaultdict(list)
pair_results = []
processed = 0
for a, b, kind, expert, cross_fold in INPUT["pairs"]:
    strata = [kind] + (["candidate_cross_fold"] if cross_fold else [])
    try:
        decision, metrics = compare(a, b, expert)
    except TimeoutError:
        failures["time_or_decode_bound"] += 1
        break
    except Exception as exc:
        failures[type(exc).__name__] += 1
        decision, metrics = "unresolved_failure", []
    for stratum in strata:
        decisions[stratum][decision] += 1
        values[stratum].extend(metrics)
    if kind == "candidate":
        pair_results.append((hashlib.sha256((a + "|" + b).encode()).hexdigest(), decision))
    processed += 1

summary = {
    "status": "bounded_candidate_confirmation_not_split_clearance",
    "source_sha256": INPUT["source_sha256"],
    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "input_selection_sha256": hashlib.sha256(json.dumps(INPUT["pairs"], sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
    "prespecified_pairs": len(INPUT["pairs"]), "processed_pairs": processed,
    "unprocessed_pairs": len(INPUT["pairs"]) - processed,
    "decisions": {key: dict(value) for key, value in sorted(decisions.items())},
    "metric_percentiles": {key: {name: np.percentile(np.asarray(data)[:, index], [0, 25, 50, 75, 100]).tolist() for index, name in enumerate(("ncc", "rmse", "ssim"))} for key, data in sorted(values.items()) if data},
    "pixel_decodes": decoded, "header_reads": header_reads, "failures": dict(failures),
    "expert_pixel_decodes": 0, "expert_label_values_used": False,
    "bounds": {"seconds": MAX_SECONDS, "pixel_decodes": MAX_PIXEL_DECODES, "slices_per_series": 5, "image_size": 320},
    "comparison_policy": {"quantiles": QUANTILES, "metadata": "same plane/fluid-sensitive/fat-suppression; orientation dot >=0.99; FOV delta <=10%; coverage ratio <=1.20", "high_similarity": "at least 3/5 slices with NCC>=0.99, RMSE<=0.035, SSIM>=0.97", "sampled_dissimilarity": "at least 3/5 slices with NCC<0.90 and SSIM<0.75"},
    "elapsed_seconds": round(time.perf_counter() - start, 2),
    "limitations": ["High similarity is suspicion, not patient identity; sampled dissimilarity does not rule out other duplicated slices or series.", "Quantile-aligned slices are not anatomical registration; acquisitions can differ in coverage or positioning.", "Controls were fixed before metrics and are diagnostic only; thresholds are not calibrated or tuned on them.", "Expert-linked pair remains quarantined with headers only; split is not modified."]}
Path("/kaggle/working/image_confirmation_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
with Path("/kaggle/working/private_candidate_decisions.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.writer(handle)
    writer.writerow(("pair_sha256", "decision"))
    writer.writerows(pair_results)
print(json.dumps(summary, indent=2))
