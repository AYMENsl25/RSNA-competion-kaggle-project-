"""Bounded exact pixel-duplicate screen; private output, no model fitting."""

from __future__ import annotations

import csv
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pydicom
from PIL import Image


ROOTS = (
    Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"),
    Path("/kaggle/input/rsna-knee-abnormality-detection"),
)
DATA = next((root for root in ROOTS if (root / "train_series.csv").exists()), None)
if DATA is None:
    raise FileNotFoundError("Official competition input is not mounted")

start = time.perf_counter()
with (DATA / "train_series.csv").open(newline="", encoding="utf-8-sig") as handle:
    series = list(csv.DictReader(handle))
with (DATA / "train.csv").open(newline="", encoding="utf-8-sig") as handle:
    train = list(csv.DictReader(handle))
targets = [column for column in train[0] if column not in {"StudyInstanceUID", "Report"}]
expert = {row["StudyInstanceUID"] for row in train if all(row[t] in {"0", "1"} for t in targets)}

# Select one official series per study deterministically. This screen samples
# three SOP files per study; it does not claim to rule out all image linkage.
first = {}
for row in series:
    first.setdefault(row["StudyInstanceUID"], row["SeriesInstanceUID"])
pixel_owners = defaultdict(set)
dicom_uid_owners = defaultdict(set)
perceptual = []
failures = Counter()
decoded = 0


def dhash64(arr):
    """Center-cropped difference hash for candidate screening, never proof of identity."""
    h, w = arr.shape
    crop = np.asarray(arr[h // 10:h - h // 10, w // 10:w - w // 10], dtype=np.float32)
    if not np.isfinite(crop).all() or crop.max() <= crop.min():
        return None
    lo, hi = np.percentile(crop[::4, ::4], [1, 99])
    if hi <= lo:
        return None
    crop = np.clip((crop - lo) / (hi - lo), 0, 1)
    small = np.asarray(Image.fromarray(crop).resize((9, 8), Image.Resampling.BILINEAR))
    bits = (small[:, 1:] > small[:, :-1]).reshape(-1)
    result = 0
    for bit in bits:
        result = (result << 1) | int(bit)
    return result


for study_uid, series_uid in sorted(first.items()):
    paths = sorted((DATA / "train_series" / study_uid / series_uid).glob("*.dcm"))
    if not paths:
        failures["missing_series_or_dicom"] += 1
        continue
    for sample_index, index in enumerate(sorted({0, len(paths) // 2, len(paths) - 1})):
        try:
            ds = pydicom.dcmread(paths[index])
            if str(ds.StudyInstanceUID) != study_uid or str(ds.SeriesInstanceUID) != series_uid:
                raise ValueError("UID/path mismatch")
            arr = np.asarray(ds.pixel_array)
            if arr.ndim != 2 or arr.size == 0:
                raise ValueError("Unexpected pixel array")
            digest = hashlib.sha256(
                str((arr.shape, arr.dtype.str)).encode("ascii") +
                np.ascontiguousarray(arr).tobytes()
            ).hexdigest()
            pixel_owners[digest].add(study_uid)
            dicom_uid_owners[str(ds.SOPInstanceUID)].add(study_uid)
            perceptual.append((study_uid, sample_index, dhash64(arr)))
            decoded += 1
        except Exception as exc:
            failures[type(exc).__name__] += 1

pixel_pairs = set()
for owners in pixel_owners.values():
    if len(owners) > 1:
        ordered = sorted(owners)
        for i, left in enumerate(ordered):
            for right in ordered[i + 1:]:
                pixel_pairs.add((left, right))
uid_pairs = set()
for owners in dicom_uid_owners.values():
    if len(owners) > 1:
        ordered = sorted(owners)
        for i, left in enumerate(ordered):
            for right in ordered[i + 1:]:
                uid_pairs.add((left, right))
all_pairs = pixel_pairs | uid_pairs

# Locality-sensitive buckets generate candidate hashes. Requiring two distinct
# sampled slices on each side limits obvious generic-anatomy collisions.
near_images = defaultdict(set)
postings = defaultdict(list)
for image_index, (_, _, value) in enumerate(perceptual):
    if value is None:
        continue
    for chunk in range(8):
        postings[(chunk, (value >> (8 * chunk)) & 255)].append(image_index)
seen_close_image_pairs = set()
skipped_large_buckets = 0
for members in postings.values():
    if len(members) > 500:
        skipped_large_buckets += 1
        continue
    for i, left_idx in enumerate(members):
        uid_a, sample_a, hash_a = perceptual[left_idx]
        for right_idx in members[i + 1:]:
            uid_b, sample_b, hash_b = perceptual[right_idx]
            if uid_a == uid_b or (hash_a ^ hash_b).bit_count() > 4:
                continue
            image_pair = (min(left_idx, right_idx), max(left_idx, right_idx))
            if image_pair in seen_close_image_pairs:
                continue
            seen_close_image_pairs.add(image_pair)
            if uid_a < uid_b:
                near_images[(uid_a, uid_b)].add((sample_a, sample_b))
            else:
                near_images[(uid_b, uid_a)].add((sample_b, sample_a))
strong_near_pairs = {
    pair for pair, matched in near_images.items()
    if len({a for a, _ in matched}) >= 2 and len({b for _, b in matched}) >= 2
}
with Path("/kaggle/working/duplicate_pairs.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.writer(handle)
    writer.writerow(("study_uid_a", "study_uid_b", "pixel_match", "sop_uid_match", "touches_expert"))
    for left, right in sorted(all_pairs):
        writer.writerow((left, right, int((left, right) in pixel_pairs),
                         int((left, right) in uid_pairs), int(left in expert or right in expert)))
with Path("/kaggle/working/near_image_pairs.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.writer(handle)
    writer.writerow(("study_uid_a", "study_uid_b", "matching_sampled_image_pairs", "touches_expert"))
    for left, right in sorted(strong_near_pairs):
        writer.writerow((left, right, len(near_images[(left, right)]), int(left in expert or right in expert)))
summary = {
    "scope": "Exact decoded-pixel and SOP UID screen on first/middle/last filename-sorted files in one deterministic series per study; not exhaustive near-duplicate detection",
    "studies_expected": len(first),
    "expert_studies": len(expert),
    "decoded_images": decoded,
    "failures": dict(failures),
    "cross_study_exact_pixel_pairs": len(pixel_pairs),
    "cross_study_sop_uid_pairs": len(uid_pairs),
    "cross_study_pairs_touching_expert": sum(left in expert or right in expert for left, right in all_pairs),
    "perceptual_hashes": sum(value is not None for _, _, value in perceptual),
    "near_image_candidate_study_pairs_one_slice": len(near_images),
    "near_image_candidate_study_pairs_two_distinct_slices": len(strong_near_pairs),
    "near_image_candidates_touching_expert": sum(left in expert or right in expert for left, right in strong_near_pairs),
    "perceptual_large_buckets_skipped": skipped_large_buckets,
    "perceptual_limitations": "64-bit difference hash <=4 on center crop is a candidate signal; similar anatomy may be false matches, different acquisition may evade detection",
    "elapsed_seconds": round(time.perf_counter() - start, 2),
}
Path("/kaggle/working/duplicate_screen_summary.json").write_text(
    json.dumps(summary, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(summary, indent=2))
