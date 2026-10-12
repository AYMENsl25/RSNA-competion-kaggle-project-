"""Audit official RSNA CSV metadata without exporting reports or study IDs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
    "Contusion", "Fracture",
]
UID = "StudyInstanceUID"
FILES = ("train.csv", "train_series.csv", "test.csv", "test_series.csv", "sample_submission.csv")


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit(directory: Path) -> dict:
    tables = {name: read_csv(directory / name) for name in FILES}
    train_cols, train = tables["train.csv"]
    series_cols, series = tables["train_series.csv"]
    test_cols, test = tables["test.csv"]
    test_series_cols, test_series = tables["test_series.csv"]
    sample_cols, sample = tables["sample_submission.csv"]

    if train_cols != [UID, "Report", *TARGETS]:
        raise ValueError(f"Unexpected train schema: {train_cols}")
    if series_cols != [UID, "SeriesInstanceUID", "Fluid_Sensitive", "Fat_Suppression", "Anatomical_Plane"]:
        raise ValueError(f"Unexpected train series schema: {series_cols}")
    if test_cols != [UID] or test_series_cols != series_cols or sample_cols != [UID, *TARGETS]:
        raise ValueError("Test/sample schema differs from the competition contract")

    train_ids = [row[UID] for row in train]
    test_ids = [row[UID] for row in test]
    sample_ids = [row[UID] for row in sample]
    train_id_set = set(train_ids)
    test_id_set = set(test_ids)
    if len(train_id_set) != len(train_ids) or len(test_id_set) != len(test_ids):
        raise ValueError("Duplicate study ID in train or test")
    if train_id_set & test_id_set or sample_ids != test_ids:
        raise ValueError("Train/test overlap or sample submission order mismatch")

    study_series = defaultdict(list)
    for row in series:
        if row[UID] not in train_id_set:
            raise ValueError("Train series contains unknown study")
        study_series[row[UID]].append(row)
    if set(study_series) != train_id_set:
        raise ValueError("Train study missing from series metadata")
    if {row[UID] for row in test_series} != test_id_set:
        raise ValueError("Test series and test study IDs differ")

    label_counts = {
        target: dict(Counter(row[target] if row[target] else "missing" for row in train))
        for target in TARGETS
    }
    gold = sum(all(row[target] in {"0", "1"} for target in TARGETS) for row in train)
    partial = sum(any(row[target] in {"0", "1"} for target in TARGETS)
                  and not all(row[target] in {"0", "1"} for target in TARGETS)
                  for row in train)
    report_norm = [" ".join(row["Report"].lower().split()) for row in train]
    series_counts = [len(rows) for rows in study_series.values()]

    return {
        "source": "Official competition CSVs downloaded through Kaggle CLI",
        "files_sha256": {name: sha256(directory / name) for name in FILES},
        "train_studies": len(train),
        "train_series": len(series),
        "test_visible_studies": len(test),
        "test_visible_series": len(test_series),
        "report_nonempty": sum(bool(row["Report"].strip()) for row in train),
        "unique_normalized_reports": len(set(report_norm)),
        "fully_labeled_studies": gold,
        "partially_labeled_studies": partial,
        "labels_by_target": label_counts,
        "series_per_train_study": {
            "min": min(series_counts), "median": statistics.median(series_counts),
            "max": max(series_counts),
        },
        "train_plane_counts": dict(Counter(row["Anatomical_Plane"] for row in series)),
        "train_studies_missing_plane": {
            plane: sum(not any(row["Anatomical_Plane"] == plane for row in rows)
                       for rows in study_series.values())
            for plane in ("Sagittal", "Coronal", "Axial")
        },
        "train_fluid_fat_suppression_disagreements": sum(
            row["Fluid_Sensitive"] != row["Fat_Suppression"] for row in series
        ),
        "test_fluid_fat_suppression_disagreements": sum(
            row["Fluid_Sensitive"] != row["Fat_Suppression"] for row in test_series
        ),
        "test_reports_available": "Report" in test_cols,
        "scope": "Metadata-only; no image decode, clinical validation, or leaderboard score",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.metadata)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Audited {result['train_studies']} training studies and {result['train_series']} series")


if __name__ == "__main__":
    main()
