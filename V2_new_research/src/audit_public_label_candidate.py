"""Profile a public report-label candidate without using protected expert labels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


TARGETS = (
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
    "Contusion", "Fracture",
)


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def audit(train_path: Path, labels_path: Path) -> dict:
    train = read(train_path)
    labels = read(labels_path)
    if len(train) != len(labels) or {r["StudyInstanceUID"] for r in train} != {r["StudyInstanceUID"] for r in labels}:
        raise ValueError("Candidate label UIDs do not exactly match official train UIDs")
    expert = {r["StudyInstanceUID"] for r in train if all(r[t] in {"0", "1"} for t in TARGETS)}
    report_groups: dict[str, set[str]] = defaultdict(set)
    for row in train:
        report_groups[digest(" ".join(row["Report"].casefold().split()))].add(row["StudyInstanceUID"])
    excluded = set(expert)
    for group in report_groups.values():
        if group & expert:
            excluded |= group
    candidates = [row for row in labels if row["StudyInstanceUID"] not in excluded]
    buckets = {}
    for target in TARGETS:
        values = [float(row[target]) for row in candidates]
        if not all(math.isfinite(value) and 0 <= value <= 1 for value in values):
            raise ValueError(f"Invalid candidate probabilities for {target}")
        buckets[target] = {
            "neutral_exact_0p5": sum(value == 0.5 for value in values),
            "high_at_least_0p8": sum(value >= 0.8 for value in values),
            "low_at_most_0p2": sum(value <= 0.2 for value in values),
            "other": sum(0.2 < value < 0.8 and value != 0.5 for value in values),
        }
    return {
        "scope": "CC0 public report-derived probability table; descriptive profile only, no comparison with 58 expert labels or model training",
        "source": "https://www.kaggle.com/datasets/stevenleehans/rsna-knee-llm-report-labels",
        "candidate_file_sha256": hashlib.sha256(labels_path.read_bytes()).hexdigest(),
        "rows_in_public_file": len(labels),
        "protected_expert_and_exact_report_links_excluded": len(excluded),
        "nonexpert_candidate_rows_profiled": len(candidates),
        "target_probability_buckets": buckets,
        "next_gate": "Blinded report-only quality review on the nonexpert sample; do not select candidate from its published 58-case score",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.train, args.labels)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Profiled {result['nonexpert_candidate_rows_profiled']} nonexpert candidate rows; no expert evaluation")


if __name__ == "__main__":
    main()
