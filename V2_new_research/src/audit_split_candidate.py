"""Check expert exclusion and exact-report grouping; do not freeze a split yet."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
    "Contusion", "Fracture",
]


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def audit(path: Path) -> dict:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        normalized = " ".join(row["Report"].casefold().split())
        groups[digest(normalized)].append(row)

    def expert(row: dict[str, str]) -> bool:
        return all(row[target] in {"0", "1"} for target in TARGETS)

    expert_groups = {key for key, members in groups.items() if any(expert(row) for row in members)}
    expert_rows = [row for row in rows if expert(row)]
    linked_nonexpert = sum(not expert(row) for key in expert_groups for row in groups[key])
    eligible = [row for key, members in groups.items() if key not in expert_groups for row in members]
    sizes = Counter(map(len, groups.values()))
    return {
        "scope": "Exact normalized-report grouping only; candidate exclusion, not a frozen patient-safe split",
        "source_rows": len(rows),
        "expert_studies_excluded": len(expert_rows),
        "nonexpert_studies_linked_to_expert_by_exact_report": linked_nonexpert,
        "candidate_eligible_studies_before_patient_image_duplicate_checks": len(eligible),
        "exact_report_groups": len(groups),
        "groups_with_multiple_studies": sum(len(members) > 1 for members in groups.values()),
        "largest_exact_report_group": max(sizes),
        "expert_groups": len(expert_groups),
        "remaining_unknowns": [
            "Patient identifiers and de-identified subject linkage across studies",
            "Near-duplicate reports with changed wording",
            "Duplicate or near-duplicate DICOM images across study IDs",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Candidate eligible: {result['candidate_eligible_studies_before_patient_image_duplicate_checks']}; split not frozen")


if __name__ == "__main__":
    main()
