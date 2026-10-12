"""Validate a returned blinded report packet without printing report text or study IDs."""

import argparse
import csv
import json
from collections import Counter
from datetime import date
from pathlib import Path

TARGETS = ("ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
           "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture")
STATES = {"positive", "negative", "uncertain", "not_addressed"}


def read(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames, list(reader)


def check(original, reviewed):
    headers, source = read(original)
    returned_headers, rows = read(reviewed)
    errors = Counter()
    counts = {target: Counter() for target in TARGETS}
    if headers != returned_headers or len(source) != len(rows):
        return {"status": "invalid", "rows_expected": len(source), "rows_returned": len(rows),
                "errors": {"header_or_row_count": 1}}
    if len({r["review_id"] for r in rows}) != len(rows):
        errors["duplicate_review_id"] += 1
    completed = 0
    for before, after in zip(source, rows):
        if any(before[key] != after[key] for key in ("review_id", "detected_language", "Report")):
            errors["source_field_changed"] += 1
        states = [after[f"{target} state"].strip().lower() for target in TARGETS]
        if any(states):
            if not after["reviewer_id"].strip():
                errors["missing_reviewer_id"] += 1
            try:
                date.fromisoformat(after["review_date"].strip())
            except ValueError:
                errors["invalid_review_date"] += 1
        if all(state in STATES for state in states):
            completed += 1
        for target, state in zip(TARGETS, states):
            evidence = after[f"{target} evidence span"].strip()
            if state and state not in STATES:
                errors["invalid_state"] += 1
            if state in {"positive", "negative", "uncertain"}:
                if not evidence or evidence not in after["Report"]:
                    errors["missing_or_nonliteral_evidence"] += 1
                counts[target][state] += 1
            elif state == "not_addressed":
                if evidence:
                    errors["evidence_for_not_addressed"] += 1
                counts[target][state] += 1
            elif evidence:
                errors["evidence_without_state"] += 1
    support = {target: {"positive": counts[target]["positive"],
                        "negative": counts[target]["negative"],
                        "meets_20_each": counts[target]["positive"] >= 20 and
                        counts[target]["negative"] >= 20} for target in TARGETS}
    return {"status": "invalid" if errors else "complete" if completed == len(rows) else "incomplete",
            "rows_expected": len(source), "rows_returned": len(rows),
            "complete_rows": completed, "errors": dict(errors), "support": support}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("original", type=Path)
    parser.add_argument("reviewed", type=Path)
    args = parser.parse_args()
    result = check(args.original, args.reviewed)
    print(json.dumps(result, indent=2))
    raise SystemExit(1 if result["status"] == "invalid" else 0)
