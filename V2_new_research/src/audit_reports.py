"""Aggregate report-text properties without exporting any report or study identifier."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter
from pathlib import Path


def quantiles(values: list[int]) -> dict[str, int]:
    ordered = sorted(values)
    return {str(p): ordered[round((len(ordered) - 1) * p / 100)] for p in (0, 25, 50, 75, 90, 95, 99, 100)}


def audit(path: Path) -> dict:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    reports = [row["Report"] for row in rows]
    normalized = [" ".join(report.casefold().split()) for report in reports]
    duplicates = Counter(normalized)
    return {
        "scope": "Aggregate text properties only; no language identification or clinical labels",
        "reports": len(reports),
        "nonempty": sum(bool(report.strip()) for report in reports),
        "character_count_percentiles": quantiles([len(report) for report in reports]),
        "whitespace_word_count_percentiles": quantiles([len(report.split()) for report in reports]),
        "with_non_ascii_character": sum(any(ord(char) > 127 for char in report) for report in reports),
        "with_newline": sum("\n" in report for report in reports),
        "unique_normalized_reports": len(duplicates),
        "duplicate_report_groups": sum(count > 1 for count in duplicates.values()),
        "max_group_size": max(duplicates.values()),
        "with_findings_heading": sum("findings:" in report.casefold() for report in reports),
        "with_impression_heading": sum("impression:" in report.casefold() for report in reports),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Audited {result['reports']} reports; wrote aggregate profile")


if __name__ == "__main__":
    main()
