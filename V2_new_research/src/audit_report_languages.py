"""Profile likely report languages and draw a blinded local review sample."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from langdetect import DetectorFactory, LangDetectException, detect_langs


TARGETS = (
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
    "Contusion", "Fracture",
)


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def audit(input_csv: Path, aggregate_json: Path, private_sample_csv: Path) -> None:
    DetectorFactory.seed = 0
    with input_csv.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    expert = {row["StudyInstanceUID"] for row in rows if all(row[t] in {"0", "1"} for t in TARGETS)}
    by_report = defaultdict(set)
    for row in rows:
        by_report[digest(" ".join(row["Report"].casefold().split()))].add(row["StudyInstanceUID"])
    excluded = set(expert)
    for group in by_report.values():
        if group & expert:
            excluded.update(group)

    candidates = []
    counts = Counter()
    uncertain_count = 0
    for row in rows:
        if row["StudyInstanceUID"] in excluded:
            continue
        try:
            guesses = detect_langs(row["Report"])
            lang = guesses[0].lang
            confidence = float(guesses[0].prob)
        except LangDetectException:
            lang, confidence = "und", 0.0
        if confidence < 0.90:
            uncertain_count += 1
            lang = "und"
        counts[lang] += 1
        candidates.append((row, lang, confidence))

    sample = []
    for language, _ in counts.most_common(6):
        subset = [x for x in candidates if x[1] == language]
        subset.sort(key=lambda x: digest(x[0]["StudyInstanceUID"] + "|rsna-language-review-v1"))
        sample.extend(subset[:15])
    sample_ids = {item[0]["StudyInstanceUID"] for item in sample}
    remainder = [x for x in candidates if x[0]["StudyInstanceUID"] not in sample_ids]
    remainder.sort(key=lambda x: digest(x[0]["StudyInstanceUID"] + "|rsna-random-review-v1"))
    sample.extend(remainder[:20])

    aggregate = {
        "scope": "Heuristic language identification only; verify languages in manual review",
        "nonexpert_unlinked_reports": len(candidates),
        "language_counts": dict(counts.most_common()),
        "low_confidence_or_undetermined": uncertain_count,
        "blinded_review_sample_size": len(sample),
        "review_sample_stored_at": "ignored outputs/report_review_sample.csv",
        "sample_policy": "15 per six most common detected languages plus 20 deterministic random remainder; expert and exact-report-linked studies excluded",
    }
    aggregate_json.parent.mkdir(parents=True, exist_ok=True)
    aggregate_json.write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")
    private_sample_csv.parent.mkdir(parents=True, exist_ok=True)
    with private_sample_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["StudyInstanceUID", "detected_language", "language_confidence", "Report"])
        for row, lang, confidence in sample:
            writer.writerow([row["StudyInstanceUID"], lang, round(confidence, 4), row["Report"]])
    print(f"Profiled {len(candidates)} reports; saved {len(sample)} locally for blinded review")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--aggregate", required=True, type=Path)
    parser.add_argument("--sample", required=True, type=Path)
    args = parser.parse_args()
    audit(args.input, args.aggregate, args.sample)


if __name__ == "__main__":
    main()
