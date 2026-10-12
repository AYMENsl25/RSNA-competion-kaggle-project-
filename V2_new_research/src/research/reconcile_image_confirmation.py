"""Join private Kaggle image-audit decisions locally; never change the split."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--private-out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    args = parser.parse_args()

    pair_rows = rows(args.pairs)
    manifest = {r["StudyInstanceUID"]: r["partition"] for r in rows(args.manifest)}
    decisions = rows(args.decisions)
    by_hash = {r["pair_sha256"]: r["decision"] for r in decisions}
    if len(by_hash) != len(decisions):
        raise ValueError("Duplicate decision hashes")
    if len(pair_rows) != 272 or len(decisions) > len(pair_rows):
        raise ValueError("Unexpected candidate/decision count")

    counts: dict[str, Counter[str]] = defaultdict(Counter)
    private = []
    used = set()
    for row in pair_rows:
        a, b = sorted((row["study_uid_a"], row["study_uid_b"]))
        digest = hashlib.sha256((a + "|" + b).encode()).hexdigest()
        decision = by_hash.get(digest, "unprocessed")
        if digest in by_hash:
            used.add(digest)
        partition_pair = "/".join(sorted((manifest[a], manifest[b])))
        stratum = "expert_linked" if row["touches_expert"] == "1" else (
            "cross_fold" if {manifest[a], manifest[b]} == {"candidate_train", "candidate_validation"}
            else "within_partition"
        )
        counts[stratum][decision] += 1
        counts[partition_pair][decision] += 1
        private.append((a, b, partition_pair, stratum, decision))
    if used != set(by_hash):
        raise ValueError("Decision hash absent from local candidate set")

    args.private_out.parent.mkdir(parents=True, exist_ok=True)
    with args.private_out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("study_uid_a", "study_uid_b", "partition_pair", "stratum", "decision"))
        writer.writerows(private)
    summary = {
        "status": "candidate_similarity_triage_not_patient_identity_or_split_clearance",
        "candidate_pairs": len(pair_rows),
        "decision_rows": len(decisions),
        "counts": {stratum: dict(counter) for stratum, counter in sorted(counts.items())},
        "source_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in
                          (args.pairs, args.manifest, args.decisions)},
        "limitations": ["Threshold categories are audit triage, not proof of identical patients.",
                        "Unprocessed and unresolved cross-fold pairs still block split freeze.",
                        "Expert-linked nonexpert study remains quarantined until confirmed or excluded."],
    }
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"candidate_pairs": len(pair_rows), "decision_rows": len(decisions),
                      "cross_fold": dict(counts["cross_fold"]),
                      "expert_linked": dict(counts["expert_linked"])}, indent=2))


if __name__ == "__main__":
    main()
