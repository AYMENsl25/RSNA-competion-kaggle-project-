"""Build local, blinded report-review packets after a development split is frozen."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

TARGETS = ("ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
           "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture")
TRAIN_QUOTA = {"en": 14, "es": 8, "tr": 7, "hr": 7, "el": 7, "de": 7, "bg": 4, "nl": 3, "fr": 3}
VALIDATION_QUOTA = {"en": 30, "es": 18, "tr": 14, "hr": 12, "el": 12, "de": 12,
                    "bg": 10, "nl": 6, "fr": 6}
RARE = {"ACL", "MCL", "Medial OA", "Lateral OA", "PF OA", "Synovitis", "Baker's", "Contusion", "Fracture"}
SEED = "report-review-20261009-v1"


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def rank(uid: str, salt: str) -> str:
    return hashlib.sha256((SEED + "|" + salt + "|" + uid).encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--quarantine", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--frozen-manifest-sha256", required=True,
                        help="Exact approved manifest hash; refuses a changed split")
    args = parser.parse_args()
    actual = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    if actual.lower() != args.frozen_manifest_sha256.lower():
        raise ValueError("Manifest does not match approved frozen hash")
    manifest = {row["StudyInstanceUID"]: row for row in read(args.manifest)}
    reports = {row["StudyInstanceUID"]: row["Report"] for row in read(args.train)}
    quarantine = {row["StudyInstanceUID"] for row in read(args.quarantine)}
    evidence = defaultdict(dict)
    languages = {}
    for row in read(args.evidence):
        uid = row["StudyInstanceUID"]
        languages.setdefault(uid, row["language"])
        evidence[uid][row["target"]] = row

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"seed": SEED, "manifest_sha256": actual, "quarantine_count": len(quarantine),
               "source_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in
                                 (("train.csv", args.train), ("evidence.csv", args.evidence),
                                  ("quarantine.csv", args.quarantine), ("packet_builder.py", Path(__file__)))},
               "sampling_rule": "Per-language quota; half ranked by rare-target lexical evidence, half hashed random controls; all cues are unvalidated and only route review."}
    for partition, quotas, filename in (("candidate_train", TRAIN_QUOTA, "packet_A.csv"),
                                        ("candidate_validation", VALIDATION_QUOTA, "packet_B.csv")):
        chosen = []
        per_language = {}
        selected_groups = set()
        for language, quota in quotas.items():
            pool = [uid for uid, row in manifest.items() if row["partition"] == partition
                    and uid not in quarantine and languages.get(uid) == language and uid in reports]
            if len(pool) < quota:
                raise ValueError(f"Insufficient {language} reports in {partition}: {len(pool)} < {quota}")
            def score(uid: str) -> int:
                return sum(int(evidence[uid].get(target, {}).get("mention", 0)) +
                           2 * int(evidence[uid].get(target, {}).get("severity", 0))
                           for target in RARE)
            def take(candidates: list[str], count: int) -> list[str]:
                picked = []
                for uid in candidates:
                    group = manifest[uid]["group_id"]
                    if group in selected_groups:
                        continue
                    picked.append(uid)
                    selected_groups.add(group)
                    if len(picked) == count:
                        break
                if len(picked) != count:
                    raise ValueError(f"Insufficient independent {language} groups in {partition}")
                return picked
            enriched_n = quota // 2
            enriched = take(sorted(pool, key=lambda uid: (-score(uid), rank(uid, "enriched"))), enriched_n)
            remaining = take(sorted(set(pool) - set(enriched), key=lambda uid: rank(uid, "control")), quota - enriched_n)
            for uid in enriched:
                chosen.append((uid, language, "lexical_enriched"))
            for uid in remaining:
                chosen.append((uid, language, "hashed_control"))
            per_language[language] = {"total": quota, "lexical_enriched": len(enriched),
                                      "hashed_control": len(remaining)}
        if len({uid for uid, _, _ in chosen}) != sum(quotas.values()):
            raise ValueError("Duplicate review study")
        out = args.out_dir / filename
        with out.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(("review_id", "detected_language", "Report",
                             "reviewer_id", "review_date", *[f"{target} state" for target in TARGETS],
                             *[f"{target} evidence span" for target in TARGETS], "notes"))
            for uid, language, stratum in sorted(chosen, key=lambda entry: rank(entry[0], "packet")):
                writer.writerow((rank(uid, "review-id")[:16], language, reports[uid], "", "",
                                 *([""] * len(TARGETS)), *([""] * len(TARGETS)), ""))
        mapping = args.out_dir / (filename.replace(".csv", "_private_uid_map.csv"))
        with mapping.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(("review_id", "StudyInstanceUID", "partition", "selection_stratum", "group_id"))
            for uid, _, stratum in chosen:
                writer.writerow((rank(uid, "review-id")[:16], uid, partition, stratum, manifest[uid]["group_id"]))
        summary[partition] = {"rows": len(chosen), "by_language": per_language,
                              "packet_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
                              "private_uid_map_sha256": hashlib.sha256(mapping.read_bytes()).hexdigest()}
    (args.out_dir / "review_packet_manifest.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value["rows"] for key, value in summary.items() if key.startswith("candidate_")}))


if __name__ == "__main__":
    main()
