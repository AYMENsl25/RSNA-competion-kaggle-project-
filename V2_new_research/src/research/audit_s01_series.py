"""Aggregate S01 series coverage; never export study or series identifiers."""

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SERIES = ROOT / "data/metadata/train_series.csv"
SPLIT = ROOT / "outputs/kaggle_split_source/split.csv"
OUT = ROOT / "research/s01_series_coverage.json"
ROLES = ("Sagittal", "Coronal", "Axial")
SPLIT_SHA = "dbe879ab5cb719e09c0f9aec612d5b6b53648d6f32dd8d405a075482339b3819"


def main():
    assert hashlib.sha256(SPLIT.read_bytes()).hexdigest() == SPLIT_SHA
    with SPLIT.open(newline="", encoding="utf-8-sig") as f:
        partition = {r["StudyInstanceUID"]: r["partition"] for r in csv.DictReader(f)}
    grouped = defaultdict(lambda: defaultdict(dict))
    with SERIES.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            uid, sid = row["StudyInstanceUID"], row["SeriesInstanceUID"]
            role = row["Anatomical_Plane"].strip().capitalize()
            assert uid in partition
            if role in ROLES:
                assert sid not in grouped[uid][role], "duplicate series UID"
                grouped[uid][role][sid] = row
    assert set(grouped) == set(partition)
    result = {"source_sha256": hashlib.sha256(SERIES.read_bytes()).hexdigest(),
              "split_sha256": SPLIT_SHA, "partitions": {}}
    for part in sorted(set(partition.values())):
        studies = [uid for uid, value in partition.items() if value == part]
        role_hist = {role: Counter() for role in ROLES}
        selected = first = 0
        for uid in studies:
            for role in ROLES:
                n = len(grouped[uid][role])
                role_hist[role][min(n, 3)] += 1
                first += min(n, 1)
                selected += min(n, 2)
        result["partitions"][part] = {
            "studies": len(studies),
            "one_series_policy_loads": first,
            "two_series_policy_loads": selected,
            "added_loads": selected - first,
            "role_series_count_histogram_3_means_3_or_more": {
                role: {str(k): role_hist[role][k] for k in range(4)} for role in ROLES}}
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
