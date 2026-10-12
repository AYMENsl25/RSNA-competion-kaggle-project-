"""Fix local candidate/control inputs; build a standalone private CPU script."""
import csv
import hashlib
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
V2 = HERE.parents[1]


def read(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def main():
    source = V2 / "outputs/image_duplicate_audit/near_image_pairs.csv"
    manifest = V2 / "outputs/candidate_split/manifest.csv"
    series_source = V2 / "data/metadata/train_series.csv"
    first = {}
    for row in read(series_source):
        first.setdefault(row["StudyInstanceUID"], row)
    manifest_rows = read(manifest)
    partitions = {row["StudyInstanceUID"]: row["partition"] for row in manifest_rows}
    expert_uids = sorted(row["StudyInstanceUID"] for row in manifest_rows if row["exclusion_reason"] == "expert_holdout")
    assert len(expert_uids) == 58
    pairs = read(source)
    assert len(pairs) == 272 and sum(row["touches_expert"] == "1" for row in pairs) == 1
    selected = []
    candidate_set = set()
    for row in pairs:
        a, b = row["study_uid_a"], row["study_uid_b"]
        candidate_set.add(tuple(sorted((a, b))))
        selected.append([a, b, "candidate", row["touches_expert"] == "1",
                         {partitions[a], partitions[b]} == {"candidate_train", "candidate_validation"}])
    eligible = sorted(uid for uid, partition in partitions.items() if partition != "excluded")
    rng = random.Random(20261009)
    controls = set()
    attempts = 0
    while len(controls) < 100:
        attempts += 1
        if attempts > 100000:
            raise RuntimeError("Could not obtain 100 same-plane controls")
        a, b = sorted(rng.sample(eligible, 2))
        if (a, b) not in candidate_set and all(first[a][key] == first[b][key] for key in ("Anatomical_Plane", "Fluid_Sensitive", "Fat_Suppression")):
            controls.add((a, b))
    selected.sort(key=lambda pair: (not pair[3], not pair[4], pair[0], pair[1]))
    selected.extend([a, b, "control", False, False] for a, b in sorted(controls))
    referenced = {uid for pair in selected for uid in pair[:2]}
    payload = {"pairs": selected, "expert_pixel_quarantine": expert_uids,
               "series": {uid: first[uid] for uid in sorted(referenced)},
               "source_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest()
                                 for name, path in [("candidate_edges", source), ("candidate_manifest", manifest), ("train_series", series_source)]}}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    script = (HERE / "confirm_template.py").read_text(encoding="utf-8").replace("__INPUT_JSON__", repr(encoded))
    (HERE / "confirm.py").write_text(script, encoding="utf-8")
    summary = {"candidate_pairs": 272, "same_plane_random_controls": 100, "control_seed": 20261009,
               "candidate_cross_fold_pairs": sum(pair[4] for pair in selected),
               "expert_linked_header_only_pairs": 1,
               "control_matching": "First screened series matches plane, fluid-sensitive and fat-suppression metadata; all controls nonexcluded",
               "payload_sha256": hashlib.sha256(encoded.encode()).hexdigest(),
               "script_sha256": hashlib.sha256(script.encode()).hexdigest(), "source_sha256": payload["source_sha256"]}
    (HERE / "selection_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
