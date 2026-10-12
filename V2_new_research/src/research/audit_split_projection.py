"""Verify that the Kaggle split payload is an exact, minimal projection of frozen v3."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
source = ROOT / "outputs/candidate_split_v3/manifest.csv"
payload = ROOT / "outputs/kaggle_split_source/split.csv"
expected_source = "b5926bd9d82154debee602a1824cf874aed1d0f409386308a387ed6a2f49debc"
expected_payload = "dbe879ab5cb719e09c0f9aec612d5b6b53648d6f32dd8d405a075482339b3819"
digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
assert digest(source) == expected_source
assert digest(payload) == expected_payload
with source.open(newline="", encoding="utf-8-sig") as file:
    original = list(csv.DictReader(file))
with payload.open(newline="", encoding="utf-8-sig") as file:
    reader = csv.DictReader(file)
    assert reader.fieldnames == ["StudyInstanceUID", "group_id", "partition"]
    projected = list(reader)
assert projected == [{key: row[key] for key in reader.fieldnames} for row in original]
assert len(projected) == 4407
assert len({row["StudyInstanceUID"] for row in projected}) == 4407
counts = {name: sum(row["partition"] == name for row in projected)
          for name in ("candidate_train", "candidate_validation", "excluded")}
assert counts == {"candidate_train": 3457, "candidate_validation": 885, "excluded": 65}
assert all(row["partition"] in counts for row in projected)
group_parts = {}
for row in projected:
    group_parts.setdefault(row["group_id"], set()).add(row["partition"])
assert all(len(parts) == 1 for parts in group_parts.values())
print(json.dumps({"source_sha256": expected_source, "payload_sha256": expected_payload,
                  "columns": reader.fieldnames, "rows": len(projected), "counts": counts,
                  "group_count": len(group_parts), "exact_projection": True}, indent=2))
