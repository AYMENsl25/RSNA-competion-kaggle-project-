"""Paired grouped bootstrap of the frozen report-proxy validation predictions."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parents[2]
out = root / "outputs/m00_m01_comparison.json"
targets = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
           "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def auc(y, score):
    positive = y.astype(bool)
    npos, nneg = int(positive.sum()), int((~positive).sum())
    if not npos or not nneg:
        return np.nan
    order = np.argsort(score, kind="mergesort")
    sorted_score = score[order]
    _, first, count = np.unique(sorted_score, return_index=True, return_counts=True)
    ranks = np.empty(len(score), dtype=float)
    for start, size in zip(first, count):
        ranks[order[start:start + size]] = start + (size + 1) / 2
    return (ranks[positive].sum() - npos * (npos + 1) / 2) / (npos * nneg)

split_path = root / "outputs/kaggle_split_source/split.csv"
label_path = root / "outputs/public_label_candidate/llm_labels_full.csv"
assert digest(split_path) == "dbe879ab5cb719e09c0f9aec612d5b6b53648d6f32dd8d405a075482339b3819"
assert digest(label_path) == "96444cd3c7d396bae4a71454317d9cbe0689759b5278b30939781e04d0774537"
with split_path.open(newline="", encoding="utf-8-sig") as file:
    split = {row["StudyInstanceUID"]: row for row in csv.DictReader(file)}
with label_path.open(newline="", encoding="utf-8-sig") as file:
    reader = csv.DictReader(file)
    assert reader.fieldnames == ["StudyInstanceUID", *targets]
    labels = {row["StudyInstanceUID"]: row for row in reader
              if split[row["StudyInstanceUID"]]["partition"] == "candidate_validation"}

predictions = {}
summaries = {}
for model in ("m00", "m01"):
    folder = root / f"outputs/{model}_full_train_v1"
    summary = json.loads((folder / f"{model}_full_train_summary.json").read_text())
    assert summary["completed"] and summary["best"]["epoch"] == 3
    assert digest(folder / f"{model}_best.pt") == summary["best"]["checkpoint_sha256"]
    with (folder / f"{model}_val_predictions_epoch3.csv").open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        assert reader.fieldnames == ["StudyInstanceUID", *targets]
        rows = list(reader)
    uid = [row["StudyInstanceUID"] for row in rows]
    assert len(uid) == len(set(uid)) == 863
    assert hashlib.sha256("\n".join(uid).encode()).hexdigest() == summary["inputs"]["validation_eligible_uid_sha256"]
    assert all(split[u]["partition"] == "candidate_validation" for u in uid)
    values = np.array([[float(row[t]) for t in targets] for row in rows])
    assert np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all()
    predictions[model] = (uid, values)
    summaries[model] = summary

uid, p0 = predictions["m00"]
assert uid == predictions["m01"][0]
p1 = predictions["m01"][1]
assert summaries["m00"]["inputs"] == summaries["m01"]["inputs"]
raw = np.array([[float(labels[u][t]) for t in targets] for u in uid])
assert np.isfinite(raw).all() and ((raw >= 0) & (raw <= 1)).all()
observed = raw != 0.5
y = raw > 0.5
groups = np.array([split[u]["group_id"] for u in uid])
unique_groups, group_inv = np.unique(groups, return_inverse=True)
group_rows = [np.flatnonzero(group_inv == g) for g in range(len(unique_groups))]

def metrics(indices):
    a0, a1 = [], []
    for k in range(len(targets)):
        idx = indices[observed[indices, k]]
        a0.append(auc(y[idx, k], p0[idx, k]))
        a1.append(auc(y[idx, k], p1[idx, k]))
    return np.array(a0), np.array(a1)

base0, base1 = metrics(np.arange(len(uid)))
assert abs(float(np.nanmean(base0)) - summaries["m00"]["best"]["macro_auc"]) < 2e-5
assert abs(float(np.nanmean(base1)) - summaries["m01"]["best"]["macro_auc"]) < 2e-5
rng = np.random.default_rng(20261011)
boot = []
for _ in range(1000):
    sample = rng.integers(0, len(unique_groups), size=len(unique_groups))
    index = np.concatenate([group_rows[g] for g in sample])
    b0, b1 = metrics(index)
    if np.isfinite(b0).all() and np.isfinite(b1).all():
        boot.append(float(np.mean(b1 - b0)))
boot = np.asarray(boot)
assert len(boot) >= 950
result = {
    "comparison": "M01 minus M00, paired frozen report-proxy validation",
    "n_studies": len(uid), "n_groups": len(unique_groups), "bootstrap_seed": 20261011,
    "bootstrap_draws": 1000, "valid_draws": len(boot),
    "m00_macro_auc": float(np.mean(base0)), "m01_macro_auc": float(np.mean(base1)),
    "macro_delta": float(np.mean(base1 - base0)),
    "group_bootstrap_95_percentile_ci": [float(x) for x in np.percentile(boot, [2.5, 97.5])],
    "per_target": {name: {"m00_auc": float(base0[k]), "m01_auc": float(base1[k]),
                          "delta": float(base1[k] - base0[k])} for k, name in enumerate(targets)},
    "input_hashes_verified": True, "uids_identical": True,
}
out.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
