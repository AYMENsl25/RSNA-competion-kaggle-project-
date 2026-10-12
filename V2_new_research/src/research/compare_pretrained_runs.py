"""Audit matched weak-label validation predictions for M00, P00 and P01A."""
import csv
import hashlib
import json
import runpy
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parents[2]
base = runpy.run_path(str(Path(__file__).with_name("compare_m00_m01.py")))
targets = base["targets"]
split = base["split"]
labels = base["labels"]
auc = base["auc"]
groups = base["group_rows"]
n_groups = len(groups)

models = {
    "M00": ("m00", root / "outputs/m00_full_train_v1"),
    "P00": ("p00", root / "outputs/p00_full_train_v1"),
    "P01A": ("p01a", root / "outputs/p01a_full_train_v1"),
}
predictions = {}
summaries = {}
uids = None
for name, (prefix, folder) in models.items():
    summary = json.loads((folder / f"{prefix}_full_train_summary.json").read_text())
    assert summary["status"] == "complete" and summary["completed"]
    assert summary["best"]["epoch"] == 3
    assert hashlib.sha256((folder / f"{prefix}_best.pt").read_bytes()).hexdigest() == summary["best"]["checkpoint_sha256"]
    assert summary["inputs"]["split_sha256_verified"] and summary["inputs"]["labels_sha256_verified"]
    with (folder / f"{prefix}_val_predictions_epoch3.csv").open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        assert reader.fieldnames == ["StudyInstanceUID", *targets]
        rows = list(reader)
    ids = [row["StudyInstanceUID"] for row in rows]
    assert len(ids) == len(set(ids)) == 863
    assert hashlib.sha256("\n".join(ids).encode()).hexdigest() == summary["inputs"]["validation_eligible_uid_sha256"]
    if uids is None:
        uids = ids
    else:
        assert ids == uids
    scores = np.array([[float(row[target]) for target in targets] for row in rows])
    assert np.isfinite(scores).all() and ((scores >= 0) & (scores <= 1)).all()
    predictions[name], summaries[name] = scores, summary

raw = np.array([[float(labels[uid][target]) for target in targets] for uid in uids])
observed, truth = raw != 0.5, raw > 0.5


def target_aucs(name, indices):
    result = []
    for k in range(len(targets)):
        valid = indices[observed[indices, k]]
        result.append(auc(truth[valid, k], predictions[name][valid, k]))
    return np.array(result)


point = {name: target_aucs(name, np.arange(len(uids))) for name in models}
for name in models:
    assert abs(float(point[name].mean()) - summaries[name]["best"]["macro_auc"]) < 2e-5

rng = np.random.default_rng(20261012)
draws = {"P00_minus_M00": [], "P00_minus_P01A": []}
for _ in range(1000):
    sampled = rng.integers(0, n_groups, size=n_groups)
    indices = np.concatenate([groups[g] for g in sampled])
    measured = {name: target_aucs(name, indices) for name in models}
    if all(np.isfinite(value).all() for value in measured.values()):
        draws["P00_minus_M00"].append(float(np.mean(measured["P00"] - measured["M00"])))
        draws["P00_minus_P01A"].append(float(np.mean(measured["P00"] - measured["P01A"])))

assert all(len(values) >= 950 for values in draws.values())
result = {
    "metric": "12-target macro ROC-AUC on frozen report-proxy validation; not Kaggle leaderboard AUC",
    "studies": len(uids), "groups": n_groups, "bootstrap_seed": 20261012,
    "bootstrap_draws": 1000, "valid_draws": {key: len(value) for key, value in draws.items()},
    "macro_auc": {name: float(values.mean()) for name, values in point.items()},
    "pairwise": {
        "P00_minus_M00": {
            "point_delta": float(np.mean(point["P00"] - point["M00"])),
            "group_bootstrap_95_percentile_ci": [float(x) for x in np.percentile(draws["P00_minus_M00"], [2.5, 97.5])],
        },
        "P00_minus_P01A": {
            "point_delta": float(np.mean(point["P00"] - point["P01A"])),
            "group_bootstrap_95_percentile_ci": [float(x) for x in np.percentile(draws["P00_minus_P01A"], [2.5, 97.5])],
        },
    },
    "per_target": {target: {name: float(point[name][k]) for name in models} for k, target in enumerate(targets)},
    "checkpoint_hashes_verified": True, "validation_ids_identical": True,
    "split_and_label_hashes_verified": True,
}
out = root / "outputs/pretrained_run_comparison.json"
out.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
