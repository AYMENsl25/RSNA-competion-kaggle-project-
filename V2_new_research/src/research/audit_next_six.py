"""Read-only audit of existing V3 development predictions; never opens expert labels.

Run from V2_new_research: .venv/Scripts/python src/research/audit_next_six.py
Writes only research/NEXT_SIX_EVIDENCE_AUDIT.json. No training or submission.
"""
import csv
import hashlib
import json
import platform
from collections import Counter
from pathlib import Path

import numpy as np
import scipy
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs"
TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
           "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def auc(y, p):
    npos = int(y.sum())
    nneg = len(y) - npos
    if not npos or not nneg:
        return np.nan
    return float((rankdata(p)[y].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def main():
    split_path = OUT / "kaggle_split_source/split.csv"
    split = {r["StudyInstanceUID"]: r for r in rows(split_path)}
    uids = sorted(u for u, r in split.items() if r["partition"] == "candidate_validation")
    label_paths = {"public": OUT / "public_label_candidate/llm_labels_full.csv",
                   "v3": OUT / "labels_v3/labels_v3.csv"}
    labels = {}
    for name, path in label_paths.items():
        # Read only target cells belonging to development-validation studies.
        table = {r["StudyInstanceUID"]: r for r in rows(path)}
        labels[name] = np.array([[float(table[u][t]) for t in TARGETS] for u in uids])
    files = {"E1": "e1_v3/val_pred_epoch12.csv", "E45": "e45_v3/val_pred_epoch12.csv",
             "E2": "e2_v3/val_pred_epoch12.csv", "E6_ep16": "e6_v3/val_pred_epoch16.csv",
             "E6_ep15": "e6_v3/val_pred_epoch15.csv", "E7": "e7_320/val_pred_epoch12.csv",
             "E8": "e8_cnxt/val_pred_epoch12.csv", "E3_ep11": "e3_cnxt6/val_pred_epoch11.csv",
             "E3_ep12": "e3_cnxt6/val_pred_epoch12.csv", "Ens3": "ens_e45_e6_e2.csv",
             "Ens4": "ens_e45_e6_e2_e8.csv", "Ens5": "ens5.csv"}
    predictions, coverage = {}, {}
    for name, rel in files.items():
        table_rows = rows(OUT / rel)
        table = {r["StudyInstanceUID"]: r for r in table_rows}
        assert len(table_rows) == len(table) == len(uids), (name, "duplicates or row-count mismatch")
        assert set(table) == set(uids), (name, "wrong study set")
        p = np.array([[float(table[u][t]) for t in TARGETS] for u in uids])
        assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(), name
        predictions[name] = p
        coverage[name] = {"rows": len(table), "sha256": digest(OUT / rel), "file": rel}

    def per_target(p, label_name="public", idx=None):
        idx = np.arange(len(uids)) if idx is None else idx
        lab = labels[label_name][idx]
        pred = p[idx]
        return np.array([auc(lab[lab[:, k] != .5, k] > .5, pred[lab[:, k] != .5, k])
                         for k in range(12)])

    metrics = {}
    for name, p in predictions.items():
        metrics[name] = {}
        for label_name in labels:
            a = per_target(p, label_name)
            metrics[name][label_name] = {"macro": float(a.mean()),
                                       "macro_no_synovitis": float(np.delete(a, 8).mean()),
                                       "per_target": dict(zip(TARGETS, a.tolist()))}

    group_ids, inv = np.unique([split[u]["group_id"] for u in uids], return_inverse=True)
    groups = [np.flatnonzero(inv == g) for g in range(len(group_ids))]
    pairs = [("E45", "E2"), ("E45", "E6_ep16"), ("E45", "E7"), ("E45", "E8"),
             ("E8", "E3_ep11"), ("Ens3", "Ens4"), ("Ens4", "Ens5")]
    # Frozen single-split predictions: these intervals do not include training-seed
    # variation, repeated model selection, weak-label error or test distribution shift.
    rng = np.random.default_rng(20261012)
    deltas = {f"{a}->{b}": [] for a, b in pairs}
    for _ in range(1000):
        idx = np.concatenate([groups[g] for g in rng.integers(0, len(groups), len(groups))])
        scores = {n: per_target(predictions[n], idx=idx) for n in {x for p in pairs for x in p}}
        for a, b in pairs:
            delta = scores[b] - scores[a]
            if np.isfinite(delta).all():
                deltas[f"{a}->{b}"].append(float(delta.mean()))
    comparisons = {}
    for a, b in pairs:
        k = f"{a}->{b}"
        comparisons[k] = {"delta": metrics[b]["public"]["macro"] - metrics[a]["public"]["macro"],
                          "ci95": np.percentile(deltas[k], [2.5, 97.5]).tolist(),
                          "valid_draws": len(deltas[k])}

    ensemble_checks = {}
    for name, members in {"Ens3": ["E45", "E6_ep16", "E2"],
                          "Ens4": ["E45", "E6_ep16", "E2", "E8"],
                          "Ens5_ep11": ["E45", "E6_ep16", "E2", "E8", "E3_ep11"],
                          "Ens5_ep12": ["E45", "E6_ep16", "E2", "E8", "E3_ep12"]}.items():
        pred = np.mean([rankdata(predictions[n], axis=0) / len(uids) for n in members], axis=0)
        saved_name = "Ens5" if name.startswith("Ens5") else name
        ensemble_checks[name] = {"members": members, "public_macro": float(per_target(pred).mean()),
                                "max_abs_diff_saved": float(np.abs(pred - predictions[saved_name]).max()),
                                "max_rank_diff_saved": float(np.abs(rankdata(pred, axis=0) -
                                                              rankdata(predictions[saved_name], axis=0)).max())}

    runs = {}
    for name in ["e1_v3", "e45_v3", "e2_v3", "e6_v3", "e7_320", "e8_cnxt", "e3_cnxt6"]:
        log = json.loads((OUT / name / "log.json").read_text())
        epoch_rows = log["epochs"]
        runs[name] = {"args": log["args"], "train_studies": log["train_studies"],
                      "val_studies": log["val_studies"], "best": log["best"],
                      "sum_train_val_hours": sum(e["train_seconds"] + e["val_seconds"] for e in epoch_rows) / 3600,
                      "epoch_count": len(epoch_rows)}
    result = {"scope": "Existing V3 weak-label validation only; no expert labels or new inference read/run",
              "environment": {"python": platform.python_version(), "numpy": np.__version__,
                              "scipy": scipy.__version__, "script_sha256": digest(Path(__file__))},
              "split_sha256": digest(split_path), "split_partition_counts": dict(Counter(r["partition"] for r in split.values())),
              "validation_studies": len(uids), "validation_groups": len(group_ids),
              "bootstrap_draws": 1000, "bootstrap_seed": 20261012,
              "label_sha256": {k: digest(p) for k, p in label_paths.items()},
              "observed_cells": {name: {t: {"n": int((p[:, k] != .5).sum()),
                                                   "positive": int((p[:, k] > .5).sum())}
                                         for k, t in enumerate(TARGETS)} for name, p in labels.items()},
              "prediction_coverage": coverage, "metrics": metrics, "paired_comparisons": comparisons,
              "ensemble_reconstruction": ensemble_checks, "runs": runs}
    path = ROOT / "research/NEXT_SIX_EVIDENCE_AUDIT.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(path), "validation_studies": len(uids), "validation_groups": len(group_ids),
                      "macros_public": {k: round(v["public"]["macro"], 6) for k, v in metrics.items()},
                      "pairs": comparisons, "ensemble_reconstruction": ensemble_checks,
                      "hours": {k: round(v["sum_train_val_hours"], 3) for k, v in runs.items()}}, indent=2))


if __name__ == "__main__":
    main()
