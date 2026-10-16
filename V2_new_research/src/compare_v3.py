"""Paired group bootstrap between two validation prediction CSVs (StudyInstanceUID + 12 targets).
Same procedure as outputs/compare_p00_e01.py (groups from the frozen split, 1000 draws, seed
20261012, AUC truth p>0.5 with p==0.5 masked), on the UIDs both files share.

python src/compare_v3.py A.csv B.csv --labels public [--out result.json]   -> reports B - A
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_v3 import LABELS, TARGETS, read_labels, read_split, roc_auc  # noqa: E402


def read_preds(path):
    with open(path, newline="", encoding="utf-8") as f:
        return {r["StudyInstanceUID"]: [float(r[t]) for t in TARGETS] for r in csv.DictReader(f)}


def per_target(scores, y, obs, idx):
    out = []
    for k in range(len(TARGETS)):
        v = idx[obs[idx, k]]
        a = roc_auc(y[v, k], scores[v, k])
        out.append(np.nan if a is None else a)
    return np.array(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--labels", choices=LABELS, default="public")
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20261012)
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    split, lab = read_split(), read_labels(args.labels)
    pa, pb = read_preds(args.a), read_preds(args.b)
    uids = sorted(set(pa) & set(pb) & set(lab))
    assert all(split[u][1] == "candidate_validation" for u in uids), "non-validation study in predictions"
    sa, sb = np.array([pa[u] for u in uids]), np.array([pb[u] for u in uids])
    p = np.array([lab[u] for u in uids])
    y, obs = p > 0.5, p != 0.5
    _, inv = np.unique([split[u][0] for u in uids], return_inverse=True)
    groups = [np.flatnonzero(inv == g) for g in range(inv.max() + 1)]
    nosyn = [k for k, t in enumerate(TARGETS) if t != "Synovitis"]
    all_idx = np.arange(len(uids))
    a0, b0 = per_target(sa, y, obs, all_idx), per_target(sb, y, obs, all_idx)
    rng = np.random.default_rng(args.seed)
    d, dn = [], []
    for _ in range(args.draws):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        delta = per_target(sb, y, obs, idx) - per_target(sa, y, obs, idx)
        if np.isfinite(delta).all():
            d.append(delta.mean())
            dn.append(delta[nosyn].mean())
    res = {"a": str(args.a), "b": str(args.b), "labels": args.labels, "studies": len(uids),
           "only_in_a": len(set(pa) - set(pb)), "only_in_b": len(set(pb) - set(pa)),
           "groups": len(groups), "valid_draws": len(d),
           "a_macro": float(np.nanmean(a0)), "b_macro": float(np.nanmean(b0)),
           "b_minus_a": float(np.nanmean(b0) - np.nanmean(a0)),
           "ci95": np.percentile(d, [2.5, 97.5]).tolist() if d else None,
           "b_minus_a_no_synovitis": float(np.nanmean(b0[nosyn]) - np.nanmean(a0[nosyn])),
           "ci95_no_synovitis": np.percentile(dn, [2.5, 97.5]).tolist() if dn else None,
           "per_target": {t: {"a": float(a0[k]), "b": float(b0[k]), "delta": float(b0[k] - a0[k])}
                          for k, t in enumerate(TARGETS)}}
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    main()
