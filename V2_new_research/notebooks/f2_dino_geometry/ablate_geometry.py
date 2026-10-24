"""CPU-only C control: identical B architecture/initialization, physical inputs zeroed.

Uses a completed F2 feature cache. No encoder, CUDA, Kaggle, reports or expert labels.
This is a post-pilot diagnostic, not an independent confirmation experiment.
"""
import argparse
import contextlib
import csv
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
from f2_data import sha
from f2_model import Head, masked_bce

spec = importlib.util.spec_from_file_location("f2_run", HERE / "run.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    folder = args.run_dir.resolve()
    cfg = json.loads((folder / "config.json").read_text())
    complete = json.loads((folder / "features_complete.json").read_text())
    if complete["source_sha256"] != runner.source_hashes() or complete["feature_sha256"] != sha(folder / "features.npy"):
        raise ValueError("feature source/data integrity failure")
    out = folder / "geometry_ablation"
    out.mkdir(exist_ok=False)
    start = time.monotonic()
    torch.set_num_threads(cfg["cpu_threads"])
    prepared = runner.prepare(cfg)
    uids, partitions, values = prepared[:3]
    if uids != json.loads((folder / "private_uid_order.json").read_text()):
        raise ValueError("cohort differs from paired pilot")
    data = (torch.from_numpy(np.load(folder / "features.npy").astype(np.float32)),
            torch.from_numpy(np.load(folder / "valid.npy")),
            torch.zeros_like(torch.from_numpy(np.load(folder / "geometry.npy"))))
    target = torch.from_numpy(np.stack([values[u][0] for u in uids]))
    observed = torch.from_numpy(np.stack([values[u][1] for u in uids]))
    tr = np.array([i for i, u in enumerate(uids) if partitions[u] == "candidate_train"])
    va = np.array([i for i, u in enumerate(uids) if partitions[u] == "candidate_validation"])
    # Match B's exact construction order and shared initialization from the paired run.
    torch.manual_seed(cfg["seed"])
    base = Head(width=cfg["head_width"])
    model = Head(width=cfg["head_width"], geometry=True)
    model.load_state_dict(base.state_dict(), strict=False)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg["epochs"])
    history, best = [], None
    for epoch in range(1, cfg["epochs"] + 1):
        torch.manual_seed(cfg["seed"] + epoch)
        model.train()
        order = np.random.default_rng(cfg["seed"] + epoch).permutation(tr)
        for lo in range(0, len(order), cfg["head_batch"]):
            if time.monotonic() - start > 600:
                raise TimeoutError("CPU control exceeded ten minute budget")
            ii = order[lo:lo + cfg["head_batch"]]
            opt.zero_grad(set_to_none=True)
            loss = masked_bce(model(*(x[ii] for x in data)), target[ii], observed[ii])
            if not torch.isfinite(loss):
                raise FloatingPointError("control loss nonfinite")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            opt.step()
        sched.step()
        p, result = runner.evaluate(model, data, va, target, observed, cfg["head_batch"])
        record = {"epoch": epoch, **result}
        history.append(record)
        key = (-result["macro_auc"], result["masked_bce"], epoch)
        if best is None or key < best[0]:
            best = key, record, p.copy()
            temp = out / "f2_C_best.tmp"
            torch.save({"state_dict": model.state_dict(), "config": cfg, "arm": "C",
                        "physical_geometry_zeroed": True, "epoch": epoch,
                        "source_sha256": {**runner.source_hashes(), "ablate_geometry.py": sha(__file__)}}, temp)
            temp.replace(out / "f2_C_best.pt")
        runner.write_json(out / "history.json", history)
        print(json.dumps({"arm": "C", **record}), flush=True)
    pc = best[2]
    np.save(out / "f2_C_val_probabilities.npy", pc, allow_pickle=False)
    pb = np.load(folder / "f2_B_val_probabilities.npy")
    y, obs = target[va].numpy(), observed[va].numpy()
    with (ROOT / cfg["split"]).open(newline="", encoding="utf-8-sig") as f:
        groups = {r["StudyInstanceUID"]: r["group_id"] for r in csv.DictReader(f)}
    gu = [groups[uids[i]] for i in va]
    members = [np.flatnonzero(np.array(gu) == g) for g in sorted(set(gu))]
    rng, differences = np.random.default_rng(cfg["seed"]), []
    for _ in range(cfg["bootstrap_draws"]):
        ii = np.concatenate([members[j] for j in rng.integers(len(members), size=len(members))])
        try:
            differences.append(runner.metrics(y[ii], obs[ii], pb[ii])["macro_auc"] - runner.metrics(y[ii], obs[ii], pc[ii])["macro_auc"])
        except ValueError:
            continue
    mb, mc = runner.metrics(y, obs, pb), runner.metrics(y, obs, pc)
    result = {"status": "complete", "B_with_geometry": mb, "C_same_capacity_zero_geometry": mc,
              "B_minus_C_macro_auc": mb["macro_auc"] - mc["macro_auc"],
              "paired_group_bootstrap_95pct_exploratory": np.percentile(differences, [2.5, 97.5]).tolist(),
              "C_selected_epoch": best[1]["epoch"], "C_parameters": sum(p.numel() for p in model.parameters()),
              "source_sha256": {**runner.source_hashes(), "ablate_geometry.py": sha(__file__)},
              "elapsed_seconds": time.monotonic() - start, "kaggle_gpu_hours_used": 0,
              "limitation": "post-pilot diagnostic on reused weak-label validation; no independent F0 comparison"}
    runner.write_json(out / "comparison.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
