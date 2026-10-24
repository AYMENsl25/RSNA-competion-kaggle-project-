"""Local F2 pilot. Explicit modes: preflight, run, benchmark, analyze.

No Kaggle API calls, report parsing, expert evaluation, or test inference. A separate
supervisor process bounds total runtime, including setup, encoding and both heads.
"""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import torch
import torch.nn.functional as F

from f2_data import TARGETS, cache_index, centers_geometry, cohort, select, sha
from f2_model import Head, masked_bce


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def source_hashes():
    paths = [Path(__file__), ROOT / "src/f2_data.py", ROOT / "src/f2_model.py"]
    return {str(p.relative_to(ROOT)): sha(p) for p in paths}


def prepare(cfg):
    uids, partitions, values = cohort(ROOT / cfg["split"], ROOT / cfg["labels"])
    rows, arrays, provenance = cache_index(ROOT / cfg["cache"], set(uids), partitions)
    plans, failures, failed_uids = {}, Counter(), []
    for uid in uids:
        try:
            selected = select(rows[uid])
            if not selected:
                raise ValueError("study has no selected series")
            plans[uid] = [(slot, row, *centers_geometry(row, cfg["windows_per_series"]))
                          for slot, row in sorted(selected.items())]
        except ValueError as e:
            failures[str(e)] += 1
            failed_uids.append(uid)
    failed_train = sum(partitions[u] == "candidate_train" for u in failed_uids)
    failed_val = sum(partitions[u] == "candidate_validation" for u in failed_uids)
    accepted_failures = (failed_train <= cfg["max_failed_training_studies"] and
                         failed_val <= cfg["max_failed_validation_studies"])
    used = [u for u in uids if u not in failed_uids]
    info = {"eligible_studies": len(uids), "failed_studies": sum(failures.values()),
            "failed_training_studies": failed_train, "failed_validation_studies": failed_val,
            "failure_gate_passed": accepted_failures,
            "failures": dict(failures), "cache": provenance,
            "train_studies": sum(partitions[u] == "candidate_train" for u in uids),
            "validation_studies": sum(partitions[u] == "candidate_validation" for u in uids),
            "train_studies_used": sum(partitions[u] == "candidate_train" for u in used),
            "validation_studies_used": sum(partitions[u] == "candidate_validation" for u in used),
            "failed_uid_order_sha256": hashlib.sha256("\n".join(failed_uids).encode()).hexdigest(),
            "uid_order_sha256": hashlib.sha256("\n".join(used).encode()).hexdigest(),
            "split_sha256": sha(ROOT / cfg["split"]), "labels_sha256": sha(ROOT / cfg["labels"]),
            "selected_series": sum(len(p) for p in plans.values()),
            "windows": sum(len(c) for p in plans.values() for _, _, c, _ in p),
            "reports_read": False, "expert_labels_read": False, "test_data_read": False}
    return used, partitions, values, plans, arrays, info


def encoder(cfg, device):
    import timm
    path = ROOT / cfg["weights"]
    manifest = json.loads((path.parent / "manifest.json").read_text())
    if sha(path) != manifest["sha256"]:
        raise ValueError("generic encoder checksum differs from download manifest")
    # Local file overlay prevents any implicit hub/network lookup or downloaded code.
    net = timm.create_model(cfg["model"], pretrained=True, num_classes=0,
                            img_size=cfg["image_size"],
                            pretrained_cfg_overlay={"file": str(path), "hf_hub_id": None, "url": ""})
    net.requires_grad_(False).eval().to(device)
    if net.num_features != 384 or any(p.requires_grad for p in net.parameters()):
        raise ValueError("unexpected encoder dimensions or unfrozen parameters")
    return net, manifest


@torch.inference_mode()
def study_features(plan, arrays, net, cfg, device):
    windows = cfg["windows_per_series"]
    features = np.zeros((6, windows, 384), np.float16)
    valid = np.zeros((6, windows), bool)
    geometry = np.zeros((6, windows, 5), np.float32)
    images, destinations = [], []
    for slot, row, centers, geom in plan:
        start = int(row["start"])
        array = arrays[row["shard"]]
        for j, c in enumerate(centers):
            images.append(np.array(array[start + c - 1:start + c + 2], copy=True))
            destinations.append((slot, j))
        geometry[slot, :len(centers)] = geom
    mean = torch.tensor((.485, .456, .406), device=device)[None, :, None, None]
    std = torch.tensor((.229, .224, .225), device=device)[None, :, None, None]
    for lo in range(0, len(images), cfg["encoder_batch"]):
        x = torch.from_numpy(np.stack(images[lo:lo + cfg["encoder_batch"]])).to(device).float() / 255
        x = F.interpolate(x, size=(cfg["image_size"],) * 2, mode="bicubic",
                          align_corners=False, antialias=True).clamp_(0, 1)
        x = (x - mean) / std
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            z = net(x)
        if z.ndim != 2 or z.shape[1] != 384 or not torch.isfinite(z).all():
            raise FloatingPointError("invalid frozen features")
        z = z.float().cpu().numpy()
        for k, (slot, j) in enumerate(destinations[lo:lo + len(z)]):
            features[slot, j], valid[slot, j] = z[k], True
    if not valid.any():
        raise ValueError("no features")
    return features, valid, geometry


def extract(cfg, out, prepared, start, benchmark_only=False):
    uids, partitions, values, plans, arrays, info = prepared
    if not info["failure_gate_passed"]:
        raise ValueError("selected series QC failures; no substitutions permitted")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("local encoder benchmark requires CUDA; no surprise CPU extraction")
    net, weight_manifest = encoder(cfg, device)
    write_json(out / "encoder.json", weight_manifest)
    n = len(uids)
    # Balanced deterministic benchmark avoids trusting only easy/short studies.
    order = sorted(range(n), key=lambda i: hashlib.sha256(uids[i].encode()).digest())
    benchmark = order[:cfg["benchmark_studies"]]
    torch.cuda.reset_peak_memory_stats()
    study_features(plans[uids[benchmark[0]]], arrays, net, cfg, device)  # warmup
    measured, sample = [], {}
    for i in benchmark:
        torch.cuda.synchronize()
        t = time.monotonic()
        sample[i] = study_features(plans[uids[i]], arrays, net, cfg, device)
        torch.cuda.synchronize()
        measured.append(time.monotonic() - t)
    benchmark_windows = sum(int(v[1].sum()) for v in sample.values())
    seconds_per_window = sum(measured) / benchmark_windows
    projected = 1.5 * seconds_per_window * info["windows"]
    remaining = cfg["hard_wall_seconds"] - cfg["save_reserve_seconds"] - (time.monotonic() - start)
    gate = min(cfg["extraction_budget_seconds"], remaining - 1200)
    result = {"status": "passed" if projected < gate else "blocked_by_runtime_projection",
              "studies": len(benchmark), "windows": benchmark_windows,
              "measured_seconds": sum(measured), "projected_extraction_seconds_with_50pct_headroom": projected,
              "extraction_gate_seconds": gate, "gpu": torch.cuda.get_device_name(),
              "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
              "kaggle_gpu_hours_used": 0}
    write_json(out / "benchmark.json", result)
    print(json.dumps({"stage": "encoder_benchmark", **result}), flush=True)
    if benchmark_only or result["status"] != "passed":
        return None
    feat = np.lib.format.open_memmap(out / "features.npy", mode="w+", dtype=np.float16,
                                     shape=(n, 6, cfg["windows_per_series"], 384))
    mask = np.zeros((n, 6, cfg["windows_per_series"]), bool)
    geom = np.zeros((*mask.shape, 5), np.float32)
    extract_start = time.monotonic()
    for k, i in enumerate(order):
        if time.monotonic() - extract_start > cfg["extraction_budget_seconds"] or time.monotonic() - start > cfg["hard_wall_seconds"] - cfg["save_reserve_seconds"] - 1200:
            raise TimeoutError("extraction deadline; features are incomplete and unusable")
        feat[i], mask[i], geom[i] = sample.pop(i) if i in sample else study_features(plans[uids[i]], arrays, net, cfg, device)
        if (k + 1) % 50 == 0:
            feat.flush()
            status = {"stage": "extract", "studies_done": k + 1, "studies_total": n,
                      "elapsed_seconds": time.monotonic() - start}
            write_json(out / "status.json", status)
            print(json.dumps(status), flush=True)
    feat.flush()
    np.save(out / "valid.npy", mask, allow_pickle=False)
    np.save(out / "geometry.npy", geom, allow_pickle=False)
    write_json(out / "private_uid_order.json", uids)
    write_json(out / "features_complete.json", {"studies": n, "source_sha256": source_hashes(),
              "feature_sha256": sha(out / "features.npy"), "uid_order_sha256": info["uid_order_sha256"],
              "extraction_seconds": time.monotonic() - extract_start})
    del net
    torch.cuda.empty_cache()
    return feat, mask, geom


def evaluate(model, data, indices, target, observed, batch):
    model.eval()
    preds = []
    with torch.no_grad():
        for offset in range(0, len(indices), batch):
            ii = indices[offset:offset + batch]
            logits = model(*(x[ii] for x in data))
            if not torch.isfinite(logits).all():
                raise FloatingPointError("nonfinite validation logits")
            preds.append(logits)
    logits = torch.cat(preds)
    loss = masked_bce(logits, target[indices], observed[indices]).item()
    p = logits.sigmoid().numpy()
    return p, metrics(target[indices].numpy(), observed[indices].numpy(), p, loss)


def metrics(target, observed, probability, loss=None):
    from sklearn.metrics import roc_auc_score
    auc = {}
    for j, c in enumerate(TARGETS):
        keep = observed[:, j]
        if len(np.unique(target[keep, j])) == 2:
            auc[c] = float(roc_auc_score(target[keep, j], probability[keep, j]))
    if len(auc) != 12:
        raise ValueError("insufficient validation support")
    return {"macro_auc": float(np.mean(list(auc.values()))), "per_target_auc": auc,
            **({"masked_bce": loss} if loss is not None else {})}


def train(cfg, out, prepared, features, start):
    uids, partitions, values, _, _, _ = prepared
    # Head fitting is CPU float32. No encoder backward or Kaggle GPU is used.
    f, m, g = features
    data = (torch.from_numpy(np.asarray(f).astype(np.float32)), torch.from_numpy(m), torch.from_numpy(g))
    target = torch.from_numpy(np.stack([values[u][0] for u in uids]))
    observed = torch.from_numpy(np.stack([values[u][1] for u in uids]))
    tr = np.array([i for i, u in enumerate(uids) if partitions[u] == "candidate_train"])
    va = np.array([i for i, u in enumerate(uids) if partitions[u] == "candidate_validation"])
    torch.manual_seed(cfg["seed"])
    a = Head(width=cfg["head_width"])
    b = Head(width=cfg["head_width"], geometry=True)
    b.load_state_dict(a.state_dict(), strict=False)  # identical shared initial projection/pool
    heads = {"A": a, "B": b}
    optimizers = {k: torch.optim.AdamW(v.parameters(), lr=cfg["learning_rate"],
                                      weight_decay=cfg["weight_decay"]) for k, v in heads.items()}
    schedulers = {k: torch.optim.lr_scheduler.CosineAnnealingLR(o, cfg["epochs"]) for k, o in optimizers.items()}
    history, best = {k: [] for k in heads}, {k: None for k in heads}
    for epoch in range(1, cfg["epochs"] + 1):
        order = np.random.default_rng(cfg["seed"] + epoch).permutation(tr)
        for arm, model in heads.items():
            torch.manual_seed(cfg["seed"] + epoch)
            model.train()
            losses = []
            for offset in range(0, len(order), cfg["head_batch"]):
                if time.monotonic() - start > cfg["hard_wall_seconds"] - cfg["save_reserve_seconds"]:
                    raise TimeoutError("head training deadline")
                ii = order[offset:offset + cfg["head_batch"]]
                opt = optimizers[arm]
                opt.zero_grad(set_to_none=True)
                logits = model(*(x[ii] for x in data))
                loss = masked_bce(logits, target[ii], observed[ii])
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite head loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                opt.step()
                losses.append(float(loss.detach()))
            schedulers[arm].step()
            p, result = evaluate(model, data, va, target, observed, cfg["head_batch"])
            record = {"epoch": epoch, "train_batch_bce_mean": float(np.mean(losses)), **result}
            history[arm].append(record)
            # Same predeclared selector for both; no 58-study or leaderboard selection.
            key = (-result["macro_auc"], result["masked_bce"], epoch)
            if best[arm] is None or key < best[arm][0]:
                best[arm] = (key, record)
                checkpoint = {"state_dict": model.state_dict(), "config": cfg, "arm": arm,
                              "epoch": epoch, "source_sha256": source_hashes(),
                              "inputs": prepared[-1], "encoder": json.loads((out / "encoder.json").read_text())}
                temp = out / f"f2_{arm}_best.tmp"
                torch.save(checkpoint, temp)
                temp.replace(out / f"f2_{arm}_best.pt")
                np.save(out / f"f2_{arm}_val_probabilities.npy", p, allow_pickle=False)
            write_json(out / "history.json", history)
            write_json(out / "status.json", {"stage": "heads", "arm": arm, **record,
                                              "elapsed_seconds": time.monotonic() - start})
            print(json.dumps({"stage": "heads", "arm": arm, **record}), flush=True)
    write_json(out / "head_selection.json", {k: {"selected": v[1],
               "parameters": sum(p.numel() for p in heads[k].parameters())} for k, v in best.items()})
    return analyze(cfg, out, prepared)


def analyze(cfg, out, prepared):
    """Paired group bootstrap is exploratory after epoch selection on the same fold."""
    uids, partitions, values, _, _, _ = prepared
    va = [u for u in uids if partitions[u] == "candidate_validation"]
    y = np.stack([values[u][0] for u in va])
    obs = np.stack([values[u][1] for u in va])
    pa, pb = [np.load(out / f"f2_{k}_val_probabilities.npy", allow_pickle=False) for k in ("A", "B")]
    with (ROOT / cfg["split"]).open(newline="", encoding="utf-8-sig") as f:
        groups = {r["StudyInstanceUID"]: r["group_id"] for r in csv.DictReader(f)}
    group_ids = sorted({groups[u] for u in va})
    members = [np.array([i for i, u in enumerate(va) if groups[u] == g]) for g in group_ids]
    rng, diffs = np.random.default_rng(cfg["seed"]), []
    for _ in range(cfg["bootstrap_draws"]):
        ii = np.concatenate([members[j] for j in rng.integers(len(members), size=len(members))])
        try:
            diffs.append(metrics(y[ii], obs[ii], pb[ii])["macro_auc"] - metrics(y[ii], obs[ii], pa[ii])["macro_auc"])
        except ValueError:
            continue
    ma, mb = metrics(y, obs, pa), metrics(y, obs, pb)
    result = {"A": ma, "B": mb, "B_minus_A_macro_auc": mb["macro_auc"] - ma["macro_auc"],
              "paired_group_bootstrap_95pct_exploratory": np.percentile(diffs, [2.5, 97.5]).tolist() if diffs else None,
              "bootstrap_valid_draws": len(diffs), "validation_groups": len(members),
              "endpoint": "frozen public report-proxy labels, not expert image labels",
              "limitations": ["same fold used for checkpoint selection; intervals are exploratory",
                              "B changes capacity, sequence context and geometry together",
                              "no complete-F0 independent predictions; no incremental-F0 claim"],
              "kaggle_gpu_hours_used": 0}
    write_json(out / "f2_comparison.json", result)
    print(json.dumps({"stage": "comparison", **result}), flush=True)
    return result


def worker(args, cfg, out):
    start = time.monotonic()
    torch.set_num_threads(cfg["cpu_threads"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.backends.cudnn.benchmark = False
    summary = {"status": "running", "experiment": cfg["experiment"], "config": cfg,
               "source_sha256": source_hashes(), "python": sys.version, "platform": platform.platform(),
               "torch": torch.__version__, "numpy": np.__version__, "kaggle_gpu_hours_used": 0}
    write_json(out / "summary.json", summary)
    try:
        prepared = prepare(cfg)
        write_json(out / "preflight.json", prepared[-1])
        print(json.dumps({"stage": "preflight", **{k: v for k, v in prepared[-1].items() if k != "cache"}}), flush=True)
        if not prepared[-1]["failure_gate_passed"]:
            raise ValueError("selected-series QC gate failed")
        if args.mode == "preflight":
            summary["status"] = "preflight_passed"
        elif args.mode == "analyze":
            summary["comparison"] = analyze(cfg, out, prepared)
            summary["status"] = "complete"
        else:
            features = extract(cfg, out, prepared, start, args.mode == "benchmark")
            if features is None:
                summary["status"] = "benchmark_only"
            else:
                summary["comparison"] = train(cfg, out, prepared, features, start)
                summary["status"] = "complete"
    except Exception as e:
        summary.update(status="failed", error_type=type(e).__name__, error=str(e))
        raise
    finally:
        summary["elapsed_seconds"] = time.monotonic() - start
        write_json(out / "summary.json", summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "benchmark", "run", "analyze"), required=True)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    out = args.out or ROOT / "outputs/f2_dino_geometry" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if args.worker:
        worker(args, cfg, out)
        return
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "config.json", cfg)
    command = [sys.executable, str(Path(__file__).resolve()), "--mode", args.mode,
               "--config", str(args.config.resolve()), "--out", str(out.resolve()), "--worker"]
    env = dict(os.environ, HF_HUB_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1", PYTHONUNBUFFERED="1")
    print(f"F2 output: {out}", flush=True)
    with (out / "stdout.log").open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
        try:
            code = proc.wait(timeout=cfg["hard_wall_seconds"])
        except subprocess.TimeoutExpired:
            proc.kill()  # worker has no subprocesses/DataLoader workers
            proc.wait()
            write_json(out / "supervisor.json", {"status": "hard_timeout", "worker_killed": True})
            raise SystemExit(2)
    write_json(out / "supervisor.json", {"status": "exited", "returncode": code})
    raise SystemExit(code)


if __name__ == "__main__":
    main()
