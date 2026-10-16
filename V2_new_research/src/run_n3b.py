"""N3b: six-slot sequence-position contextual MIL, matched to N3a.

Run: python -u src/run_n3b.py [--preflight-only] [--resume]
Uses the existing trainer and N1's argument serializer; no new dependencies.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
import platform
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import numpy as np
import torch
import torch.nn.functional as F

import train_v3 as T
from run_n1 import sha, trainer_argv

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/n3b_context"
CONTROL = ROOT / "outputs/n3a_attn6"
SOURCES = ("run_n3b.py", "run_n1.py", "train_v3.py", "v3_model.py", "m00_data.py", "m00_model.py", "p00_model.py", "canon.py")


def save(name: str, data: dict) -> None:
    path = OUT / (name + ".tmp")
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    path.replace(OUT / name)


def config() -> dict:
    control = json.loads((CONTROL / "log.json").read_text())["args"]
    assert control["slots"] == 6 and control["head"] == "attn" and control["epochs"] == 12
    assert control["slab"] == 0 and control["train_windows"] == 4 and control["eval_windows"] == 16
    return control | {"out": "outputs/n3b_context", "slots": 6, "val_bs": 1, "head": "context"}


def preflight(args: dict, identity: dict) -> None:
    path = OUT / "preflight.json"
    if path.exists():
        old = json.loads(path.read_text())
        if old.get("passed") and old.get("identity") == identity:
            print("Reusing matching successful N3b preflight", flush=True)
            return
    assert torch.cuda.is_available()
    split, labels = T.read_split(), T.read_labels("v3")
    cache = [ROOT / p for p in args["cache"]]
    rows = T.read_index(cache)
    studies, primary = T.build_studies(rows, split, 6), T.build_studies(rows, split, 3)
    assert studies.keys() == primary.keys()
    tr = sorted(u for u in studies if split[u][1] == "candidate_train" and T.targets(labels[u], "soft")[1].any())
    va = sorted(u for u in studies if split[u][1] == "candidate_validation")
    assert len(tr) == 3428 and len(va) == 885
    assert not {split[u][0] for u in tr} & {split[u][0] for u in va}
    for u in tr + va:
        assert split[u][1] != "excluded"
        assert all(v in studies[u].values() for v in primary[u].values())
    for d in cache:
        summary = json.loads((d / "summary.json").read_text())
        assert sha(d / "index.csv") == summary["sha256"]["index.csv"]
        vol = np.load(d / "slices.npy", mmap_mode="r")
        assert vol.shape[1:] == (256, 256) and vol.dtype == np.uint8 and vol.nbytes == summary["slices_bytes"]
    # Stress the fullest available studies rather than a lucky low-memory sample.
    selected = sorted(tr, key=lambda u: (-len(studies[u]), u))[:2]
    ds = T.CacheDS(cache, [(u, studies[u]) for u in selected], 4, True, seed=0, mirror_left=True, slots=6, with_positions=True)
    x = torch.stack([ds[(k, 1)][1] for k in range(2)]).cuda()
    mask = torch.stack([ds[(k, 1)][2] for k in range(2)]).cuda()
    positions = torch.stack([ds[(k, 1)][3] for k in range(2)]).cuda()
    assert x.shape == (2, 6, 4, 3, 256, 256)
    assert torch.equal(ds[(0, 1)][1], ds[(0, 1)][1]) and not torch.equal(ds[(0, 1)][1], ds[(0, 2)][1])
    torch.manual_seed(0)
    model = T.make_model(torch.device("cuda"), "context", "convnext_tiny")
    target, observed = zip(*(T.targets(labels[u], "soft") for u in selected))
    target, observed = torch.tensor(np.stack(target), device="cuda"), torch.tensor(np.stack(observed), device="cuda")
    opt = torch.optim.AdamW(model.parameters(), lr=args["lr"], weight_decay=args["wd"])
    scaler = torch.amp.GradScaler("cuda")
    torch.cuda.reset_peak_memory_stats()
    losses, skipped = [], 0
    for _ in range(10):
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(T.prep(x, args["res"]), mask, positions)
        loss = F.binary_cross_entropy_with_logits(logits.float(), target, reduction="none")[observed].mean()
        assert torch.isfinite(loss)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        finite = all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        if finite:
            assert model.att_w.weight.grad.abs().sum() > 0
            assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.parameters())
            losses.append(float(loss.detach()))
        else:
            skipped += 1
        scaler.step(opt)
        scaler.update()
        if len(losses) == 3:
            break
    assert len(losses) == 3, "AMP did not reach three finite encoder/attention updates"
    train_peak = torch.cuda.max_memory_allocated() / 2**20
    chosen_val = max(va, key=lambda u: len(studies[u]))
    ev = T.CacheDS(cache, [(chosen_val, studies[chosen_val])], 16, False, mirror_left=True, slots=6, with_positions=True)
    xv, mv, pv = ev[0][1:]
    model.eval()
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        p = model(T.prep(xv[None].cuda(), args["res"]), mv[None].cuda(), pv[None].cuda())
        assert p.shape == (1, 12) and torch.isfinite(p).all()
    result = {"passed": True, "identity": identity, "train_studies": len(tr), "validation_studies": len(va),
              "excluded_studies_loaded": 0, "original_selected_series_retained": True,
              "slot_count_train": dict(Counter(len(studies[u]) for u in tr)),
              "stress_training_valid_windows": int(mask.sum()), "fp16_losses": losses,
              "amp_startup_skips": skipped, "train_peak_mib": train_peak,
              "validation_peak_mib": torch.cuda.max_memory_allocated() / 2**20,
              "note": "Fresh smoke weights discarded; full run initializes from ImageNet"}
    save("preflight.json", result)
    print("PREFLIGHT PASSED", json.dumps({k: v for k, v in result.items() if k != "identity"}), flush=True)


def comparisons() -> None:
    import csv
    from compare_v3 import main as compare, read_preds
    from scipy.stats import rankdata
    candidate = OUT / "val_pred_epoch12.csv"
    for label in ("public", "v3"):
        compare([str(CONTROL / "val_pred_epoch12.csv"), str(candidate), "--labels", label,
                 "--out", str(OUT / f"vs_n3a_{label}.json")])
    compare([str(ROOT / "outputs/e3_cnxt6/val_pred_epoch12.csv"), str(candidate),
             "--out", str(OUT / "vs_e3_public.json")])
    # Compare one fixed addition to the already-tested Ens5/N1 blend; no weight search.
    base = CONTROL / "base80_n3a20.csv"
    a, b = read_preds(base), read_preds(candidate)
    assert set(a) == set(b)
    uids = sorted(a)
    p = .8 * np.array([a[u] for u in uids]) + .2 * rankdata(np.array([b[u] for u in uids]), axis=0) / len(uids)
    blend = OUT / "base80_n3b20.csv"
    with blend.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["StudyInstanceUID", *T.TARGETS])
        writer.writerows([u, *v] for u, v in zip(uids, p))
    compare([str(base), str(blend), "--out", str(OUT / "ensemble_vs_base_public.json")])


def main() -> None:
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--preflight-only", action="store_true")
    ap.add_argument("--resume", action="store_true")
    options = ap.parse_args()
    os.chdir(ROOT)
    OUT.mkdir(parents=True, exist_ok=True)
    args = config()
    inputs = [T.SPLIT, *T.LABELS.values(), CONTROL / "log.json", CONTROL / "val_pred_epoch12.csv",
              CONTROL / "base80_n3a20.csv", *[ROOT / d / "index.csv" for d in args["cache"]],
              *[ROOT / d / "summary.json" for d in args["cache"]]]
    identity = {"args": args, "source_sha256": {n: sha(ROOT / "src" / n) for n in SOURCES},
                "input_sha256": {str(p.relative_to(ROOT)): sha(p) for p in inputs}}
    if (OUT / "last.pt").exists():
        if not options.resume:
            raise RuntimeError("Run already has a checkpoint; use --resume to preserve prior results")
        if json.loads((OUT / "run_manifest.json").read_text())["identity"] != identity:
            raise RuntimeError("Changed source/input/configuration; refusing unmatched resume")
    preflight(args, identity)
    if options.preflight_only:
        return
    for n in SOURCES:
        path = OUT / "source" / n
        path.parent.mkdir(exist_ok=True)
        path.write_bytes((ROOT / "src" / n).read_bytes())
    revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    save("run_manifest.json", {"identity": identity, "pid": os.getpid(), "git_head": revision,
         "started_istanbul": datetime.now(timezone(timedelta(hours=3))).isoformat(),
         "command": [str(Path(__file__)), *trainer_argv(args)],
         "environment": {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
                         "numpy": np.__version__, "omp_threads": os.environ.get("OMP_NUM_THREADS"),
                         "mkl_threads": os.environ.get("MKL_NUM_THREADS")},
         "control": "N3a epoch 12; E3 epoch 12 secondary",
         "modeling_change": "384 projection, slot and slice-position embeddings, two window-transformer layers, finding-specific pooling",
         "operational_change": "Same training batch 2 and validation batch 1 as N3a",
         "sampling_note": "Same six-slot sampling and RNG draws as N3a; position recording consumes no randomness",
         "secondary_ensemble": "0.8 * N3a fixed blend + 0.2 rank(N3b_epoch12)"})
    save("run_status.json", {"state": "running", "pid": os.getpid()})
    started = time.perf_counter()
    try:
        T.main(trainer_argv(args))
        torch.cuda.empty_cache()
        comparisons()
    except BaseException as exc:
        save("run_status.json", {"state": "failed", "error": f"{type(exc).__name__}: {exc}"})
        raise
    save("run_status.json", {"state": "completed", "epochs": 12, "elapsed_hours": (time.perf_counter() - started) / 3600})


if __name__ == "__main__":
    main()
