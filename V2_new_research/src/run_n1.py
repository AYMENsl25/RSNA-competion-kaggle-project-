"""Run N1 against the exact E8 recipe, with a real-cache CUDA preflight.

python src/run_n1.py --preflight-only
python -u src/run_n1.py              # reuses a matching successful preflight
python -u src/run_n1.py --resume     # explicit, identical-configuration resume
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Use only the ImageNet weights already cached for E8; never fetch weights here.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import numpy as np
import torch
import torch.nn.functional as F

import train_v3 as T

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/n1_attn_random"
SOURCES = ("run_n1.py", "train_v3.py", "v3_model.py", "m00_data.py", "m00_model.py", "p00_model.py", "canon.py")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(name: str, data: dict) -> None:
    tmp = OUT / (name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    tmp.replace(OUT / name)


def config(workers: int) -> dict:
    args = json.loads((ROOT / "outputs/e8_cnxt/log.json").read_text())["args"]
    assert args["head"] == "max" and args["epochs"] == 12
    assert args["train_windows"] == 4 and args["eval_windows"] == 16 and not args["slab"]
    return args | {"out": "outputs/n1_attn_random", "head": "attn", "workers": workers,
                   "prefetch": 1, "log_every": 100}


def trainer_argv(args: dict) -> list[str]:
    argv = []
    for key, value in args.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                argv.append(flag)
        elif value is not None:
            argv.extend([flag, *map(str, value)] if isinstance(value, list) else [flag, str(value)])
    return argv


def preflight(args: dict, identity: dict) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("N1 requires the laptop CUDA GPU")
    if (OUT / "preflight.json").exists():
        old = json.loads((OUT / "preflight.json").read_text())
        if old.get("identity") == identity and old.get("passed"):
            print("Reusing successful preflight with identical configuration, sources and inputs", flush=True)
            return old
    split = T.read_split()
    studies = T.build_studies(T.read_index([ROOT / p for p in args["cache"]]), split, 3)
    labels = T.read_labels("v3")
    train = sorted(u for u in studies if split[u][1] == "candidate_train" and T.targets(labels[u], "soft")[1].any())
    val = sorted(u for u in studies if split[u][1] == "candidate_validation")
    assert len(train) == 3428 and len(val) == 885
    assert not {split[u][0] for u in train} & {split[u][0] for u in val}
    assert not any(split[u][1] == "excluded" for u in train + val)
    for p in args["cache"]:
        path = ROOT / p
        summary = json.loads((path / "summary.json").read_text())
        assert sha(path / "index.csv") == summary["sha256"]["index.csv"]
        vol = np.load(path / "slices.npy", mmap_mode="r")
        assert vol.dtype == np.uint8 and vol.shape[1:] == (256, 256)
        assert vol.nbytes == summary["slices_bytes"]
    ds = T.CacheDS([ROOT / p for p in args["cache"]], [(u, studies[u]) for u in train[:2]],
                   4, True, seed=0, mirror_left=True)
    assert torch.equal(ds[(0, 1)][1], ds[(0, 1)][1])
    assert not torch.equal(ds[(0, 1)][1], ds[(0, 2)][1]), "random window augmentation was lost"

    # The shared encoder AND classifier start identically; only attention is new.
    torch.manual_seed(0)
    control = T.make_model(torch.device("cpu"), "max", "convnext_tiny")
    torch.manual_seed(0)
    model = T.make_model(torch.device("cpu"), "attn", "convnext_tiny")
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in control.state_dict().items())
    del control
    model = model.cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=.05)
    scaler = torch.amp.GradScaler("cuda")
    x = torch.stack([ds[(k, 1)][1] for k in range(2)]).cuda()
    mask = torch.stack([ds[(k, 1)][2] for k in range(2)]).cuda()
    target, observed = zip(*(T.targets(labels[u], "soft") for u in train[:2]))
    target = torch.from_numpy(np.stack(target)).cuda()
    observed = torch.from_numpy(np.stack(observed)).cuda()
    torch.cuda.reset_peak_memory_stats()
    losses = []
    skipped = 0
    for _ in range(10):
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(T.prep(x, 224), mask)
        loss = F.binary_cross_entropy_with_logits(logits.float(), target, reduction="none")[observed].mean()
        assert torch.isfinite(loss)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        finite = all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        if finite:
            assert model.att_w.weight.grad.abs().sum() > 0
            assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.parameters())
        else:
            skipped += 1  # GradScaler skips overflowed startup steps and lowers its scale.
        scaler.step(opt)
        scaler.update()
        if finite:
            losses.append(float(loss.detach()))
        if len(losses) == 3:
            break
    assert len(losses) == 3, "fp16 did not reach three finite encoder+attention updates"
    train_peak = torch.cuda.max_memory_allocated() / 2**20
    model.eval()
    ev = T.CacheDS([ROOT / p for p in args["cache"]], [(u, studies[u]) for u in val[:2]],
                   16, False, mirror_left=True)
    xv = torch.stack([ev[k][1] for k in range(2)]).cuda()
    mv = torch.stack([ev[k][2] for k in range(2)]).cuda()
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        pv = model(T.prep(xv, 224), mv)
        assert pv.shape == (2, 12) and torch.isfinite(pv).all()
        # Padding must be invisible to the attention head, even with NaN pixels.
        masked = mask.clone()
        masked[:, 0, -1] = False
        clean = T.prep(x, 224)
        expected = model(clean, masked)
        corrupted = clean.clone()
        corrupted[~masked] = float("nan")
        assert torch.allclose(model(corrupted, masked), expected, atol=1e-4)
    result = {"passed": True, "identity": identity, "train_studies": len(train), "val_studies": len(val),
              "expert_or_excluded_studies": 0, "train_windows_random": True,
              "encoder_and_classifier_initialization_match_E8": True, "fp16_losses": losses,
              "amp_startup_skips": skipped,
              "train_peak_mib": train_peak, "validation_peak_mib": torch.cuda.max_memory_allocated() / 2**20,
              "gpu": torch.cuda.get_device_name(), "note": "Smoke model discarded; full training starts fresh"}
    write_json("preflight.json", result)
    print("PREFLIGHT PASSED", json.dumps({k: v for k, v in result.items() if k != "identity"}), flush=True)
    return result


def compare_finished_run() -> None:
    from compare_v3 import main as compare, read_preds
    from scipy.stats import rankdata
    candidate = OUT / "val_pred_epoch12.csv"
    control = ROOT / "outputs/e8_cnxt/val_pred_epoch12.csv"
    for labels in ("public", "v3"):
        compare([str(control), str(candidate), "--labels", labels, "--out", str(OUT / f"vs_e8_{labels}.json")])
    base = ROOT / "outputs/ens5.csv"
    a, b = read_preds(base), read_preds(candidate)
    assert set(a) == set(b)
    uids = sorted(a)
    p = .8 * np.array([a[u] for u in uids]) + .2 * rankdata(np.array([b[u] for u in uids]), axis=0) / len(uids)
    path = OUT / "ens5_80_n1_20.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["StudyInstanceUID", *T.TARGETS])
        w.writerows([u, *v] for u, v in zip(uids, p))
    compare([str(base), str(path), "--out", str(OUT / "ensemble_vs_ens5_public.json")])


def main() -> None:
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--preflight-only", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--workers", type=int, default=1)
    options = ap.parse_args()
    if options.workers < 0:
        ap.error("--workers must be nonnegative")
    os.chdir(ROOT)
    OUT.mkdir(parents=True, exist_ok=True)
    args = config(options.workers)
    identity = {"args": args, "source_sha256": {n: sha(ROOT / "src" / n) for n in SOURCES},
                "input_sha256": {str(p.relative_to(ROOT)): sha(p) for p in
                                 [T.SPLIT, *T.LABELS.values(), ROOT / "outputs/e8_cnxt/log.json"]}}
    previous = OUT / "run_manifest.json"
    if (OUT / "last.pt").exists():
        if not options.resume:
            raise RuntimeError("Existing checkpoint: use --resume; refusing to overwrite this run")
        if not previous.exists() or json.loads(previous.read_text())["identity"] != identity:
            raise RuntimeError("Resume configuration/sources/inputs changed; refusing an unmatched continuation")
    preflight(args, identity)
    if options.preflight_only:
        return
    for name in SOURCES:
        snap = OUT / "source" / name
        snap.parent.mkdir(exist_ok=True)
        snap.write_bytes((ROOT / "src" / name).read_bytes())
    rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    write_json("run_manifest.json", {"identity": identity, "control": "E8 epoch 12",
               "started_istanbul": datetime.now(timezone(timedelta(hours=3))).isoformat(),
               "pid": os.getpid(), "git_head": rev.stdout.strip() or None,
               "environment": {"python": platform.python_version(), "torch": torch.__version__,
                               "cuda": torch.version.cuda, "numpy": np.__version__},
               "command": [str(Path(__file__)), *trainer_argv(args)],
               "decision": "12 matched epochs; one new attention head; no slab; no gold evaluation",
               "operational_changes": "1 worker per loader, prefetch 1, progress/thermal check each 100 steps",
               "secondary_ensemble": "0.80 Ens5 + 0.20 N1 per-finding percentile ranks; diagnostic only"})
    write_json("run_status.json", {"state": "running", "pid": os.getpid()})
    started = time.perf_counter()
    try:
        T.main(trainer_argv(args))
        torch.cuda.empty_cache()
        compare_finished_run()
    except BaseException as exc:
        write_json("run_status.json", {"state": "failed", "error": f"{type(exc).__name__}: {exc}"})
        raise
    write_json("run_status.json", {"state": "completed", "epochs": 12,
                                   "elapsed_hours": (time.perf_counter() - started) / 3600})


if __name__ == "__main__":
    main()
