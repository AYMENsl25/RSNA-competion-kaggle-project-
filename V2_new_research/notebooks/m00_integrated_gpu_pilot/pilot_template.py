"""Integrated M00 GPU engineering pilot (second and final budgeted GPU probe).

Real DICOM decode through a 4-worker DataLoader + GPU transfer + AMP train/val/infer steps on
100 deterministically chosen NON-expert studies. Targets are SEEDED SYNTHETIC noise; real labels are
never used for any value. Weights/predictions are discarded: no AUC, no submission.csv, no
checkpoint kept. Output JSON is aggregate-only (no UIDs, reports, labels, images)."""
import csv
import hashlib
import json
import os
import platform
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pydicom
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset

# __M00_DATA_SOURCE__
# __M00_MODEL_SOURCE__

ROOTS = (
    Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"),
    Path("/kaggle/input/rsna-knee-abnormality-detection"),
)
OUT = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path(".")
SEED, WORKERS, N_STUDIES = 0, 4, 100
# Disjoint slices of the 100 hash-ordered studies.
N_TRAIN, N_VAL, N_INFER = 70, 20, 10
TRAIN_BS, WARMUP_STEPS = 2, 3
PHASE_CAP_S = {"train": 900.0, "val": 420.0, "infer": 420.0}
TOTAL_BUDGET_S = 2400.0          # hard script cap (45 min limit, 1 GPU-hour)
ASSUMED_TRAIN_STUDIES, ASSUMED_VAL_STUDIES, ASSUMED_EPOCHS, ASSUMED_INFER_STUDIES = 3457, 885, 8, 1300
HEADROOM = 1.5

T0 = time.perf_counter()
summary = {"engineering_only": True, "synthetic_targets_only": True, "completed": False,
           "failures_by_exception": {}, "phases": {}, "estimates": {}}
fail_counter = Counter()


def save():
    summary["failures_by_exception"] = dict(fail_counter)
    summary["script_seconds"] = time.perf_counter() - T0
    (OUT / "integrated_pilot_summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def worker_init(_wid):
    torch.set_num_threads(1)
    s = torch.initial_seed() % 2**32
    np.random.seed(s); random.seed(s)


def h64(text):
    return int(hashlib.sha256(text.encode()).hexdigest()[:16], 16)


def read_rows(data, name):
    with (data / name).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def select_pilot_studies():
    """Same rule as the loader probe: no complete expert labels, no exact report match to any expert
    study; first N_STUDIES by SHA256 of UID. Metadata only; label values are never kept."""
    data = next((p for p in ROOTS if (p / "train.csv").exists()), None)
    if data is None:
        raise FileNotFoundError("Official competition input missing")
    train, series = read_rows(data, "train.csv"), read_rows(data, "train_series.csv")
    targets = [k for k in train[0] if k not in {"StudyInstanceUID", "Report"}]
    norm = lambda r: " ".join(r["Report"].casefold().split())
    expert = {r["StudyInstanceUID"] for r in train if all(r[t] in {"0", "1"} for t in targets)}
    expert_reports = {norm(r) for r in train if r["StudyInstanceUID"] in expert}
    eligible = {r["StudyInstanceUID"] for r in train
                if r["StudyInstanceUID"] not in expert and norm(r) not in expert_reports}
    by_study = {}
    for row in series:
        by_study.setdefault(row["StudyInstanceUID"], []).append(
            {k: row[k] for k in ("SeriesInstanceUID", "Fluid_Sensitive", "Fat_Suppression", "Anatomical_Plane")})
    summary["eligible_nonexpert_studies"] = len(eligible)
    del train
    return data, sorted(eligible, key=lambda u: hashlib.sha256(u.encode()).hexdigest())[:N_STUDIES], by_study


class StudyDS(Dataset):
    """Returns a dict per study; failures return the exception TYPE name (never silently replaced)."""

    def __init__(self, data, uids, by_study, train):
        self.data, self.uids, self.by_study, self.train = data, uids, by_study, train

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, i):
        uid, t0 = self.uids[i], time.perf_counter()
        try:
            roles = {}
            for role, sid in select_series(self.by_study.get(uid, [])).items():
                if sid is None:
                    continue
                paths = sorted((self.data / "train_series" / uid / sid).glob("*.dcm"))
                roles[role] = load_series([pydicom.dcmread(p) for p in paths])
            rng = np.random.default_rng(h64(uid)) if self.train else None
            x, mask = build_study(roles, train=self.train, rng=rng)
            g = np.random.default_rng(h64("synthetic-target:" + uid))
            tgt = torch.from_numpy(g.integers(0, 2, 12)).float()
            obs = torch.from_numpy(g.random(12) < 0.7)
            return {"ok": True, "x": x, "mask": mask, "t": tgt, "obs": obs, "sec": time.perf_counter() - t0}
        except Exception as exc:
            return {"ok": False, "err": type(exc).__name__, "sec": time.perf_counter() - t0}


def collate(items):
    good = [d for d in items if d["ok"]]
    out = {"errs": [d["err"] for d in items if not d["ok"]], "decode_sec": [d["sec"] for d in items], "n": len(items)}
    if good:
        out.update(x=torch.stack([d["x"] for d in good]), mask=torch.stack([d["mask"] for d in good]),
                   t=torch.stack([d["t"] for d in good]), obs=torch.stack([d["obs"] for d in good]))
        out["valid_windows"] = [int(d["mask"].sum()) for d in good]
    return out


def pct(values, q):
    s = sorted(values)
    return s[int(q * (len(s) - 1))] if s else None


def run_phase(name, data, uids, by_study, model, opt, scaler, dev, bs, train_windows, mode):
    """mode: train | val | infer. Per-batch timings with CUDA sync; hard wall cap per phase."""
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    loader = DataLoader(StudyDS(data, uids, by_study, train_windows), batch_size=bs, shuffle=False,
                        num_workers=WORKERS, collate_fn=collate, pin_memory=True, prefetch_factor=2,
                        worker_init_fn=worker_init, multiprocessing_context="spawn", timeout=300,
                        generator=torch.Generator().manual_seed(SEED))
    waits, h2ds, steps, step_studies, valid, decode = [], [], [], [], [], []
    seen = ok_studies = failed = skipped_unobserved = nonfinite = steps_done = 0
    phase_fail, truncated = Counter(), False
    t_phase = time.perf_counter()
    it = iter(loader)
    try:
        while True:
            t = time.perf_counter()
            try:
                b = next(it)
            except StopIteration:
                break
            wait = time.perf_counter() - t
            seen += b["n"]; decode += b["decode_sec"]
            for e in b["errs"]:
                phase_fail[e] += 1; fail_counter[f"{name}:{e}"] += 1
            failed += len(b["errs"])
            if "x" in b:
                ok_studies += len(b["valid_windows"]); valid += b["valid_windows"]
                if not b["obs"].any():           # all-unobserved synthetic batch: skip + log, never train
                    skipped_unobserved += 1
                else:
                    try:
                        torch.cuda.synchronize(); t = time.perf_counter()
                        x = b["x"].to(dev, non_blocking=True); mask = b["mask"].to(dev, non_blocking=True)
                        tgt = b["t"].to(dev, non_blocking=True); obs = b["obs"].to(dev, non_blocking=True)
                        torch.cuda.synchronize(); h2d = time.perf_counter() - t
                        t = time.perf_counter()
                        if mode == "train":
                            model.train(); opt.zero_grad(set_to_none=True)
                            with torch.autocast("cuda", dtype=torch.float16):
                                logits = model(x, mask)
                            loss = masked_bce(logits.float(), tgt, obs)
                            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
                            nonfinite += int(not torch.isfinite(loss))
                        else:
                            model.eval()
                            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                                if mode == "val":
                                    loss = masked_bce(model(x, mask).float(), tgt, obs)
                                    nonfinite += int(not torch.isfinite(loss))
                                else:
                                    nonfinite += int(not torch.isfinite(predict_proba(model, x, mask)).all())
                        torch.cuda.synchronize(); step = time.perf_counter() - t
                        steps_done += 1
                        waits.append(wait); h2ds.append(h2d); steps.append(step); step_studies.append(len(b["valid_windows"]))
                    except Exception as exc:     # counted, not replaced; OOM ends the phase
                        n = len(b["valid_windows"]); failed += n; ok_studies -= n
                        fail_counter[f"{name}:{type(exc).__name__}"] += n; phase_fail[type(exc).__name__] += n
                        if isinstance(exc, torch.cuda.OutOfMemoryError):
                            truncated = True; break
            if time.perf_counter() - t_phase > PHASE_CAP_S[name] or time.perf_counter() - T0 > TOTAL_BUDGET_S:
                truncated = True; break
    finally:
        del it, loader
    wall = time.perf_counter() - t_phase
    steady = slice(WARMUP_STEPS if mode == "train" else 0, None)
    sw, sh, ss, sn = waits[steady], h2ds[steady], steps[steady], step_studies[steady]
    steady_total = sum(sw) + sum(sh) + sum(ss)
    rate = lambda n, s: n * 3600 / s if s > 0 else None
    summary["phases"][name] = {
        "mode": mode, "batch_size": bs, "workers": WORKERS, "windows_per_role": 4 if train_windows else 16,
        "studies_requested": len(uids), "studies_seen": seen, "studies_ok": ok_studies, "studies_failed": failed,
        "failed_by_exception": dict(phase_fail), "batches_skipped_all_unobserved": skipped_unobserved,
        "nonfinite_loss_or_prob": nonfinite, "gpu_steps": steps_done, "truncated_by_budget": truncated,
        "wall_seconds": wall, "studies_per_hour_wall_incl_startup": rate(ok_studies, wall),
        "studies_per_hour_steady": rate(sum(sn), steady_total),
        "studies_per_hour_gpu_only": rate(sum(sn), sum(ss)),
        "step_seconds_median": pct(ss, 0.5), "step_seconds_p95": pct(ss, 0.95),
        "h2d_seconds_median": pct(sh, 0.5), "data_wait_seconds_median": pct(sw, 0.5),
        "data_wait_seconds_p95": pct(sw, 0.95),
        "data_wait_fraction_steady": sum(sw) / steady_total if steady_total else None,
        "worker_decode_seconds_per_study_median": pct(decode, 0.5), "worker_decode_seconds_per_study_p95": pct(decode, 0.95),
        "valid_windows_total": sum(valid), "valid_windows_per_study_min": min(valid, default=None),
        "valid_windows_per_study_median": pct(valid, 0.5), "valid_windows_per_study_max": max(valid, default=None),
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20}
    save()


def time_checkpoint(model, opt):
    path = OUT / "_pilot_ckpt_tmp.pt"
    try:
        t = time.perf_counter()
        torch.save({"model": model.state_dict(), "opt": opt.state_dict()}, path)
        sec = time.perf_counter() - t
        return {"serialize_seconds": sec, "size_mib": path.stat().st_size / 2**20}
    finally:
        path.unlink(missing_ok=True)


def estimate():
    tr, va, inf = (summary["phases"].get(k) for k in ("train", "val", "infer"))
    est = {"provisional_from_sample": True, "headroom_factor": HEADROOM,
           "assumptions": [
               f"{ASSUMED_TRAIN_STUDIES} train and {ASSUMED_VAL_STUDIES} validation studies/epoch, {ASSUMED_EPOCHS} epochs, {ASSUMED_INFER_STUDIES} inference studies (frozen split counts and approximate hidden test count)",
               "steady-state rate from this sample incl. DICOM decode every epoch (no cache); decode+GPU overlap as measured",
               "cold-disk first reads included; later epochs may be faster if page cache holds, so this errs high",
               "full-run total includes one validation pass and checkpoint write each epoch, but excludes model-load/setup and unexpected retries",
               "100-study sample may be unrepresentative of series size/count; same GPU type assumed"]}
    if tr and tr["studies_per_hour_steady"]:
        sec = ASSUMED_EPOCHS * ASSUMED_TRAIN_STUDIES * 3600 / tr["studies_per_hour_steady"]
        est["train_8_epoch_seconds_raw"], est["train_8_epoch_hours_with_headroom"] = sec, sec * HEADROOM / 3600
        if tr["studies_per_hour_gpu_only"]:
            est["train_8_epoch_hours_gpu_only_lower_bound"] = ASSUMED_EPOCHS * ASSUMED_TRAIN_STUDIES / tr["studies_per_hour_gpu_only"]
    if va and va["studies_per_hour_steady"]:
        est["val_seconds_per_1000_studies"] = 1000 * 3600 / va["studies_per_hour_steady"]
    if tr and va and tr["studies_per_hour_steady"] and va["studies_per_hour_steady"]:
        checkpoint_s = summary.get("checkpoint", {}).get("serialize_seconds", 0.0)
        train_s = ASSUMED_TRAIN_STUDIES * 3600 / tr["studies_per_hour_steady"]
        val_s = ASSUMED_VAL_STUDIES * 3600 / va["studies_per_hour_steady"]
        total_s = ASSUMED_EPOCHS * (train_s + val_s + checkpoint_s)
        est["m00_full_8_epoch_hours_raw_incl_val_checkpoint"] = total_s / 3600
        est["m00_full_8_epoch_hours_with_headroom"] = total_s * HEADROOM / 3600
        est["within_8_gpu_hour_cap_estimate"] = total_s * HEADROOM <= 8 * 3600
    if inf and inf["studies_per_hour_steady"]:
        sec = ASSUMED_INFER_STUDIES * 3600 / inf["studies_per_hour_steady"]
        est["infer_1300_seconds_raw"], est["infer_1300_minutes_with_headroom"] = sec, sec * HEADROOM / 60
    summary["estimates"] = est


def main():
    if not torch.cuda.is_available():
        fail_counter["setup:CudaUnavailable"] += 1
        save(); raise SystemExit("CUDA not available: enable GPU")
    seed_all(SEED)
    props = torch.cuda.get_device_properties(0)
    summary["environment"] = {
        "gpu": props.name, "vram_mib": props.total_memory / 2**20, "gpu_count": torch.cuda.device_count(),
        "capability": list(torch.cuda.get_device_capability(0)), "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(), "torch": torch.__version__,
        "torchvision": torchvision.__version__, "pydicom": pydicom.__version__, "numpy": np.__version__,
        "python": sys.version.split()[0], "cpu_count": os.cpu_count(), "workers": WORKERS}
    summary["config"] = {"train_windows_per_role": TRAIN_WINDOWS, "infer_windows_per_role": INFER_WINDOWS,
                         "roles": len(ROLES), "train_bs": TRAIN_BS, "amp": "fp16 autocast + GradScaler",
                         "split": [N_TRAIN, N_VAL, N_INFER]}
    data, uids, by_study = select_pilot_studies()
    summary["pilot_studies_selected"] = len(uids)
    tr_u, va_u, in_u = uids[:N_TRAIN], uids[N_TRAIN:N_TRAIN + N_VAL], uids[N_TRAIN + N_VAL:N_TRAIN + N_VAL + N_INFER]
    dev = torch.device("cuda")
    model = M00().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    run_phase("train", data, tr_u, by_study, model, opt, scaler, dev, TRAIN_BS, True, "train")
    summary["checkpoint"] = time_checkpoint(model, opt)
    run_phase("val", data, va_u, by_study, model, opt, scaler, dev, 1, False, "val")
    run_phase("infer", data, in_u, by_study, model, opt, scaler, dev, 1, False, "infer")
    estimate()
    p = summary["phases"]
    summary["requirements_met"] = {
        "train_steps_ge_30": p["train"]["gpu_steps"] >= 30,
        "val_studies_ok_eq_20": p["val"]["studies_ok"] == N_VAL,
        "infer_studies_ok_ge_10": p["infer"]["studies_ok"] >= N_INFER,
        "no_failures": not fail_counter}
    summary["completed"] = all(summary["requirements_met"].values())
    save()


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except BaseException as exc:                 # type name only: no paths/UIDs in output
        fail_counter[f"fatal:{type(exc).__name__}"] += 1
        save(); raise
