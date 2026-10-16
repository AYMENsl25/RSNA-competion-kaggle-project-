"""E01 six-epoch offline submission inference. No training labels are opened.

Preprocessing is exactly E01 validation: m00_data select_series / IPP-ordered load_series /
build_study(train=False) -> 16 deterministic windows per plane, per-series 1-99 percentile [0,1]
scaling, no ImageNet normalization. Model call is model(x, mask) under fp16 autocast with batch 1,
as in E01 validate(). Runs on CPU (fp32) for the preflight on the visible test studies.

Failure policy (every case counted in the summary, never silent):
  1. whole study loads -> normal prediction;
  2. a series fails -> retry dropping only the failing plane(s) ("degraded");
  3. still nothing -> column mean of the successful test predictions ("fallback"; never 0.5).
"""
import csv
import hashlib
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pydicom
import torch
from torch.utils.data import DataLoader, Dataset

from m00_data import ROLES, build_study, load_series, select_series
from m00_model import M00, TARGETS

CHECKPOINT_NAME = "e01_best.pt"
CHECKPOINT_SHA256 = "6b357912031969c267d670df1db304a0f554bb83ea31a6247ea1aef1dd062a60"
SOURCE_SHA256 = {
    "m00_data.py": "84190f34e7f4f25518f2f0855104c1c8514bb6e0f4789c1e6e9d12d4803811c5",
    "m00_model.py": "e8410ff5ebcfcf2036c5da18538fc194fe6db24eb88f274981bfba9de3ed531e",
}
COMPETITION_DIRS = (
    Path("/kaggle/input/rsna-knee-abnormality-detection"),
    Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"),
)
CHECKPOINT_DIRS = (
    Path("/kaggle/input/rsna-knee-v2-e01-six-epoch-checkpoint"),
    Path("/kaggle/input/datasets/aymenslimani/rsna-knee-v2-e01-six-epoch-checkpoint"),
)
OUT = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path(".")
STOP_AT_S = 8 * 3600  # remaining studies get the counted fallback instead of a failed submission
WORKERS = 4
T0 = time.monotonic()


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def exactly_one(name, directories):
    hits = {p.resolve() for d in directories if d.is_dir() for p in d.rglob(name)}
    if len(hits) != 1:
        raise FileNotFoundError(f"expected one {name}, found {len(hits)}")
    return hits.pop()


def read_schema(sample_path, test_path, series_path):
    with sample_path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
        cols = list(rows[0]) if rows else []
    if cols != ["StudyInstanceUID", *TARGETS] or not rows:
        raise ValueError("sample submission schema mismatch or empty")
    uids = [r["StudyInstanceUID"] for r in rows]
    if len(uids) != len(set(uids)):
        raise ValueError("duplicate sample UID")
    with test_path.open(newline="", encoding="utf-8-sig") as f:
        test_uids = [r["StudyInstanceUID"] for r in csv.DictReader(f)]
    if set(test_uids) != set(uids):
        raise ValueError("test and sample UIDs differ")
    by_study = defaultdict(list)
    with series_path.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            by_study[row["StudyInstanceUID"]].append(row)
    return cols, uids, by_study


class TestDS(Dataset):
    def __init__(self, data, uids, by_study):
        self.data, self.uids, self.by_study = data, uids, by_study

    def __len__(self):
        return len(self.uids)

    def load_role(self, uid, sid, syntaxes):
        paths = sorted((self.data / "test_series" / uid / sid).glob("*.dcm"))
        if not paths:
            raise FileNotFoundError("selected series has no DICOM files")
        datasets = [pydicom.dcmread(p) for p in paths]
        syntaxes[str(getattr(datasets[0].file_meta, "TransferSyntaxUID", "?"))] += 1
        return load_series(datasets)

    def __getitem__(self, i):
        uid = self.uids[i]
        syntaxes, errors = Counter(), []
        selected = {r: s for r, s in select_series(self.by_study.get(uid, [])).items() if s is not None}
        try:  # identical to E01 validation: any series error fails the whole study
            roles = {r: self.load_role(uid, s, syntaxes) for r, s in selected.items()}
            x, mask = build_study(roles, train=False)
            return {"i": i, "state": "ok", "x": x, "mask": mask, "syntaxes": syntaxes, "errors": errors}
        except Exception as exc:
            errors.append(type(exc).__name__)
        roles = {}
        for r, s in selected.items():  # degraded retry: drop only the failing plane(s)
            try:
                roles[r] = self.load_role(uid, s, Counter())
            except Exception as exc:
                errors.append(f"{r}:{type(exc).__name__}")
        try:
            x, mask = build_study(roles, train=False)
            return {"i": i, "state": "degraded", "x": x, "mask": mask, "syntaxes": syntaxes, "errors": errors}
        except Exception as exc:
            errors.append(type(exc).__name__)
            return {"i": i, "state": "failed", "syntaxes": syntaxes, "errors": errors,
                    "n_selected_series": len(selected)}


def main():
    for filename, digest in SOURCE_SHA256.items():
        if sha256(Path(__file__).with_name(filename)) != digest:
            raise ValueError(f"source hash mismatch: {filename}")
    checkpoint = exactly_one(CHECKPOINT_NAME, CHECKPOINT_DIRS)
    if sha256(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("E01 checkpoint SHA256 mismatch")
    data = next((p for p in COMPETITION_DIRS if (p / "sample_submission.csv").is_file()), None)
    if data is None:
        raise FileNotFoundError("official competition data mount missing")
    columns, uids, by_study = read_schema(data / "sample_submission.csv", data / "test.csv", data / "test_series.csv")
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    for name, digest in SOURCE_SHA256.items():
        if ckpt.get("source_sha256", {}).get(name) != digest:
            raise ValueError(f"checkpoint was trained with a different {name}")
    if ckpt.get("config", {}).get("targets") != TARGETS or ckpt.get("epoch") != 6:
        raise ValueError("checkpoint config/epoch mismatch")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} torch={torch.__version__} pydicom={pydicom.__version__} studies={len(uids)}", flush=True)
    model = M00().to(device).eval()  # P00 == M00 architecture; encoder weights come from the checkpoint
    model.load_state_dict(ckpt["model"], strict=True)

    loader = DataLoader(TestDS(data, uids, by_study), batch_size=None, shuffle=False,
                        num_workers=WORKERS, multiprocessing_context="fork" if sys.platform == "linux" else None)
    probs = np.full((len(uids), len(TARGETS)), np.nan)
    states, syntaxes, errors, windows = Counter(), Counter(), Counter(), []
    timed_out = 0
    for n, item in enumerate(loader, 1):
        if time.monotonic() - T0 > STOP_AT_S:
            timed_out = len(uids) - n + 1
            print(f"time limit reached; {timed_out} studies left for fallback", flush=True)
            break
        states[item["state"]] += 1
        syntaxes.update(item["syntaxes"])
        errors.update(item["errors"])
        if item["state"] == "failed":
            print(f"study {item['i']} FAILED: {item['errors']} (selected series {item['n_selected_series']})", flush=True)
            continue
        if item["state"] == "degraded":
            print(f"study {item['i']} degraded: {item['errors']}", flush=True)
        x, mask = item["x"][None].to(device), item["mask"][None].to(device)
        windows.append(int(mask.sum()))
        with torch.inference_mode(), torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(x, mask).float()
        p = torch.sigmoid(logits)[0].cpu().numpy()
        if p.shape != (len(TARGETS),) or not np.isfinite(p).all():
            raise FloatingPointError("nonfinite or misshaped probabilities")
        probs[item["i"]] = p
        if n % 100 == 0:
            print(f"processed {n}/{len(uids)} in {time.monotonic() - T0:.0f}s", flush=True)

    good = np.isfinite(probs).all(1)
    if not good.any():
        raise RuntimeError("no test study could be predicted")
    fallback = probs[good].mean(0)
    n_fallback = int((~good).sum())
    probs[~good] = fallback
    tmp = OUT / "submission.csv.tmp"
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(columns)
        for uid, p in zip(uids, probs):
            w.writerow([uid, *[format(float(v), ".9g") for v in p]])
    os.replace(tmp, OUT / "submission.csv")
    summary = {"status": "complete", "device": str(device), "studies": len(uids),
               "studies_ok": states["ok"], "studies_degraded": states["degraded"],
               "studies_failed": states["failed"], "studies_timed_out": timed_out,
               "studies_mean_fallback": n_fallback, "fallback_vector": fallback.round(6).tolist(),
               "errors": dict(errors), "transfer_syntaxes": dict(syntaxes),
               "valid_windows_min_median_max": [min(windows), float(np.median(windows)), max(windows)] if windows else None,
               "checkpoint_sha256": CHECKPOINT_SHA256, "submission_sha256": sha256(OUT / "submission.csv"),
               "elapsed_s": round(time.monotonic() - T0, 1),
               "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved() if device.type == "cuda" else 0}
    (OUT / "e01_inference_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
