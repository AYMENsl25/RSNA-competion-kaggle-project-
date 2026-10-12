"""Offline, fail-closed P00 epoch-3 inference. No training labels are opened."""
import csv
import hashlib
import json
import os
import signal
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pydicom
import torch

from m00_data import ROLES, build_study, load_series, select_series
from m00_model import M00, TARGETS

CHECKPOINT_SHA256 = "5dba7e9eceda646edcde3dc37f5371dadf20fb186cd999319d834bb7996656f4"
SOURCE_SHA256 = {
    "m00_data.py": "84190f34e7f4f25518f2f0855104c1c8514bb6e0f4789c1e6e9d12d4803811c5",
    "m00_model.py": "e8410ff5ebcfcf2036c5da18538fc194fe6db24eb88f274981bfba9de3ed531e",
}
COMPETITION_DIRS = (
    Path("/kaggle/input/rsna-knee-abnormality-detection"),
    Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"),
)
CHECKPOINT_DIRS = (
    Path("/kaggle/input/rsna-knee-v2-p00-epoch3-checkpoint"),
    Path("/kaggle/input/datasets/aymenslimani/rsna-knee-v2-p00-epoch3-checkpoint"),
)
OUT = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path(".")
HARD_CAP_S = 9 * 3600
STOP_AT_S = 8.5 * 3600
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
        reader = csv.DictReader(f)
        if reader.fieldnames != ["StudyInstanceUID"]:
            raise ValueError("test.csv schema mismatch")
        test_uids = [r["StudyInstanceUID"] for r in reader]
    if len(test_uids) != len(set(test_uids)) or set(test_uids) != set(uids):
        raise ValueError("test and sample UIDs differ")
    by_study = defaultdict(list)
    fields = ["StudyInstanceUID", "SeriesInstanceUID", "Fluid_Sensitive", "Fat_Suppression", "Anatomical_Plane"]
    with series_path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != fields:
            raise ValueError("test_series schema mismatch")
        seen = set()
        for row in reader:
            uid, sid = row["StudyInstanceUID"], row["SeriesInstanceUID"]
            if uid not in set(uids) or (uid, sid) in seen:
                raise ValueError("test_series unknown or duplicate UID")
            seen.add((uid, sid))
            by_study[uid].append(row)
    if set(by_study) != set(uids):
        raise ValueError("some test studies have no series metadata")
    return cols, uids, by_study


def predict_study(model, x, mask, device):
    """Chunk 48 views to bound VRAM; max pooling equals M00.forward in eval mode."""
    views = x[mask]
    if views.shape[0] == 0 or not torch.isfinite(views).all():
        raise ValueError("no finite valid views")
    pooled = None
    with torch.inference_mode():
        for chunk in views.split(8):
            feat = model.encoder(chunk.to(device)).amax(dim=0)
            pooled = feat if pooled is None else torch.maximum(pooled, feat)
        prob = torch.sigmoid(model.head(pooled[None]))[0].cpu().numpy()
    if prob.shape != (len(TARGETS),) or not np.isfinite(prob).all() or (prob < 0).any() or (prob > 1).any():
        raise ValueError("invalid model probabilities")
    return prob


def self_test():
    model = M00().eval()
    x = torch.zeros(1, len(ROLES), 16, 3, 224, 224)
    mask = torch.zeros(1, len(ROLES), 16, dtype=torch.bool)
    mask[0, 0, :2] = True
    with torch.inference_mode():
        direct = torch.sigmoid(model(x, mask))[0].numpy()
    chunked = predict_study(model, x[0], mask[0], torch.device("cpu"))
    assert np.allclose(direct, chunked, rtol=1e-5, atol=1e-6)
    assert len(chunked) == 12
    return {"status": "passed", "chunked_equal_direct": True}


def main():
    if hasattr(signal, "SIGALRM"):
        signal.signal(signal.SIGALRM, lambda _s, _f: (_ for _ in ()).throw(TimeoutError("nine-hour hard cap")))
        signal.alarm(HARD_CAP_S)
    for filename, digest in SOURCE_SHA256.items():
        if sha256(Path(__file__).with_name(filename)) != digest:
            raise ValueError(f"source hash mismatch: {filename}")
    checkpoint = exactly_one("p00_best.pt", CHECKPOINT_DIRS)
    if sha256(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("P00 checkpoint SHA256 mismatch")
    data = next((p for p in COMPETITION_DIRS if (p / "sample_submission.csv").is_file()), None)
    if data is None:
        raise FileNotFoundError("official competition data mount missing")
    columns, uids, by_study = read_schema(data / "sample_submission.csv", data / "test.csv", data / "test_series.csv")
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if ckpt.get("source_sha256", {}).get("m00_data.py") != SOURCE_SHA256["m00_data.py"] or ckpt.get("source_sha256", {}).get("m00_model.py") != SOURCE_SHA256["m00_model.py"]:
        raise ValueError("checkpoint source hashes differ")
    if ckpt.get("config", {}).get("targets") != TARGETS or ckpt.get("epoch") != 3:
        raise ValueError("checkpoint config/epoch mismatch")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Kaggle GPU accelerator is required")
    model = M00().to(device).eval()
    model.load_state_dict(ckpt["model"], strict=True)
    output_rows = []
    for i, uid in enumerate(uids):
        if time.monotonic() - T0 > STOP_AT_S:
            raise TimeoutError("inference exceeded 8.5-hour graceful limit")
        roles = {}
        for role, sid in select_series(by_study[uid]).items():
            if sid is None:
                continue
            paths = sorted((data / "test_series" / uid / sid).glob("*.dcm"))
            if not paths:
                raise FileNotFoundError("selected test series has no DICOM files")
            roles[role] = load_series([pydicom.dcmread(p) for p in paths])
        x, mask = build_study(roles, train=False)
        prob = predict_study(model, x, mask, device)
        output_rows.append([uid, *[format(float(v), ".9g") for v in prob]])
        if (i + 1) % 25 == 0:
            print(f"processed {i+1}/{len(uids)} studies", flush=True)
    if len(output_rows) != len(uids) or [r[0] for r in output_rows] != uids:
        raise ValueError("submission UID/order mismatch")
    tmp = OUT / "submission.csv.tmp"
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(output_rows)
    os.replace(tmp, OUT / "submission.csv")
    summary = {"status": "complete", "studies": len(uids), "failed_studies": 0,
               "checkpoint_sha256": CHECKPOINT_SHA256, "submission_sha256": sha256(OUT / "submission.csv"),
               "elapsed_s": time.monotonic() - T0, "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved()}
    (OUT / "p00_inference_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    import sys
    if "--self-test" in sys.argv:
        print(json.dumps(self_test()))
    else:
        main()
