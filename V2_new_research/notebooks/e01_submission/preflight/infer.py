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


EMBEDDED_SOURCES = {'m00_data.py': '"""M00 data path: series selection, DICOM ordering/scaling, window sampling, study tensor.\n\nOffline, deterministic, no flips. Failures raise SeriesError (series-level) or StudyError\n(study-level) so callers can count them; nothing is silently substituted.\n"""\nimport numpy as np\nimport torch\nimport torch.nn.functional as F\n\nROLES = ("Sagittal", "Coronal", "Axial")\nTRAIN_WINDOWS, INFER_WINDOWS, SIZE = 4, 16, 224\nTOL_MM = 1e-3\n\n\nclass SeriesError(ValueError):\n    pass\n\n\nclass StudyError(ValueError):\n    pass\n\n\ndef select_series(rows):\n    """rows: iterable of dicts with SeriesInstanceUID, Fluid_Sensitive, Fat_Suppression,\n    Anatomical_Plane (official train_series.csv fields) for ONE study.\n    Returns {role: SeriesInstanceUID or None}. Deterministic.\n    ponytail: preference fluid-sensitive > fat-suppressed > smallest UID is MY choice, not a\n    clinical rule; revisit after the A05 role study."""\n    best = {r: None for r in ROLES}\n    for row in rows:\n        role = str(row["Anatomical_Plane"]).strip().capitalize()\n        if role not in best:\n            continue\n        key = (-int(row["Fluid_Sensitive"] or 0), -int(row["Fat_Suppression"] or 0),\n               str(row["SeriesInstanceUID"]))\n        if best[role] is None or key < best[role][0]:\n            best[role] = (key, str(row["SeriesInstanceUID"]))\n    return {r: (v[1] if v else None) for r, v in best.items()}\n\n\ndef slice_position(ds):\n    """Projection of ImagePositionPatient on row x col normal; None if tags missing."""\n    iop, ipp = getattr(ds, "ImageOrientationPatient", None), getattr(ds, "ImagePositionPatient", None)\n    if iop is None or ipp is None:\n        return None\n    iop, ipp = np.asarray(iop, float), np.asarray(ipp, float)\n    if iop.shape != (6,) or ipp.shape != (3,) or not (np.isfinite(iop).all() and np.isfinite(ipp).all()):\n        raise SeriesError("malformed orientation/position tags")\n    r, c = iop[:3], iop[3:]\n    if abs(np.linalg.norm(r) - 1) > 1e-2 or abs(np.linalg.norm(c) - 1) > 1e-2 or abs(r @ c) > 1e-2:\n        raise SeriesError("ImageOrientationPatient not orthonormal")\n    return float(ipp @ np.cross(r, c)), tuple(np.round(np.cross(r, c), 3))\n\n\ndef order_slices(datasets):\n    """Sort datasets by physical position. Fallback to InstanceNumber ONLY if every slice lacks\n    orientation/position tags; mixed presence, duplicates, irregular gaps or in-plane changes raise."""\n    if not datasets:\n        raise SeriesError("empty series")\n    spacings = []\n    for ds in datasets:\n        raw = getattr(ds, "PixelSpacing", None)\n        if raw is None:\n            raise SeriesError("missing PixelSpacing")\n        spacing = np.asarray(raw, dtype=float)\n        if spacing.shape != (2,) or not np.isfinite(spacing).all() or np.any(spacing <= 0):\n            raise SeriesError("invalid PixelSpacing")\n        spacings.append(spacing)\n    if any(np.max(np.abs(spacing - spacings[0])) > 1e-3 for spacing in spacings[1:]):\n        raise SeriesError("pixel spacing differs within series")\n    pos = [slice_position(d) for d in datasets]\n    if all(p is None for p in pos):\n        nums = [getattr(d, "InstanceNumber", None) for d in datasets]\n        if any(n is None for n in nums) or len(set(int(n) for n in nums)) != len(nums):\n            raise SeriesError("no geometry tags and InstanceNumber missing/duplicated")\n        keys = [float(n) for n in nums]\n    elif any(p is None for p in pos):\n        raise SeriesError("geometry tags present on only some slices")\n    else:\n        if len({p[1] for p in pos}) != 1:\n            raise SeriesError("slice normal differs within series")\n        orientation = np.asarray(datasets[0].ImageOrientationPatient, dtype=float)\n        if any(np.max(np.abs(np.asarray(ds.ImageOrientationPatient, dtype=float) - orientation)) > 1e-3\n               for ds in datasets[1:]):\n            raise SeriesError("in-plane orientation differs within series")\n        keys = [p[0] for p in pos]\n    order = np.argsort(keys, kind="stable")\n    s = np.asarray(keys)[order]\n    if len(s) > 1 and np.diff(s).min() < TOL_MM:\n        raise SeriesError("duplicate slice positions")\n    if len(s) > 2:\n        gaps = np.diff(s)\n        if gaps.max() > 1.5 * np.median(gaps) + 0.01:\n            raise SeriesError("irregular slice gap")\n    return [datasets[i] for i in order]\n\n\ndef slice_pixels(ds):\n    """float32 pixels after rescale and MONOCHROME1 inversion (negation; scaling is affine)."""\n    a = ds.pixel_array.astype(np.float32)\n    if a.ndim != 2:\n        raise SeriesError("non-2D pixel array")\n    a = a * float(getattr(ds, "RescaleSlope", 1.0)) + float(getattr(ds, "RescaleIntercept", 0.0))\n    if str(getattr(ds, "PhotometricInterpretation", "MONOCHROME2")) == "MONOCHROME1":\n        a = -a\n    return a\n\n\ndef load_series(datasets, lo_q=1.0, hi_q=99.0):\n    """-> (volume [N,H,W] float32 in [0,1], slice_ok [N] bool). Slices with nonfinite values or\n    constant content are flagged not-ok (zero-filled, never used in a window)."""\n    ds = order_slices(datasets)\n    arrs = [slice_pixels(d) for d in ds]\n    if len({a.shape for a in arrs}) != 1:\n        raise SeriesError("inconsistent slice shapes")\n    ok = np.array([np.isfinite(a).all() and a.max() > a.min() for a in arrs])\n    if ok.sum() < 3:\n        raise SeriesError("fewer than 3 usable slices")\n    # Bound percentile scratch memory for large series while sampling each slice\n    # deterministically. Full decoded slices are still used for final scaling.\n    vals = np.concatenate([\n        a.ravel()[::max(1, a.size // 16384)] for a, k in zip(arrs, ok) if k\n    ])\n    lo, hi = np.percentile(vals, [lo_q, hi_q])\n    if not hi > lo:\n        raise SeriesError("degenerate intensity range")\n    vol = np.stack([np.clip((a - lo) / (hi - lo), 0, 1) if k else np.zeros_like(a)\n                    for a, k in zip(arrs, ok)]).astype(np.float32)\n    return vol, ok\n\n\ndef window_centers(ok, n, rng=None):\n    """Centers c such that slices c-1,c,c+1 are all usable (physically adjacent in sorted order).\n    rng None -> deterministic evenly spaced (inference); else random without replacement (train)."""\n    valid = np.flatnonzero(ok[:-2] & ok[1:-1] & ok[2:]) + 1\n    if len(valid) == 0:\n        return valid\n    if rng is None:\n        return np.unique(valid[np.round(np.linspace(0, len(valid) - 1, n)).astype(int)])\n    return np.sort(rng.choice(valid, size=min(n, len(valid)), replace=False))\n\n\ndef series_windows(vol, ok, n, rng=None):\n    """-> (x [n,3,224,224], mask [n]); unused rows zero with mask False."""\n    x = torch.zeros(n, 3, SIZE, SIZE)\n    m = torch.zeros(n, dtype=torch.bool)\n    for i, c in enumerate(window_centers(ok, n, rng)):\n        w = torch.from_numpy(vol[c - 1:c + 2])[None]\n        x[i] = F.interpolate(w, size=(SIZE, SIZE), mode="bilinear", align_corners=False)[0]\n        m[i] = True\n    return x, m\n\n\ndef build_study(role_series, train, rng=None):\n    """role_series: {role: (vol, ok) or None}. -> (x [3,W,3,224,224], mask [3,W]).\n    Raises StudyError if no role yields a valid window."""\n    n = TRAIN_WINDOWS if train else INFER_WINDOWS\n    xs, ms = [], []\n    for r in ROLES:\n        s = role_series.get(r)\n        x, m = series_windows(*s, n, rng if train else None) if s else (\n            torch.zeros(n, 3, SIZE, SIZE), torch.zeros(n, dtype=torch.bool))\n        xs.append(x), ms.append(m)\n    mask = torch.stack(ms)\n    if not mask.any():\n        raise StudyError("study has no valid window in any role")\n    return torch.stack(xs), mask\n', 'm00_model.py': '"""M00: random-init ResNet18 shared encoder, shared max over valid windows (all roles jointly),\none linear head -> 12 logits. No pretrained weights, no network access."""\nimport torch\nimport torch.nn as nn\nimport torch.nn.functional as F\nfrom torchvision.models import resnet18\n\nTARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",\n           "PF OA", "Effusion", "Synovitis", "Baker\'s", "Contusion", "Fracture"]\n\n\nclass M00(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.encoder = resnet18(weights=None)\n        self.encoder.fc = nn.Identity()\n        self.head = nn.Linear(512, len(TARGETS))\n\n    def forward(self, x, mask):\n        """x [B,R,W,3,H,W2] float, mask [B,R,W] bool -> logits [B,12]. Only valid windows are\n        encoded, so padded windows cannot influence features, BatchNorm stats or the max."""\n        if x.ndim != 6 or x.shape[3] != 3 or mask.dtype != torch.bool or mask.shape != x.shape[:3]:\n            raise ValueError(f"bad shapes x={tuple(x.shape)} mask={tuple(mask.shape)} {mask.dtype}")\n        has = mask.flatten(1).any(1)\n        if not has.all():\n            raise ValueError(f"studies with no valid window (batch idx {(~has).nonzero().flatten().tolist()})")\n        sel = x[mask]\n        if not torch.isfinite(sel).all():\n            raise ValueError("nonfinite pixels in a valid window")\n        feats = self.encoder(sel)                                   # [n,512]\n        b = mask.flatten(1).nonzero()[:, 0]                         # study index per valid window\n        pooled = feats.new_full((x.shape[0], feats.shape[1]), float("-inf"))\n        pooled = pooled.scatter_reduce(0, b[:, None].expand_as(feats), feats, "amax", include_self=True)\n        return self.head(pooled)\n\n\n@torch.no_grad()\ndef predict_proba(model, x, mask):\n    """Finite [B,12] probabilities in [0,1] or a clear error."""\n    model.eval()\n    p = torch.sigmoid(model(x, mask))\n    if not torch.isfinite(p).all():\n        raise FloatingPointError("nonfinite probabilities")\n    return p\n\n\ndef masked_bce(logits, targets, observed):\n    """Mean BCE over observed cells only. observed bool [B,12]; unobserved targets ignored."""\n    if logits.shape != targets.shape or logits.shape != observed.shape or observed.dtype != torch.bool:\n        raise ValueError("logits, targets and boolean observed mask must have identical shapes")\n    if not observed.any():\n        raise ValueError("no observed targets in batch")\n    selected_logits, selected_targets = logits[observed], targets[observed].float()\n    if not torch.isfinite(selected_logits).all() or not torch.isfinite(selected_targets).all():\n        raise ValueError("nonfinite observed logits or targets")\n    return F.binary_cross_entropy_with_logits(selected_logits, selected_targets)\n'}
exec(EMBEDDED_SOURCES['m00_data.py'], globals())
exec(EMBEDDED_SOURCES['m00_model.py'], globals())

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
        if hashlib.sha256(EMBEDDED_SOURCES[filename].encode('utf-8')).hexdigest() != digest:
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
