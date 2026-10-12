"""P02 full training: at most three complete epochs on frozen V2 dev split + public report labels.

Private, offline Kaggle GPU script. Inputs: official competition data (DICOM + train_series.csv
only; train.csv is never read), public `stevenleehans/rsna-knee-llm-report-labels` v1
`llm_labels_full.csv`, private `aymenslimani/rsna-knee-v2-frozen-dev-split` `split.csv`.
Labels are thresholded report proxies (p>0.5 -> 1, p<0.5 -> 0, p==0.5 -> masked); validation AUC
measures weak-label mimicry, NOT expert image-label accuracy. No expert check, no submission.
Outputs (Kaggle working dir only): best model state + config + source hashes, per-epoch validation
probabilities, aggregate JSON (no UIDs, reports or labels)."""
import csv
import hashlib
import json
import math
import os
import platform
import random
import signal
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pydicom
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset

SOURCE_SHA256 = {'m00_data.py': '84190f34e7f4f25518f2f0855104c1c8514bb6e0f4789c1e6e9d12d4803811c5', 'm00_model.py': 'e8410ff5ebcfcf2036c5da18538fc194fe6db24eb88f274981bfba9de3ed531e', 'p02_model.py': '5f0f078cd27d66fcecff92df18583d3eb6457332fa3547de8651c34d626bf223', 'p00_full_train.py': '63371847c937fedbe299f2bdd3487101d53a082d8917ae3bdddb88558ccaaae5', 'p00_model.py': '1cadda51f0df11d2ced1d420475b1d8f870e11c50e4714689f7640a392b189c2', 'train_template.py': '4b94722d6436326fc29e218a29d8722f4ef50e9e349a97fb819da639a9f465c9'}
# Embedded m00_data.py SHA256: 84190f34e7f4f25518f2f0855104c1c8514bb6e0f4789c1e6e9d12d4803811c5
"""M00 data path: series selection, DICOM ordering/scaling, window sampling, study tensor.

Offline, deterministic, no flips. Failures raise SeriesError (series-level) or StudyError
(study-level) so callers can count them; nothing is silently substituted.
"""
import numpy as np
import torch
import torch.nn.functional as F

ROLES = ("Sagittal", "Coronal", "Axial")
TRAIN_WINDOWS, INFER_WINDOWS, SIZE = 4, 16, 224
TOL_MM = 1e-3


class SeriesError(ValueError):
    pass


class StudyError(ValueError):
    pass


def select_series(rows):
    """rows: iterable of dicts with SeriesInstanceUID, Fluid_Sensitive, Fat_Suppression,
    Anatomical_Plane (official train_series.csv fields) for ONE study.
    Returns {role: SeriesInstanceUID or None}. Deterministic.
    ponytail: preference fluid-sensitive > fat-suppressed > smallest UID is MY choice, not a
    clinical rule; revisit after the A05 role study."""
    best = {r: None for r in ROLES}
    for row in rows:
        role = str(row["Anatomical_Plane"]).strip().capitalize()
        if role not in best:
            continue
        key = (-int(row["Fluid_Sensitive"] or 0), -int(row["Fat_Suppression"] or 0),
               str(row["SeriesInstanceUID"]))
        if best[role] is None or key < best[role][0]:
            best[role] = (key, str(row["SeriesInstanceUID"]))
    return {r: (v[1] if v else None) for r, v in best.items()}


def slice_position(ds):
    """Projection of ImagePositionPatient on row x col normal; None if tags missing."""
    iop, ipp = getattr(ds, "ImageOrientationPatient", None), getattr(ds, "ImagePositionPatient", None)
    if iop is None or ipp is None:
        return None
    iop, ipp = np.asarray(iop, float), np.asarray(ipp, float)
    if iop.shape != (6,) or ipp.shape != (3,) or not (np.isfinite(iop).all() and np.isfinite(ipp).all()):
        raise SeriesError("malformed orientation/position tags")
    r, c = iop[:3], iop[3:]
    if abs(np.linalg.norm(r) - 1) > 1e-2 or abs(np.linalg.norm(c) - 1) > 1e-2 or abs(r @ c) > 1e-2:
        raise SeriesError("ImageOrientationPatient not orthonormal")
    return float(ipp @ np.cross(r, c)), tuple(np.round(np.cross(r, c), 3))


def order_slices(datasets):
    """Sort datasets by physical position. Fallback to InstanceNumber ONLY if every slice lacks
    orientation/position tags; mixed presence, duplicates, irregular gaps or in-plane changes raise."""
    if not datasets:
        raise SeriesError("empty series")
    spacings = []
    for ds in datasets:
        raw = getattr(ds, "PixelSpacing", None)
        if raw is None:
            raise SeriesError("missing PixelSpacing")
        spacing = np.asarray(raw, dtype=float)
        if spacing.shape != (2,) or not np.isfinite(spacing).all() or np.any(spacing <= 0):
            raise SeriesError("invalid PixelSpacing")
        spacings.append(spacing)
    if any(np.max(np.abs(spacing - spacings[0])) > 1e-3 for spacing in spacings[1:]):
        raise SeriesError("pixel spacing differs within series")
    pos = [slice_position(d) for d in datasets]
    if all(p is None for p in pos):
        nums = [getattr(d, "InstanceNumber", None) for d in datasets]
        if any(n is None for n in nums) or len(set(int(n) for n in nums)) != len(nums):
            raise SeriesError("no geometry tags and InstanceNumber missing/duplicated")
        keys = [float(n) for n in nums]
    elif any(p is None for p in pos):
        raise SeriesError("geometry tags present on only some slices")
    else:
        if len({p[1] for p in pos}) != 1:
            raise SeriesError("slice normal differs within series")
        orientation = np.asarray(datasets[0].ImageOrientationPatient, dtype=float)
        if any(np.max(np.abs(np.asarray(ds.ImageOrientationPatient, dtype=float) - orientation)) > 1e-3
               for ds in datasets[1:]):
            raise SeriesError("in-plane orientation differs within series")
        keys = [p[0] for p in pos]
    order = np.argsort(keys, kind="stable")
    s = np.asarray(keys)[order]
    if len(s) > 1 and np.diff(s).min() < TOL_MM:
        raise SeriesError("duplicate slice positions")
    if len(s) > 2:
        gaps = np.diff(s)
        if gaps.max() > 1.5 * np.median(gaps) + 0.01:
            raise SeriesError("irregular slice gap")
    return [datasets[i] for i in order]


def slice_pixels(ds):
    """float32 pixels after rescale and MONOCHROME1 inversion (negation; scaling is affine)."""
    a = ds.pixel_array.astype(np.float32)
    if a.ndim != 2:
        raise SeriesError("non-2D pixel array")
    a = a * float(getattr(ds, "RescaleSlope", 1.0)) + float(getattr(ds, "RescaleIntercept", 0.0))
    if str(getattr(ds, "PhotometricInterpretation", "MONOCHROME2")) == "MONOCHROME1":
        a = -a
    return a


def load_series(datasets, lo_q=1.0, hi_q=99.0):
    """-> (volume [N,H,W] float32 in [0,1], slice_ok [N] bool). Slices with nonfinite values or
    constant content are flagged not-ok (zero-filled, never used in a window)."""
    ds = order_slices(datasets)
    arrs = [slice_pixels(d) for d in ds]
    if len({a.shape for a in arrs}) != 1:
        raise SeriesError("inconsistent slice shapes")
    ok = np.array([np.isfinite(a).all() and a.max() > a.min() for a in arrs])
    if ok.sum() < 3:
        raise SeriesError("fewer than 3 usable slices")
    # Bound percentile scratch memory for large series while sampling each slice
    # deterministically. Full decoded slices are still used for final scaling.
    vals = np.concatenate([
        a.ravel()[::max(1, a.size // 16384)] for a, k in zip(arrs, ok) if k
    ])
    lo, hi = np.percentile(vals, [lo_q, hi_q])
    if not hi > lo:
        raise SeriesError("degenerate intensity range")
    vol = np.stack([np.clip((a - lo) / (hi - lo), 0, 1) if k else np.zeros_like(a)
                    for a, k in zip(arrs, ok)]).astype(np.float32)
    return vol, ok


def window_centers(ok, n, rng=None):
    """Centers c such that slices c-1,c,c+1 are all usable (physically adjacent in sorted order).
    rng None -> deterministic evenly spaced (inference); else random without replacement (train)."""
    valid = np.flatnonzero(ok[:-2] & ok[1:-1] & ok[2:]) + 1
    if len(valid) == 0:
        return valid
    if rng is None:
        return np.unique(valid[np.round(np.linspace(0, len(valid) - 1, n)).astype(int)])
    return np.sort(rng.choice(valid, size=min(n, len(valid)), replace=False))


def series_windows(vol, ok, n, rng=None):
    """-> (x [n,3,224,224], mask [n]); unused rows zero with mask False."""
    x = torch.zeros(n, 3, SIZE, SIZE)
    m = torch.zeros(n, dtype=torch.bool)
    for i, c in enumerate(window_centers(ok, n, rng)):
        w = torch.from_numpy(vol[c - 1:c + 2])[None]
        x[i] = F.interpolate(w, size=(SIZE, SIZE), mode="bilinear", align_corners=False)[0]
        m[i] = True
    return x, m


def build_study(role_series, train, rng=None):
    """role_series: {role: (vol, ok) or None}. -> (x [3,W,3,224,224], mask [3,W]).
    Raises StudyError if no role yields a valid window."""
    n = TRAIN_WINDOWS if train else INFER_WINDOWS
    xs, ms = [], []
    for r in ROLES:
        s = role_series.get(r)
        x, m = series_windows(*s, n, rng if train else None) if s else (
            torch.zeros(n, 3, SIZE, SIZE), torch.zeros(n, dtype=torch.bool))
        xs.append(x), ms.append(m)
    mask = torch.stack(ms)
    if not mask.any():
        raise StudyError("study has no valid window in any role")
    return torch.stack(xs), mask

# Embedded m00_model.py SHA256: e8410ff5ebcfcf2036c5da18538fc194fe6db24eb88f274981bfba9de3ed531e
"""M00: random-init ResNet18 shared encoder, shared max over valid windows (all roles jointly),
one linear head -> 12 logits. No pretrained weights, no network access."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
           "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]


class M00(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = resnet18(weights=None)
        self.encoder.fc = nn.Identity()
        self.head = nn.Linear(512, len(TARGETS))

    def forward(self, x, mask):
        """x [B,R,W,3,H,W2] float, mask [B,R,W] bool -> logits [B,12]. Only valid windows are
        encoded, so padded windows cannot influence features, BatchNorm stats or the max."""
        if x.ndim != 6 or x.shape[3] != 3 or mask.dtype != torch.bool or mask.shape != x.shape[:3]:
            raise ValueError(f"bad shapes x={tuple(x.shape)} mask={tuple(mask.shape)} {mask.dtype}")
        has = mask.flatten(1).any(1)
        if not has.all():
            raise ValueError(f"studies with no valid window (batch idx {(~has).nonzero().flatten().tolist()})")
        sel = x[mask]
        if not torch.isfinite(sel).all():
            raise ValueError("nonfinite pixels in a valid window")
        feats = self.encoder(sel)                                   # [n,512]
        b = mask.flatten(1).nonzero()[:, 0]                         # study index per valid window
        pooled = feats.new_full((x.shape[0], feats.shape[1]), float("-inf"))
        pooled = pooled.scatter_reduce(0, b[:, None].expand_as(feats), feats, "amax", include_self=True)
        return self.head(pooled)


@torch.no_grad()
def predict_proba(model, x, mask):
    """Finite [B,12] probabilities in [0,1] or a clear error."""
    model.eval()
    p = torch.sigmoid(model(x, mask))
    if not torch.isfinite(p).all():
        raise FloatingPointError("nonfinite probabilities")
    return p


def masked_bce(logits, targets, observed):
    """Mean BCE over observed cells only. observed bool [B,12]; unobserved targets ignored."""
    if logits.shape != targets.shape or logits.shape != observed.shape or observed.dtype != torch.bool:
        raise ValueError("logits, targets and boolean observed mask must have identical shapes")
    if not observed.any():
        raise ValueError("no observed targets in batch")
    selected_logits, selected_targets = logits[observed], targets[observed].float()
    if not torch.isfinite(selected_logits).all() or not torch.isfinite(selected_targets).all():
        raise ValueError("nonfinite observed logits or targets")
    return F.binary_cross_entropy_with_logits(selected_logits, selected_targets)

# Embedded p00_model.py SHA256: 1cadda51f0df11d2ced1d420475b1d8f870e11c50e4714689f7640a392b189c2
"""P00: M00 architecture with strictly verified ImageNet-1K ResNet18 initialization."""
from pathlib import Path
import hashlib
import torch
import torch.nn as nn

from torchvision.models import resnet18

WEIGHT_FILE = "resnet18-f37072fd.pth"
WEIGHT_SHA256 = "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
WEIGHT_DIRS = (
    Path("/kaggle/input/rsna-v2-official-resnet18-imagenet-v1"),
    Path("/kaggle/input/datasets/aymenslimani/rsna-v2-official-resnet18-imagenet-v1"),
)


def verified_pretrained_encoder():
    hits = {p.resolve() for d in WEIGHT_DIRS if d.is_dir() for p in d.rglob(WEIGHT_FILE)}
    if len(hits) != 1:
        raise FileNotFoundError(f"expected exactly one {WEIGHT_FILE}, got {len(hits)}")
    path = hits.pop()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    if digest.hexdigest() != WEIGHT_SHA256:
        raise ValueError("pretrained ResNet18 weight SHA256 mismatch")
    # The source file is torchvision ResNet18_Weights.IMAGENET1K_V1.
    weights = torch.load(path, map_location="cpu", weights_only=True)
    encoder = resnet18(weights=None)
    encoder.load_state_dict(weights, strict=True)
    encoder.fc = nn.Identity()
    return encoder


class P00(M00):
    def __init__(self):
        super().__init__()
        self.encoder = verified_pretrained_encoder()


# Embedded p02_model.py SHA256: 5f0f078cd27d66fcecff92df18583d3eb6457332fa3547de8651c34d626bf223
"""P02: P00 encoder and head with target-specific max over valid MRI windows."""

import torch



class P02(P00):
    def forward(self, x, mask):
        if x.ndim != 6 or x.shape[3] != 3 or mask.dtype != torch.bool or mask.shape != x.shape[:3]:
            raise ValueError(f"bad shapes x={tuple(x.shape)} mask={tuple(mask.shape)} {mask.dtype}")
        if not mask.flatten(1).any(1).all():
            raise ValueError("study with no valid window")
        selected = x[mask]
        if not torch.isfinite(selected).all():
            raise ValueError("nonfinite pixels in a valid window")
        logits = self.head(self.encoder(selected))
        study = mask.flatten(1).nonzero()[:, 0]
        pooled = logits.new_full((x.shape[0], logits.shape[1]), float("-inf"))
        return pooled.scatter_reduce(0, study[:, None].expand_as(logits), logits, "amax", include_self=True)


COMPETITION_ROOTS = (
    Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"),
    Path("/kaggle/input/rsna-knee-abnormality-detection"),
)
LABEL_DIRS = (
    Path("/kaggle/input/rsna-knee-llm-report-labels"),
    Path("/kaggle/input/datasets/stevenleehans/rsna-knee-llm-report-labels"),
)
SPLIT_DIRS = (
    Path("/kaggle/input/rsna-knee-v2-frozen-dev-split"),
    Path("/kaggle/input/datasets/aymenslimani/rsna-knee-v2-frozen-dev-split"),
)
LABEL_FILE, LABEL_SHA256 = "llm_labels_full.csv", "96444cd3c7d396bae4a71454317d9cbe0689759b5278b30939781e04d0774537"
SPLIT_FILE, SPLIT_SHA256 = "split.csv", "dbe879ab5cb719e09c0f9aec612d5b6b53648d6f32dd8d405a075482339b3819"
SPLIT_COLUMNS = ["StudyInstanceUID", "group_id", "partition"]
LABEL_COLUMNS = ["StudyInstanceUID", *TARGETS]
EXPECTED_COUNTS = {"candidate_train": 3457, "candidate_validation": 885, "excluded": 65}
EXPECTED_TRAIN_ALL_MASKED, EXPECTED_TRAIN_OBSERVED_CELLS, MIN_VAL_SUPPORT = 29, 31088, 20

OUT = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path(".")
SEED, WORKERS, TRAIN_BS, VAL_BS, MAX_EPOCHS, LR = 0, 4, 2, 1, 3, 3e-4
HARD_CAP_S = 8 * 3600.0                  # approved GPU-hour cap, measured from script start
SAVE_RESERVE_S = 30 * 60.0               # left for final checkpoint/predictions/JSON
STOP_AT_S = HARD_CAP_S - SAVE_RESERVE_S  # graceful stop deadline
PROJECTION_HEADROOM = 1.2                # next epoch must fit: slowest measured epoch x 1.2
MAX_AMP_SKIP_FRACTION = 0.01
WATCHDOG_CODE = '''import os, signal, sys, time
parent = int(sys.argv[1]); deadline = float(sys.argv[2])
def identity(pid):
    try:
        fields = open('/proc/%d/stat' % pid).read().rsplit(')', 1)[1].split()
        return int(fields[1]), fields[19]
    except (OSError, IndexError, ValueError):
        return None
def tree():
    children = {}
    for name in os.listdir('/proc'):
        if name.isdigit():
            state = identity(int(name))
            if state and int(name) != os.getpid(): children.setdefault(state[0], []).append((int(name), state[1]))
    found = []; pending = [parent]
    while pending:
        for pid, start in children.get(pending.pop(), []):
            found.append((pid, start)); pending.append(pid)
    return found
def alive(pid, start):
    state = identity(pid)
    return state is not None and state[1] == start
def send(pid, sig):
    try: os.kill(pid, sig)
    except (ProcessLookupError, OSError): pass
parent_state = identity(parent)
if parent_state is None: sys.exit(0)
while time.monotonic() < deadline - 60:
    if os.getppid() != parent: sys.exit(0)
    time.sleep(min(1, max(0, deadline - 60 - time.monotonic())))
if os.getppid() != parent: sys.exit(0)
tracked = tree()
send(parent, signal.SIGTERM)
time.sleep(max(0, deadline - time.monotonic()))
all_children = {(pid, start) for pid, start in tracked + tree()}
for pid, start in all_children:
    if alive(pid, start): send(pid, signal.SIGKILL)
if alive(parent, parent_state[1]): send(parent, signal.SIGKILL)
'''
CONFIG = {
    "model": "P02 ImageNet-pretrained ResNet18 2.5D, per-finding max of valid window logits",
    "pretrained_source": "torchvision ResNet18_Weights.IMAGENET1K_V1",
    "pretrained_weight_file": WEIGHT_FILE, "pretrained_weight_sha256": WEIGHT_SHA256,
    "roles": list(ROLES), "train_windows_total": len(ROLES) * TRAIN_WINDOWS,
    "infer_windows_total": len(ROLES) * INFER_WINDOWS, "image_size": SIZE, "targets": TARGETS,
    "max_epochs": MAX_EPOCHS, "train_bs": TRAIN_BS, "val_bs": VAL_BS, "workers": WORKERS,
    "amp": "fp16 autocast + GradScaler", "loss": "masked BCE", "optimizer": "AdamW (default wd 0.01)",
    "lr": LR, "schedule": "cosine per optimizer call over planned steps, eta_min 0", "seed": SEED,
    "hflip": False, "cache": False, "image_quantization": False,
    "label_rule": "p>0.5 -> 1, p<0.5 -> 0, p==0.5 -> masked; finite [0,1] required",
    "label_sha256": LABEL_SHA256, "split_sha256": SPLIT_SHA256,
    "selection": "max 12-target macro ROC-AUC, then lower masked val BCE, then earlier epoch",
    "hard_cap_s": HARD_CAP_S, "stop_at_s": STOP_AT_S, "projection_headroom": PROJECTION_HEADROOM,
    "max_amp_skip_fraction": MAX_AMP_SKIP_FRACTION, "external_watchdog": "SIGTERM 60s before 8h, SIGKILL at 8h"}

T0 = time.perf_counter()
summary = {"status": "incomplete", "completed": False, "stop_reason": None, "proxy_labels_only": True,
           "config": CONFIG, "source_sha256": SOURCE_SHA256, "epochs": [], "best": None}


def elapsed():
    return time.perf_counter() - T0


def save():
    summary["script_seconds"] = elapsed()
    tmp = OUT / "p02_full_train_summary.json.tmp"
    tmp.write_text(json.dumps(summary, indent=2) + "\n")
    os.replace(tmp, OUT / "p02_full_train_summary.json")


def deadline_handler(_signal, _frame):
    summary["status"], summary["completed"] = "incomplete", False
    summary["stop_reason"] = "external_watchdog_deadline"
    save()
    raise SystemExit(1)


def start_watchdog():
    deadline = time.monotonic() + max(0, HARD_CAP_S - elapsed())
    return subprocess.Popen([sys.executable, "-c", WATCHDOG_CODE, str(os.getpid()), str(deadline)],
                            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def worker_init(_wid):
    torch.set_num_threads(1)
    s = torch.initial_seed() % 2**32
    np.random.seed(s); random.seed(s)


def h64(text):
    return int(hashlib.sha256(text.encode()).hexdigest()[:16], 16)


def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_one(name, dirs):
    """Exactly one file `name` under the fixed candidate mount dirs (deduped by resolved path)."""
    hits = {p.resolve() for d in dirs if d.is_dir() for p in d.rglob(name)}
    if len(hits) != 1:
        raise FileNotFoundError(f"expected exactly one {name} in attached dataset, found {len(hits)}")
    return hits.pop()


def hard_labels(p):
    """Report probabilities [..., 12] -> (targets float32, observed bool). Strict finite [0,1]."""
    p = np.asarray(p, dtype=np.float64)
    if p.shape[-1] != len(TARGETS) or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("label probabilities must be finite, in [0,1], 12 per study")
    return (p > 0.5).astype(np.float32), p != 0.5


def roc_auc(y, s):
    """Tie-aware ROC-AUC (Mann-Whitney with average ranks); None if a class is absent."""
    y, s = np.asarray(y, dtype=bool), np.asarray(s, dtype=np.float64)
    npos, nneg = int(y.sum()), int((~y).sum())
    if npos == 0 or nneg == 0:
        return None
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    ranks = (np.cumsum(cnt) - (cnt - 1) / 2.0)[inv]
    return float((ranks[y].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def evaluate(probs, tgt, obs):
    auc, support = {}, {}
    for k, name in enumerate(TARGETS):
        m = obs[:, k]
        auc[name] = roc_auc(tgt[m, k] > 0.5, probs[m, k])
        support[name] = {"positive": int((tgt[m, k] > 0.5).sum()), "negative": int((tgt[m, k] < 0.5).sum())}
    if any(v is None for v in auc.values()):
        raise ValueError("a target lacks positive or negative validation support")
    return auc, support, float(np.mean(list(auc.values())))


def selection_key(macro_auc, bce, epoch):
    return (-macro_auc, bce, epoch)


def self_check():
    """Synthetic checks for label masking, AUC ties and checkpoint selection. No data access."""
    t, o = hard_labels([[0.0, 0.3, 0.5, 0.5000001, 0.7, 1.0] * 2])
    assert t.tolist() == [[0, 0, 0, 1, 1, 1] * 2] and o.tolist() == [[True, True, False, True, True, True] * 2]
    for bad in (float("nan"), float("inf"), -0.1, 1.1):
        try:
            hard_labels([[bad] + [0.2] * 11]); raise AssertionError("bad label accepted")
        except ValueError:
            pass
    assert roc_auc([0, 1], [0.5, 0.5]) == 0.5
    assert roc_auc([0, 0, 1, 1], [0.1, 0.4, 0.4, 0.8]) == 0.875
    assert roc_auc([0, 1, 1], [0.2, 0.9, 0.3]) == 1.0 and roc_auc([1, 0], [0.2, 0.9]) == 0.0
    assert roc_auc([1, 1], [0.1, 0.2]) is None
    keys = [selection_key(0.70, 0.50, 1), selection_key(0.70, 0.40, 3), selection_key(0.60, 0.10, 2)]
    assert min(keys)[2] == 3 and min(selection_key(0.7, 0.4, e) for e in (2, 1, 3))[2] == 1
    return True


def read_series(data):
    """train_series.csv -> ({uid: [series rows]}, universe). Only the four selection fields kept."""
    by_study = {}
    with (data / "train_series.csv").open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            by_study.setdefault(row["StudyInstanceUID"], []).append(
                {k: row[k] for k in ("SeriesInstanceUID", "Fluid_Sensitive", "Fat_Suppression", "Anatomical_Plane")})
    return by_study, set(by_study)


def read_split(path, universe):
    parts, seen, group_part = {k: [] for k in EXPECTED_COUNTS}, set(), {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        if next(reader) != SPLIT_COLUMNS:
            raise ValueError("split schema mismatch")
        for row in reader:
            if len(row) != 3 or row[2] not in parts:
                raise ValueError("malformed split row")
            uid, gid, part = row
            if uid in seen:
                raise ValueError("duplicate split UID (partition overlap)")
            # Groups are grouped-split components: one group must never span partitions.
            if group_part.setdefault(gid, part) != part:
                raise ValueError("split group spans partitions")
            seen.add(uid); parts[part].append(uid)
    if seen != universe:
        raise ValueError("split UID universe differs from official train_series.csv")
    counts = {k: len(v) for k, v in parts.items()}
    if counts != EXPECTED_COUNTS:
        raise ValueError(f"split counts {counts} != {EXPECTED_COUNTS}")
    return {k: sorted(v) for k, v in parts.items()}


def read_labels(path, universe, wanted):
    """Hard/masked labels for `wanted` UIDs only; other rows are schema/UID-checked, values unparsed."""
    labels, seen = {}, set()
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        if next(reader) != LABEL_COLUMNS:
            raise ValueError("public label schema mismatch")
        for row in reader:
            if len(row) != len(LABEL_COLUMNS):
                raise ValueError("malformed label row")
            uid = row[0]
            if uid in seen or uid not in universe:
                raise ValueError("duplicate or unknown label UID")
            seen.add(uid)
            if uid in wanted:
                labels[uid] = hard_labels([float(v) for v in row[1:]])
    if len(labels) != len(wanted):
        raise ValueError(f"{len(wanted) - len(labels)} train/validation studies lack public labels")
    return labels


class StudyDS(Dataset):
    """Per-study tensors; failures return the exception TYPE name (never silently replaced)."""

    def __init__(self, data, uids, by_study, train, epoch=0):
        self.data, self.uids, self.by_study, self.train, self.epoch = data, uids, by_study, train, epoch

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, i):
        uid = self.uids[i]
        try:
            roles = {}
            for role, sid in select_series(self.by_study.get(uid, [])).items():
                if sid is None:
                    continue
                paths = sorted((self.data / "train_series" / uid / sid).glob("*.dcm"))
                roles[role] = load_series([pydicom.dcmread(p) for p in paths])
            rng = np.random.default_rng(h64(f"train:{SEED}:{self.epoch}:{uid}")) if self.train else None
            x, mask = build_study(roles, train=self.train, rng=rng)
            return {"i": i, "ok": True, "x": x, "mask": mask}
        except Exception as exc:
            return {"i": i, "ok": False, "err": type(exc).__name__}


def collate(items):
    good = [d for d in items if d["ok"]]
    out = {"idx": [d["i"] for d in good], "errs": [d["err"] for d in items if not d["ok"]], "n": len(items)}
    if good:
        out["x"], out["mask"] = torch.stack([d["x"] for d in good]), torch.stack([d["mask"] for d in good])
    return out


def make_loader(ds, bs, shuffle, seed):
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=WORKERS, collate_fn=collate,
                      pin_memory=True, prefetch_factor=2, worker_init_fn=worker_init,
                      multiprocessing_context="fork", timeout=300,
                      generator=torch.Generator().manual_seed(seed))


def check_batch(b, windows):
    x, mask = b["x"], b["mask"]
    if tuple(x.shape[1:]) != (len(ROLES), windows, 3, SIZE, SIZE) or mask.shape != x.shape[:3] \
            or mask.dtype != torch.bool or x.shape[0] != len(b["idx"]):
        raise ValueError("unexpected study tensor or mask shape")


def window_stats(valid):
    s = sorted(valid)
    return {"total": sum(s), "per_study_min": s[0] if s else None,
            "per_study_median": s[len(s) // 2] if s else None, "per_study_max": s[-1] if s else None}


def train_epoch(model, opt, sched, scaler, ds, tgt, obs, epoch, dev):
    """-> (stats, stopped). Stops gracefully at STOP_AT_S; stopped epochs are never validated."""
    model.train()
    fails, valid = Counter(), []
    st = {"studies_expected": len(ds), "studies_ok": 0, "optimizer_calls": 0, "optimizer_steps_succeeded": 0,
          "amp_skipped_steps": 0,
          "batches_skipped_zero_observed": 0, "loss_weighted_sum": 0.0, "observed_cells": 0}
    seen = 0
    for b in make_loader(ds, TRAIN_BS, True, SEED * 1000 + epoch):
        seen += b["n"]; fails.update(b["errs"])
        if b["errs"]:
            break
        if b["idx"]:
            check_batch(b, TRAIN_WINDOWS)
            st["studies_ok"] += len(b["idx"]); valid += b["mask"].flatten(1).sum(1).tolist()
            t, o = tgt[b["idx"]], obs[b["idx"]]
            if not o.any():
                st["batches_skipped_zero_observed"] += 1
            else:
                x, mask = b["x"].to(dev, non_blocking=True), b["mask"].to(dev, non_blocking=True)
                t, o = t.to(dev, non_blocking=True), o.to(dev, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.float16):
                    logits = model(x, mask)
                loss = masked_bce(logits.float(), t, o)       # raises on shape/nonfinite logits
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite training loss")
                scale = scaler.get_scale()
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
                skipped = scaler.get_scale() < scale
                st["amp_skipped_steps"] += int(skipped)
                if not skipped:
                    sched.step()
                    st["optimizer_steps_succeeded"] += 1
                n_observed = int(o.sum())
                st["optimizer_calls"] += 1
                st["loss_weighted_sum"] += float(loss) * n_observed
                st["observed_cells"] += n_observed
        if elapsed() > STOP_AT_S:
            break
    stopped = seen < len(ds) or elapsed() > STOP_AT_S
    if not stopped and st["studies_ok"] + sum(fails.values()) != len(ds):
        raise RuntimeError("training studies unaccounted for")
    st["studies_failed"], st["failed_by_exception"] = sum(fails.values()), dict(fails)
    if st["optimizer_calls"] and st["amp_skipped_steps"] / st["optimizer_calls"] > MAX_AMP_SKIP_FRACTION:
        st["amp_overflow_gate_failed"] = True
    st["mean_train_loss"] = st.pop("loss_weighted_sum") / max(1, st["observed_cells"])
    st["valid_windows"], st["lr_end"] = window_stats(valid), sched.get_last_lr()[0]
    return st, stopped


@torch.no_grad()
def validate(model, ds, tgt, obs, dev):
    """-> (stats, probs [N,12] float32 with NaN rows for failed studies, ok bool [N], stopped)."""
    model.eval()
    logits = np.full((len(ds), len(TARGETS)), np.nan, dtype=np.float32)
    ok, fails, valid, seen = np.zeros(len(ds), dtype=bool), Counter(), [], 0
    for b in make_loader(ds, VAL_BS, False, SEED):
        seen += b["n"]; fails.update(b["errs"])
        if b["idx"]:
            check_batch(b, INFER_WINDOWS)
            idx = b["idx"]; valid += b["mask"].flatten(1).sum(1).tolist()
            with torch.autocast("cuda", dtype=torch.float16):
                out = model(b["x"].to(dev, non_blocking=True), b["mask"].to(dev, non_blocking=True)).float()
            if tuple(out.shape) != (len(idx), len(TARGETS)) or not torch.isfinite(out).all():
                raise FloatingPointError("nonfinite or misshaped validation logits")
            if ok[idx].any():
                raise RuntimeError("duplicate validation prediction")
            logits[idx], ok[idx] = out.cpu().numpy(), True
        if elapsed() > STOP_AT_S:
            break
    if seen < len(ds):
        return None, None, None, True
    if int(ok.sum()) + sum(fails.values()) != len(ds):
        raise RuntimeError("validation studies unaccounted for")
    probs = torch.sigmoid(torch.from_numpy(logits)).numpy()
    t, o = tgt.numpy()[ok], obs.numpy()[ok]
    auc, support, macro = evaluate(probs[ok], t, o)
    bce = float(masked_bce(torch.from_numpy(logits[ok]), torch.from_numpy(t), torch.from_numpy(o)))
    st = {"studies_expected": len(ds), "studies_ok": int(ok.sum()), "studies_failed": sum(fails.values()),
          "failed_by_exception": dict(fails), "auc_by_target": auc, "support_by_target": support,
          "macro_auc": macro, "masked_bce": bce, "valid_windows": window_stats(valid)}
    return st, probs, ok, False


def write_predictions(path, uids, probs, ok):
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        w = csv.writer(handle)
        w.writerow(["StudyInstanceUID", *TARGETS])
        for uid, p, good in zip(uids, probs, ok):
            if good:
                w.writerow([uid, *(f"{v:.8g}" for v in p)])
    os.replace(tmp, path)


def save_checkpoint(model, epoch, val):
    path, tmp = OUT / "p02_best.pt", OUT / "p02_best.pt.tmp"
    torch.save({"model": model.state_dict(), "config": CONFIG, "source_sha256": SOURCE_SHA256, "epoch": epoch,
                "macro_auc": val["macro_auc"], "masked_bce": val["masked_bce"]}, tmp)
    os.replace(tmp, path)


def prepare():
    data = next((p for p in COMPETITION_ROOTS if (p / "train_series.csv").exists()), None)
    if data is None:
        raise FileNotFoundError("Official competition input missing")
    split_path, label_path = find_one(SPLIT_FILE, SPLIT_DIRS), find_one(LABEL_FILE, LABEL_DIRS)
    if sha256_file(split_path) != SPLIT_SHA256:
        raise ValueError("split.csv SHA256 mismatch")
    if sha256_file(label_path) != LABEL_SHA256:
        raise ValueError("llm_labels_full.csv SHA256 mismatch")
    by_study, universe = read_series(data)
    split = read_split(split_path, universe)
    labels = read_labels(label_path, universe, set(split["candidate_train"]) | set(split["candidate_validation"]))
    excluded = set(split["excluded"])
    train_uids = [u for u in split["candidate_train"] if labels[u][1].any()]
    val_uids = [u for u in split["candidate_validation"] if labels[u][1].any()]
    train_cells = int(sum(labels[u][1].sum() for u in split["candidate_train"]))
    if len(split["candidate_train"]) - len(train_uids) != EXPECTED_TRAIN_ALL_MASKED:
        raise ValueError("unexpected number of all-masked training studies")
    if train_cells != EXPECTED_TRAIN_OBSERVED_CELLS:
        raise ValueError("unexpected number of observed training cells")
    if excluded & (set(train_uids) | set(val_uids)) or set(train_uids) & set(val_uids):
        raise ValueError("excluded/overlapping study reached a training or validation list")
    stack = lambda uids, j: torch.from_numpy(np.stack([labels[u][j] for u in uids]))
    tr_t, tr_o, va_t, va_o = stack(train_uids, 0), stack(train_uids, 1), stack(val_uids, 0), stack(val_uids, 1)
    for k in range(len(TARGETS)):
        m = va_o[:, k]
        if int((va_t[m, k] > 0.5).sum()) < MIN_VAL_SUPPORT or int((va_t[m, k] < 0.5).sum()) < MIN_VAL_SUPPORT:
            raise ValueError("validation target support below minimum")
    summary["inputs"] = {
        "split_sha256_verified": True, "labels_sha256_verified": True, "split_counts": EXPECTED_COUNTS,
        "excluded_studies_dropped": len(excluded), "train_all_masked_dropped": EXPECTED_TRAIN_ALL_MASKED,
        "train_studies_used": len(train_uids), "train_observed_cells": train_cells,
        "train_target_cells": len(split["candidate_train"]) * len(TARGETS),
        "validation_zero_observed_skipped": len(split["candidate_validation"]) - len(val_uids),
        "validation_eligible_uid_sha256": hashlib.sha256("\n".join(val_uids).encode()).hexdigest(),
        "validation_studies_used": len(val_uids), "train_csv_read": False}
    return data, by_study, train_uids, val_uids, tr_t, tr_o, va_t, va_o


def main():
    summary["self_check_passed"] = self_check()
    if not torch.cuda.is_available():
        summary["stop_reason"] = "cuda_unavailable"; save()
        raise SystemExit("CUDA not available: enable GPU")
    seed_all(SEED)
    props = torch.cuda.get_device_properties(0)
    summary["environment"] = {
        "gpu": props.name, "vram_mib": props.total_memory / 2**20, "gpu_count": torch.cuda.device_count(),
        "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(), "torch": torch.__version__,
        "torchvision": torchvision.__version__, "pydicom": pydicom.__version__, "numpy": np.__version__,
        "python": sys.version.split()[0], "platform": platform.platform(), "cpu_count": os.cpu_count()}
    data, by_study, train_uids, val_uids, tr_t, tr_o, va_t, va_o = prepare()
    save()
    dev = torch.device("cuda")
    model = P02().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    planned = MAX_EPOCHS * math.ceil(len(train_uids) / TRAIN_BS)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=planned)
    scaler = torch.amp.GradScaler("cuda")
    val_ds = StudyDS(data, val_uids, by_study, train=False)
    best_key = None
    for epoch in range(1, MAX_EPOCHS + 1):
        if summary["epochs"]:
            need = max(e["epoch_seconds"] for e in summary["epochs"]) * PROJECTION_HEADROOM
            summary["last_projection"] = {"before_epoch": epoch, "elapsed_s": elapsed(), "projected_epoch_s": need,
                                          "stop_at_s": STOP_AT_S}
            if elapsed() + need > STOP_AT_S:
                summary["stop_reason"] = "projected_next_epoch_exceeds_cap"
                break
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t_ep = time.perf_counter()
        tr, stopped = train_epoch(model, opt, sched, scaler, StudyDS(data, train_uids, by_study, True, epoch),
                                  tr_t, tr_o, epoch, dev)
        torch.cuda.synchronize(); t_tr = time.perf_counter()
        if tr["studies_failed"]:
            summary["interrupted_epoch"] = {"epoch": epoch, "train": tr}
            raise RuntimeError("training study decode failure")
        if tr["optimizer_steps_succeeded"] == 0 or tr.get("amp_overflow_gate_failed"):
            summary["interrupted_epoch"] = {"epoch": epoch, "train": tr}
            raise RuntimeError("AMP optimizer update gate failed")
        if stopped:
            summary["stop_reason"], summary["interrupted_epoch"] = "deadline_during_training", {"epoch": epoch, "train": tr}
            break
        va, probs, ok, stopped = validate(model, val_ds, va_t, va_o, dev)
        torch.cuda.synchronize(); t_va = time.perf_counter()
        if stopped:
            summary["stop_reason"], summary["interrupted_epoch"] = "deadline_during_validation", {"epoch": epoch, "train": tr}
            break
        if va["studies_failed"]:
            summary["interrupted_epoch"] = {"epoch": epoch, "train": tr, "validation": va}
            raise RuntimeError("validation study decode failure")
        write_predictions(OUT / f"p02_val_predictions_epoch{epoch}.csv", val_uids, probs, ok)
        key = selection_key(va["macro_auc"], va["masked_bce"], epoch)
        if best_key is None or key < best_key:
            save_checkpoint(model, epoch, va)
            best_key = key
            summary["best"] = {"epoch": epoch, "macro_auc": va["macro_auc"], "masked_bce": va["masked_bce"],
                               "checkpoint": "p02_best.pt", "checkpoint_sha256": sha256_file(OUT / "p02_best.pt")}
        t_end = time.perf_counter()
        summary["epochs"].append({
            "epoch": epoch, "train": tr, "validation": va, "train_seconds": t_tr - t_ep,
            "validation_seconds": t_va - t_tr, "save_seconds": t_end - t_va, "epoch_seconds": t_end - t_ep,
            "cumulative_seconds": elapsed(), "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20})
        print(json.dumps({k: summary["epochs"][-1][k] for k in ("epoch", "epoch_seconds", "cumulative_seconds")}
                         | {"macro_auc": va["macro_auc"], "masked_bce": va["masked_bce"]}), flush=True)
        save()
    summary["epochs_completed"] = len(summary["epochs"])
    summary["completed"] = summary["epochs_completed"] == MAX_EPOCHS and elapsed() <= HARD_CAP_S
    summary["status"] = "complete" if summary["completed"] else "incomplete"
    summary["within_hard_cap"] = elapsed() <= HARD_CAP_S
    save()


if __name__ == "__main__":
    if "--self-check" in sys.argv:
        print("self_check passed" if self_check() else "self_check failed")
        raise SystemExit(0)
    if "--input-check" in sys.argv:
        summary["self_check_passed"] = self_check()
        prepare()
        save()
        print("input_check passed", json.dumps(summary["inputs"]), flush=True)
        raise SystemExit(0)
    signal.signal(signal.SIGTERM, deadline_handler)
    watchdog = start_watchdog()
    try:
        main()
    except SystemExit:
        raise
    except BaseException as exc:                 # type name only: no paths/UIDs in output
        summary["status"], summary["completed"] = "incomplete", False
        summary["stop_reason"] = f"fatal:{type(exc).__name__}"
        save(); raise
    finally:
        # Keep the watchdog alive through interpreter and DataLoader worker cleanup.
        # It exits as soon as this parent process disappears.
        pass
