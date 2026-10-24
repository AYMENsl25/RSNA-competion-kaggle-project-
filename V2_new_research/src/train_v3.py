"""V3 local trainer (E1/E4/E5) on the uint8 memmap cache (notebooks/cache_v3/cache_template.py).

P00 recipe: ImageNet ResNet18, 3-slice windows as RGB, shared max over windows, linear head,
AdamW 3e-4 (wd 0.01), per-step cosine, fp16 autocast + GradScaler (never bf16).
V3 changes: cached canonical 160 mm crop, ImageNet mean/std, optional L->R mirror at load time.
Frozen dev split; studies with index exclude=1 or a split partition other than
candidate_train/candidate_validation are never loaded (58 expert studies stay sealed).
Validation is scored against BOTH public and v3 labels every epoch.

E1:  python src/train_v3.py --cache outputs/cache_v3/s0 outputs/cache_v3/s1 outputs/cache_v3/s2
       outputs/cache_v3/s3 outputs/cache_v3/s4 --out outputs/e1_v3 --epochs 12 --mirror
Re-running the same command resumes from <out>/last.pt.
E2 options: --slab N (N evenly spaced windows/plane in train, deterministic; eval matches unless
--eval-windows), --head attn (per-finding gated attention-MIL, src/v3_model.py).
"""
import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import p00_model  # noqa: E402
from m00_data import ROLES, select_series, select_slots, slot_names  # noqa: E402
from m00_model import TARGETS  # noqa: E402
from v3_model import MEAN, STD, TIMM, V3, fill_windows, mirror, prep  # noqa: E402,F401  (shared with inference)

SPLIT = ROOT / "outputs/kaggle_split_source/split.csv"
LABELS = {"public": ROOT / "outputs/public_label_candidate/llm_labels_full.csv",
          "v3": ROOT / "outputs/labels_v3/labels_v3.csv"}
PARTS = ("candidate_train", "candidate_validation")
p00_model.WEIGHT_DIRS = (ROOT / "assets/pretrained_resnet18_v1",)  # offline, SHA-checked by P00


def h64(text):
    return int(hashlib.sha256(text.encode()).hexdigest()[:16], 16)


def read_split():
    with SPLIT.open(newline="", encoding="utf-8-sig") as f:
        return {r["StudyInstanceUID"]: (r["group_id"], r["partition"]) for r in csv.DictReader(f)}


def read_labels(name):
    """{uid: float64[12] report probabilities}; v3's change_log column is ignored."""
    with LABELS[name].open(newline="", encoding="utf-8-sig") as f:
        out = {r["StudyInstanceUID"]: np.array([float(r[t]) for t in TARGETS]) for r in csv.DictReader(f)}
    p = np.stack(list(out.values()))
    assert np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all(), f"bad {name} labels"
    return out


def targets(p, mode):
    """Probabilities -> (target, observed). hard: P00 rule (>0.5 -> 1, <0.5 -> 0); soft keeps p.
    ==0.5 is masked in both."""
    p = np.asarray(p, dtype=np.float32)
    return ((p > 0.5).astype(np.float32) if mode == "hard" else p), p != 0.5


def read_index(cache_dirs):
    rows = []
    for k, d in enumerate(cache_dirs):
        with (Path(d) / "index.csv").open(newline="", encoding="utf-8") as f:
            rows += [dict(r, shard=k) for r in csv.DictReader(f)]
    return rows


def build_studies(rows, split, slots=3):
    """-> {uid: {role: (shard, start, length, laterality, plane_geom)}}. Only ok, non-excluded series
    of candidate_train/validation studies (by BOTH the index and split.csv); P00 selection per role
    among the cached series (ponytail: one series per plane; more slots = loop over sorted keys)."""
    by_study = {}
    for r in rows:
        part = split.get(r["study"], (None, "missing"))[1]
        if r["ok"] != "1" or r["exclude"] != "0" or part not in PARTS or r["partition"] != part \
                or int(r["length"]) < 3:
            continue
        by_study.setdefault(r["study"], []).append(r)
    out = {}
    for uid, rs in by_study.items():
        sel = select_slots(({"SeriesInstanceUID": r["series"], "Fluid_Sensitive": r["fluid"],
                             "Fat_Suppression": r["fatsat"], "Anatomical_Plane": r["plane"]} for r in rs), slots)
        pick = {r["series"]: r for r in rs}
        out[uid] = {role: (pick[s]["shard"], int(pick[s]["start"]), int(pick[s]["length"]),
                           pick[s]["laterality"], pick[s]["plane_geom"]) for role, s in sel.items() if s}
    return out


class CacheDS(Dataset):
    """Key: (i, epoch) for train (window rng = f(seed, epoch, uid), like E01), i for eval.
    Returns uint8 x [R,W,3,256,256], bool mask [R,W]. Normalisation/resize happen on GPU."""

    def __init__(self, cache_dirs, items, windows, train, seed=0, mirror_left=False, slab=False, slots=3, with_positions=False):
        self.paths = [str(Path(d) / "slices.npy") for d in cache_dirs]
        self.items, self.windows, self.train, self.seed, self.mirror_left = items, windows, train, seed, mirror_left
        self.with_positions = with_positions
        self.slab = slab  # train with deterministic evenly spaced windows instead of random
        self.mms = None
        self.slot_names = slot_names(slots)  # missing slots stay masked (m=False)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, key):
        if self.mms is None:  # opened lazily: one memmap set per worker process
            self.mms = [np.load(p, mmap_mode="r") for p in self.paths]
        i, epoch = key if self.train else (key, 0)
        uid, roles = self.items[i]
        rng = np.random.default_rng(h64(f"train:{self.seed}:{epoch}:{uid}")) if self.train and not self.slab else None
        side = self.mms[0].shape[-1]
        x = np.zeros((len(self.slot_names), self.windows, 3, side, side), np.uint8)
        m = np.zeros((len(self.slot_names), self.windows), bool)
        positions = np.zeros(m.shape, np.float32)
        for r, role in enumerate(self.slot_names):
            if role not in roles:
                continue
            shard, start, length, lat, plane = roles[role]
            vol = self.mms[shard][start:start + length]
            if self.mirror_left and lat == "L":
                vol = mirror(vol, plane)
            fill_windows(x[r], m[r], vol, self.windows, rng, positions[r] if self.with_positions else None)
        result = (i, torch.from_numpy(x), torch.from_numpy(m))
        return (*result, torch.from_numpy(positions)) if self.with_positions else result


class EpochSampler:
    """Shuffled (i, epoch) keys; set .epoch before iterating (works with persistent workers)."""

    def __init__(self, n, seed):
        self.n, self.seed, self.epoch = n, seed, 1

    def __len__(self):
        return self.n

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed * 1000 + self.epoch)
        return iter([(int(i), self.epoch) for i in torch.randperm(self.n, generator=g)])


def roc_auc(y, s):
    """Tie-aware ROC-AUC (average ranks); None if a class is absent."""
    y, s = np.asarray(y, bool), np.asarray(s, np.float64)
    npos, nneg = int(y.sum()), int((~y).sum())
    if npos == 0 or nneg == 0:
        return None
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    ranks = (np.cumsum(cnt) - (cnt - 1) / 2.0)[inv]
    return float((ranks[y].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def score(uids, probs, labels):
    """Per-finding AUC vs one label set (truth p>0.5, p==0.5 / missing study masked)."""
    auc = {}
    for k, t in enumerate(TARGETS):
        rows = [(labels[u][k] > 0.5, probs[j, k]) for j, u in enumerate(uids) if u in labels and labels[u][k] != 0.5]
        auc[t] = roc_auc([a for a, _ in rows], [b for _, b in rows]) if rows else None
    vals = [v for v in auc.values() if v is not None]
    nosyn = [v for t, v in auc.items() if v is not None and t != "Synovitis"]
    return {"auc": auc, "macro": float(np.mean(vals)) if vals else None,
            "macro_no_synovitis": float(np.mean(nosyn)) if nosyn else None, "targets_scored": len(vals)}


def gpu_stats():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,clocks.sm,power.draw,clocks_throttle_reasons.active",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
        t, clk, pw, thr = [v.strip() for v in out.splitlines()[0].split(",")]
        return {"temp_c": int(t), "sm_mhz": int(clk), "power_w": float(pw), "throttle": thr}
    except Exception as e:  # logging only; never kills training
        return {"error": type(e).__name__}


def loader(ds, bs, workers, sampler=None, prefetch=4):
    return DataLoader(ds, batch_size=bs, sampler=sampler, num_workers=workers, pin_memory=True,
                      persistent_workers=workers > 0, prefetch_factor=prefetch if workers else None)


def augment(x, rot=10.0, shift=0.08, scale=0.1, inten=0.15):
    """Anatomy-safe train aug on prepped [B,R,W,3,H,W]: one random affine (rotation, shift, zoom; no flips,
    which would swap medial/lateral) and one contrast/brightness jitter per study-plane, shared by its windows."""
    B, R, W, C, H, Wd = x.shape
    n, dev = B * R, x.device
    a = torch.deg2rad((torch.rand(n, device=dev) * 2 - 1) * rot)
    s = 1 + (torch.rand(n, device=dev) * 2 - 1) * scale
    t = (torch.rand(n, 2, device=dev) * 2 - 1) * shift * 2  # grid coords span [-1,1]
    theta = torch.stack([torch.stack([a.cos() / s, -a.sin() / s, t[:, 0]], 1),
                         torch.stack([a.sin() / s, a.cos() / s, t[:, 1]], 1)], 1)
    theta = theta.repeat_interleave(W, 0)
    flat = x.reshape(n * W, C, H, Wd)
    grid = F.affine_grid(theta.to(flat.dtype), flat.shape, align_corners=False)
    flat = F.grid_sample(flat, grid, padding_mode="zeros", align_corners=False)
    c = 1 + (torch.rand(n, device=dev) * 2 - 1) * inten
    b = (torch.rand(n, device=dev) * 2 - 1) * inten
    flat = flat * c.repeat_interleave(W)[:, None, None, None] + b.repeat_interleave(W)[:, None, None, None]
    return flat.view(B, R, W, C, H, Wd)


@torch.no_grad()
def predict(model, dl, n, res, dev):
    model.eval()
    probs = np.full((n, len(TARGETS)), np.nan, np.float32)
    for batch in dl:
        i, x, m, *positions = batch
        with torch.autocast("cuda", dtype=torch.float16):
            out = model(prep(x.to(dev, non_blocking=True), res), m.to(dev, non_blocking=True),
                        *[p.to(dev, non_blocking=True) for p in positions]).float()
        probs[i.numpy()] = torch.sigmoid(out).cpu().numpy()
    assert np.isfinite(probs).all(), "missing or nonfinite validation predictions"
    return probs


def make_model(dev, head="max", backbone="resnet18"):
    if backbone != "resnet18":
        import timm
        m = V3(head, backbone=backbone)
        src = timm.create_model(TIMM[backbone], pretrained=True, num_classes=0)  # HF cache after first download
        m.encoder.load_state_dict(src.state_dict())
    elif head == "max":
        m = p00_model.P00()  # unchanged E1 init order
    else:
        m = V3(head)
        m.encoder = p00_model.verified_pretrained_encoder()
    return m.to(dev).to(memory_format=torch.channels_last)


def make_optimizer(model, lr, wd, encoder_lr=None, head_lr=None):
    """Default preserves previous optimizer state layout; optional N5 parameter groups."""
    if encoder_lr is None and head_lr is None:
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    encoder_lr = lr if encoder_lr is None else encoder_lr
    head_lr = lr if head_lr is None else head_lr
    if min(encoder_lr, head_lr) <= 0:
        raise ValueError("Learning rates must be positive")
    encoder = list(model.encoder.parameters())
    encoder_ids = {id(p) for p in encoder}
    head = [p for p in model.parameters() if id(p) not in encoder_ids]
    return torch.optim.AdamW([{"params": encoder, "lr": encoder_lr, "name": "encoder"},
                              {"params": head, "lr": head_lr, "name": "head"}],
                             lr=lr, weight_decay=wd)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", nargs="+", required=True, help="shard dirs with slices.npy + index.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--labels", choices=LABELS, default="public")
    ap.add_argument("--target", choices=("hard", "soft"), default="hard")
    ap.add_argument("--pos-weight-clip", type=float, default=None, help="pos_weight=neg/pos clipped to [1, clip]")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--val-bs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--encoder-lr", type=float, default=None)
    ap.add_argument("--head-lr", type=float, default=None)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--res", type=int, default=224)
    ap.add_argument("--train-windows", type=int, default=4)
    ap.add_argument("--eval-windows", type=int, default=None, help="default 16, or --slab N if given")
    ap.add_argument("--slab", type=int, default=0, help="train on N evenly spaced windows/plane (deterministic)")
    ap.add_argument("--head", choices=("max", "attn", "context"), default="max")
    ap.add_argument("--slots", type=int, choices=(3, 6), default=3,
                    help="3: one series/plane; 6: plane x {fat-sat, non-fat-sat} (2x windows, ~2x time)")
    ap.add_argument("--backbone", choices=("resnet18", "convnext_tiny"), default="resnet18",
                    help="convnext_tiny: timm fb_in22k_ft_in1k (E8; used with --lr 1e-4)")
    ap.add_argument("--grad-ckpt", action="store_true", help="timm gradient checkpointing (VRAM)")
    ap.add_argument("--mirror", action="store_true", help="mirror laterality-L series to R at load time")
    ap.add_argument("--workers", type=int, default=3, help="3 measured best on this laptop; 6 halved throughput (CPU/GPU power sharing)")
    ap.add_argument("--prefetch", type=int, default=4, help="batches prefetched per worker; lower on a RAM-constrained laptop")
    ap.add_argument("--log-every", type=int, default=500, help="training steps between progress and thermal checks")
    ap.add_argument("--aug", action="store_true", help="anatomy-safe affine + intensity aug (no flips)")
    ap.add_argument("--avg-last", type=int, default=0, help="also evaluate the weight average of the last K epochs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pause-temp", type=int, default=87, help="pause training while GPU >= this temp (C)")
    ap.add_argument("--max-train-studies", type=int, default=0, help="dev only: truncate train list")
    ap.add_argument("--repeat", type=int, default=1, help="dev only: repeat train list (throughput tests)")
    args = ap.parse_args(argv)
    if args.prefetch < 1 or args.log_every < 1:
        ap.error("--prefetch and --log-every must be positive")
    if args.eval_windows is None:
        args.eval_windows = args.slab or 16
    if args.slab:
        args.train_windows = args.slab

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    dev = torch.device("cuda")

    split = read_split()
    labels = {k: read_labels(k) for k in LABELS}
    studies = build_studies(read_index(args.cache), split, args.slots)
    train_lab = labels[args.labels]
    tr_uids = sorted(u for u in studies if split[u][1] == "candidate_train" and u in train_lab
                     and targets(train_lab[u], args.target)[1].any())  # E01 drops all-masked studies
    va_uids = sorted(u for u in studies if split[u][1] == "candidate_validation")
    if args.max_train_studies:
        tr_uids = tr_uids[:args.max_train_studies]
    tr_uids = tr_uids * args.repeat
    tt = [targets(train_lab[u], args.target) for u in tr_uids]
    tr_t = torch.from_numpy(np.stack([a for a, _ in tt]))
    tr_o = torch.from_numpy(np.stack([b for _, b in tt]))
    pw = None
    if args.pos_weight_clip:
        pos, neg = (tr_t * tr_o).sum(0), ((1 - tr_t) * tr_o).sum(0)
        pw = (neg / pos.clamp_min(1)).clamp(1, args.pos_weight_clip).to(dev)

    tr_ds = CacheDS(args.cache, [(u, studies[u]) for u in tr_uids], args.train_windows, True, args.seed, args.mirror,
                    slab=bool(args.slab), slots=args.slots, with_positions=args.head == "context")
    va_ds = CacheDS(args.cache, [(u, studies[u]) for u in va_uids], args.eval_windows, False, args.seed, args.mirror,
                    slots=args.slots, with_positions=args.head == "context")
    sampler = EpochSampler(len(tr_ds), args.seed)
    tr_dl = loader(tr_ds, args.bs, args.workers, sampler, prefetch=args.prefetch)
    va_dl = loader(va_ds, args.val_bs, args.workers, prefetch=args.prefetch)

    model = make_model(dev, args.head, args.backbone)
    if args.grad_ckpt:
        model.encoder.set_grad_checkpointing(True)
    opt = make_optimizer(model, args.lr, args.wd, args.encoder_lr, args.head_lr)
    steps = math.ceil(len(tr_ds) / args.bs)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * steps)
    scaler = torch.amp.GradScaler("cuda")
    log = {"args": vars(args), "train_studies": len(tr_uids), "val_studies": len(va_uids),
           "pos_weight": pw.tolist() if pw is not None else None, "epochs": []}
    start, best = 1, None
    ck = out / "last.pt"
    if ck.exists():  # resume
        s = torch.load(ck, map_location=dev, weights_only=False)
        model.load_state_dict(s["model"]), opt.load_state_dict(s["opt"])
        sched.load_state_dict(s["sched"]), scaler.load_state_dict(s["scaler"])
        start, best, log["epochs"] = s["epoch"] + 1, s["best"], s["log_epochs"]
        print(f"resumed after epoch {s['epoch']}", flush=True)
    print(f"train {len(tr_uids)} val {len(va_uids)} steps/epoch {steps}", flush=True)
    (out / "log.json").write_text(json.dumps(log, indent=1))  # lets src/dashboard.py show the run before epoch 1 ends

    for epoch in range(start, args.epochs + 1):
        model.train()
        sampler.epoch = epoch
        t0 = time.perf_counter()
        wait = loss_sum = 0.0
        n_steps = skipped = 0
        tw = time.perf_counter()
        for batch in tr_dl:
            i, x, m, *positions = batch
            wait += time.perf_counter() - tw
            t, o = tr_t[i].to(dev, non_blocking=True), tr_o[i].to(dev, non_blocking=True)
            if o.any():
                with torch.autocast("cuda", dtype=torch.float16):
                    xb = prep(x.to(dev, non_blocking=True), args.res)
                    logits = model(augment(xb) if args.aug else xb, m.to(dev, non_blocking=True),
                                   *[p.to(dev, non_blocking=True) for p in positions])
                loss = F.binary_cross_entropy_with_logits(logits.float(), t, pos_weight=pw, reduction="none")[o].mean()
                opt.zero_grad(set_to_none=True)
                scale = scaler.get_scale()
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                skipped += int(scaler.get_scale() < scale)
                loss_sum += loss.item()
            sched.step()
            n_steps += 1
            if n_steps % args.log_every == 0:
                g = gpu_stats()
                print(f"ep {epoch} step {n_steps}/{steps} loss {loss_sum / n_steps:.4f} {g}", flush=True)
                while g.get("temp_c", 0) >= args.pause_temp:  # thermal safety
                    time.sleep(30)
                    g = gpu_stats()
            tw = time.perf_counter()
        torch.cuda.synchronize()
        t_train = time.perf_counter() - t0
        assert skipped <= max(2, 0.01 * n_steps), f"AMP skipped {skipped}/{n_steps} steps"
        t1 = time.perf_counter()
        probs = predict(model, va_dl, len(va_uids), args.res, dev)
        t_val = time.perf_counter() - t1
        with (out / f"val_pred_epoch{epoch}.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["StudyInstanceUID", *TARGETS])
            w.writerows([u, *(f"{v:.8g}" for v in p)] for u, p in zip(va_uids, probs))
        val = {k: score(va_uids, probs, labels[k]) for k in LABELS}
        rec = {"epoch": epoch, "train_loss": loss_sum / max(1, n_steps), "amp_skipped": skipped,
               "train_seconds": t_train, "loader_wait_seconds": wait, "loader_wait_fraction": wait / t_train,
               "train_studies_per_hour": len(tr_ds) / t_train * 3600, "val_seconds": t_val,
               "val_studies_per_hour": len(va_uids) / t_val * 3600, "lr_end": sched.get_last_lr()[0],
               "peak_mem_mib": torch.cuda.max_memory_allocated() / 2**20, "gpu": gpu_stats(), "val": val}
        log["epochs"].append(rec)
        sel = val[args.labels]["macro"]
        if sel is not None and (best is None or sel > best["macro"]):
            best = {"epoch": epoch, "macro": sel}
            torch.save(model.state_dict(), out / "best.pt")
        log["best"] = best
        if args.avg_last and epoch > args.epochs - args.avg_last:
            torch.save(model.state_dict(), out / f"ep{epoch}.pt")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": epoch, "best": best, "log_epochs": log["epochs"],
                    "args": vars(args)}, out / "last.pt.tmp")
        os.replace(out / "last.pt.tmp", ck)
        (out / "log.json").write_text(json.dumps(log, indent=1))
        print(json.dumps({k: rec[k] for k in ("epoch", "train_loss", "train_seconds", "loader_wait_fraction",
                                               "train_studies_per_hour", "val_seconds")}
                         | {"public_macro": val["public"]["macro"], "v3_macro": val["v3"]["macro"],
                            "temp_c": rec["gpu"].get("temp_c")}), flush=True)
    if args.avg_last and not (out / "avg.pt").exists():
        sds = [torch.load(out / f"ep{e}.pt", map_location=dev) for e in range(args.epochs - args.avg_last + 1, args.epochs + 1)]
        avg = {k: (sum(sd[k].float() for sd in sds) / len(sds)).to(sds[0][k].dtype) for k in sds[0]}
        model.load_state_dict(avg)
        probs = predict(model, va_dl, len(va_uids), args.res, dev)
        with (out / "val_pred_avg.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["StudyInstanceUID", *TARGETS])
            w.writerows([u, *(f"{v:.8g}" for v in p)] for u, p in zip(va_uids, probs))
        log["avg"] = {"epochs": args.avg_last, "val": {k: score(va_uids, probs, labels[k]) for k in LABELS}}
        torch.save(avg, out / "avg.pt")
        (out / "log.json").write_text(json.dumps(log, indent=1))
        print(json.dumps({"avg_last": args.avg_last, "public_macro": log["avg"]["val"]["public"]["macro"],
                          "v3_macro": log["avg"]["val"]["v3"]["macro"]}), flush=True)
    return log


if __name__ == "__main__":
    main()
