"""K1: fine-tune nartaa's public Raptor-384 CoAtNet (LB 0.949) on lonnieqin's 96-slice tensors.

The tensors were built by the same embedded DICOM reader the 0.958 notebook uses at test time (decoded
source sha 5a59b6f8...), stored as the centre 352 px of the 384 px / 140 mm volume, so the 320 px centre
crop here is the inference input. One arm per GPU:

  A  architecture unchanged. Targets = alpha * consensus of 5 public LLM report-label tables
     + (1 - alpha) * frozen parent prediction on the same augmented windows (keeps what the parent
     learned from OAI). Mirror / crop-shift / gain / gamma augmentation.
  B  same targets, dual view: the global 320 crop plus a centre zoom (frac of the 352 px, resized to
     320) of the same windows, zero-initialised (view, slot) embeddings and a learned attention offset
     for zoom windows (starts at -2, so the start point is close to the parent).

Gold-58 is never trained on; its AUC is printed per epoch for information only (the parent's SWA epochs
were picked on gold-58, so the parent's own gold score is optimistic). The predeclared result is the
last epoch; the public LB decides.
"""
import os, sys, json, time, math, argparse, copy
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from sklearn.metrics import roc_auc_score

ap = argparse.ArgumentParser()
ap.add_argument("--arm", choices=["A", "B"], required=True)
ap.add_argument("--core", required=True, help="directory holding zoo_raptor_core.py")
ap.add_argument("--parent", default="", help="parent checkpoint; empty = random init (smoke only)")
ap.add_argument("--arch", default="coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k")
ap.add_argument("--data_root", default="/kaggle/input")
ap.add_argument("--labels_json", required=True, help="json written by run.py: {uid: [12 floats or null]}, gold list")
ap.add_argument("--out", required=True)
ap.add_argument("--budget_h", type=float, default=8.0)
ap.add_argument("--emax", type=int, default=6)
ap.add_argument("--T", type=int, default=12)
ap.add_argument("--accum", type=int, default=16)
ap.add_argument("--chunk", type=int, default=4)
ap.add_argument("--bb_lr", type=float, default=1.5e-5)
ap.add_argument("--head_lr", type=float, default=3e-4)
ap.add_argument("--new_lr", type=float, default=1e-3)
ap.add_argument("--wd", type=float, default=0.05)
ap.add_argument("--alpha", type=float, default=0.5)
ap.add_argument("--zoom", type=float, default=0.6)
ap.add_argument("--k_eval", type=int, default=94)
ap.add_argument("--workers", type=int, default=2)
ap.add_argument("--limit", type=int, default=0, help="smoke: train on N studies")
ap.add_argument("--gold_limit", type=int, default=0)
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--pretrained", action="store_true", help="timm ImageNet weights (K2 fresh members; needs internet)")
ap.add_argument("--view", choices=["global", "zoom"], default="global", help="arm A input view")
a = ap.parse_args()

sys.path.insert(0, a.core)
import zoo_raptor_core as C  # noqa: E402

torch.manual_seed(a.seed); np.random.seed(a.seed)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(a.out, exist_ok=True)
T0 = time.time()
LAB = C.LAB
D = C.MAXS


def log(**kw):
    kw["t"] = round(time.time() - T0, 1)
    line = json.dumps(kw)
    print(line, flush=True)
    with open(os.path.join(a.out, "log.jsonl"), "a") as f:
        f.write(line + "\n")


# ---------------------------------------------------------------- data
def find(name):
    for d, dirs, fs in os.walk(a.data_root):
        dirs[:] = [x for x in dirs if "abnormality-detection" not in x and x != "competitions"]
        if name in fs:
            return os.path.join(d, name)
    raise FileNotFoundError(name)


VOLS, MASKS, LOC = [], [], {}
for s in range(4):
    VOLS.append(np.load(find(f"vol_s{s}.u8.npy"), mmap_mode="r"))
    MASKS.append(np.load(find(f"mask_s{s}.u8.npy")))
    idx = pd.read_csv(find(f"index_s{s}.csv"))
    for uid, row, done in zip(idx.StudyInstanceUID.astype(str), idx.row, idx.done):
        if done:
            LOC[uid] = (s, int(row))
S = VOLS[0].shape[-1]
assert VOLS[0].shape[1] == D and S == 352, VOLS[0].shape

lab = json.load(open(a.labels_json))
gold = [u for u in lab["gold"] if u in LOC]
gold_y = {u: np.array(lab["gold_y"][u], np.float32) for u in gold}
train = [u for u, v in lab["y"].items() if u in LOC and u not in set(lab["gold"])]
Y = {u: np.array([np.nan if x is None else x for x in lab["y"][u]], np.float32) for u in train}
if a.limit:
    train = train[:a.limit]
if a.gold_limit:
    gold = gold[:a.gold_limit]
prev = np.clip(np.nanmean(np.stack([Y[u] for u in train]), 0), 0.03, 0.7)
PW = torch.tensor(np.clip((1 - prev) / prev, 1, 10), dtype=torch.float32, device=dev)
log(ev="data", arm=a.arm, train=len(train), gold=len(gold), shards=[v.shape[0] for v in VOLS], pw=PW.tolist())


class Bags(torch.utils.data.Dataset):
    def __init__(self, uids):
        self.uids = uids

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, i):
        u = self.uids[i]
        s, r = LOC[u]
        rng = np.random.default_rng((torch.initial_seed() + i) % 2**32)
        cs = C.train_centers(MASKS[s][r], D, a.T, rng)
        tri = np.ascontiguousarray(VOLS[s][r][np.stack([cs - 1, cs, cs + 1], 1)])  # [T,3,S,S] uint8
        return torch.from_numpy(tri), torch.from_numpy(cs), torch.from_numpy(Y[u])


# ---------------------------------------------------------------- model
def load_state(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    return ck["model"]


parent_sd = load_state(a.parent) if a.parent else None
if a.arch.startswith(("coatnet", "maxvit", "convnext")):
    student = C.build_model(arch=a.arch, state=parent_sd, pretrained=a.pretrained and parent_sd is None)
else:  # K2: CNN families whose timm constructor takes no img_size (EfficientNetV2 etc.)
    import timm
    _bb = timm.create_model(a.arch, pretrained=a.pretrained, num_classes=0, global_pool="avg")
    student = C.RaptorClassifier(_bb, F_dim=_bb.num_features)
    if parent_sd is not None:
        student.load_state_dict({k: v.float() for k, v in parent_sd.items()}, strict=True)
BN = [m for m in student.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
log(ev="model", arch=a.arch, pretrained=a.pretrained, bn_layers=len(BN), view=a.view,
    params_m=round(sum(p.numel() for p in student.parameters()) / 1e6, 1))
teacher = copy.deepcopy(student).half().eval().to(dev) if parent_sd is not None else None
if teacher is not None:
    for p in teacher.parameters():
        p.requires_grad_(False)
SLOT = torch.from_numpy(C._slot_index(D)[0]).long().to(dev)  # slice -> slot 0..4


class DualView(nn.Module):
    """Raptor head over global + zoom windows; zero (view, slot) embeddings keep the parent function on the
    global view at step 0, zoom windows start with an attention offset of -2."""

    def __init__(self, base):
        super().__init__()
        self.base = base
        fd = base.clsW.shape[1]
        self.emb = nn.Parameter(torch.zeros(10, fd))
        self.zbias = nn.Parameter(torch.full((base.n,), -2.0))

    def pool(self, fg, fz, slot):
        f = torch.cat([fg + self.emb[slot], fz + self.emb[slot + 5]], 1)
        h = self.base.norm(f)
        att = self.base.att(h)
        k = fg.shape[1]
        att = torch.cat([att[:, :k], att[:, k:] + self.zbias], 1).float().softmax(1).to(h.dtype)
        pooled = torch.einsum("bkn,bkf->bnf", att, h)
        return (pooled * self.base.clsW).sum(-1) + self.base.clsb


model = (DualView(student) if a.arm == "B" else student).to(dev)


def encode(m, x, train):
    b, k = x.shape[:2]
    flat = x.flatten(0, 1)
    if train:
        f = torch.cat([checkpoint(m.backbone, c, use_reentrant=False) for c in flat.split(a.chunk)])
    else:
        f = torch.cat([m.backbone(c) for c in flat.split(16)])
    return f.view(b, k, -1)


def zoom_views(tri, cs, mirror, offs, gain, gamma):
    """Centre zoom of the same uint8 windows: crop frac*S (shifted like the global crop), resize to 320."""
    b = tri.shape[0]
    z = int(round(a.zoom * S))
    rows = []
    for i in range(b):
        x = tri[i]
        if mirror is not None and bool(mirror[i]):
            x = C._mirror_tri(x, cs[i], D)
        oy, ox = (int(offs[i][0]), int(offs[i][1])) if offs is not None else (0, 0)
        y0 = min(max((S - z) // 2 + oy, 0), S - z)
        x0 = min(max((S - z) // 2 + ox, 0), S - z)
        rows.append(x[:, :, y0:y0 + z, x0:x0 + z])
    x = torch.stack(rows).float().div_(255.0)
    if gamma is not None:
        x = x.clamp_min(1e-4).pow(torch.as_tensor(gamma, dtype=torch.float32, device=x.device).view(b, 1, 1, 1, 1))
    if gain is not None:
        x = (x * torch.as_tensor(gain, dtype=torch.float32, device=x.device).view(b, 1, 1, 1, 1)).clamp_(0, 1)
    k = x.shape[1]
    x = F.interpolate(x.flatten(0, 1), size=(C.INPUT_RES, C.INPUT_RES), mode="bilinear", align_corners=False)
    x = x.view(b, k, 3, C.INPUT_RES, C.INPUT_RES)
    return (x - C._MEAN.to(x.device).view(1, 1, 3, 1, 1)) / C._STD.to(x.device).view(1, 1, 3, 1, 1)


def forward(tri, cs, train, aug=None):
    mirror, offs, gain, gamma = aug if aug is not None else (None, None, None, None)
    if a.arm == "A" and a.view == "zoom":
        xz = zoom_views(tri, cs, mirror, offs, gain, gamma)
        return student.head(encode(student, xz, train)), xz
    xg = C.gpu_views(tri, cs, D, C.INPUT_RES, mirror=mirror, offs=offs, gain=gain, gamma=gamma)
    if a.arm == "A":
        return student.head(encode(student, xg, train)), xg
    xz = zoom_views(tri, cs, mirror, offs, gain, gamma)
    slot = SLOT[torch.as_tensor(cs, device=dev).long()]
    return model.pool(encode(student, xg, train), encode(student, xz, train), slot), xg


# ---------------------------------------------------------------- gold monitor (inference recipe: k_eval windows, plain + mirror)
@torch.no_grad()
def gold_eval(tag):
    if not gold:
        return None
    model.eval()
    P, Yg = [], []
    for u in gold:
        s, r = LOC[u]
        cs = C.eval_centers(MASKS[s][r], D, a.k_eval)
        tri = torch.from_numpy(np.ascontiguousarray(VOLS[s][r][np.stack([cs - 1, cs, cs + 1], 1)])).to(dev)[None]
        ps = []
        for mir in (False, True):
            with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
                lg, _ = forward(tri, cs[None], False, ([mir], None, None, None))
            ps.append(torch.sigmoid(lg.float())[0].cpu().numpy())
        P.append(np.mean(ps, 0)); Yg.append(gold_y[u])
    P, Yg = np.stack(P), np.stack(Yg)
    aucs = {n: float(roc_auc_score(Yg[:, j], P[:, j])) for j, n in enumerate(LAB) if len(set(Yg[:, j])) > 1}
    macro = float(np.mean(list(aucs.values()))) if aucs else float("nan")
    np.save(os.path.join(a.out, f"gold_{tag}.npy"), P)
    log(ev="gold", tag=tag, macro=round(macro, 4), per={k: round(v, 3) for k, v in aucs.items()})
    if "train_mode" in globals():
        train_mode()
    return macro


# ---------------------------------------------------------------- optimiser
bb = [p for n, p in student.named_parameters() if n.startswith("backbone.")]
hd = [p for n, p in student.named_parameters() if not n.startswith("backbone.")]
groups = [{"params": bb, "lr": a.bb_lr}, {"params": hd, "lr": a.head_lr}]
if a.arm == "B":
    groups.append({"params": [model.emb, model.zbias], "lr": a.new_lr, "weight_decay": 0.0})
opt = torch.optim.AdamW(groups, weight_decay=a.wd)
base_lr = [g["lr"] for g in opt.param_groups]
scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")
WARM = 30
plan = {"total": 10 ** 9}


def set_lr(step):
    if step < WARM:
        m = (step + 1) / WARM
    else:
        m = 0.5 * (1 + math.cos(math.pi * min(1.0, (step - WARM) / max(1, plan["total"] - WARM))))
    for g, b0 in zip(opt.param_groups, base_lr):
        g["lr"] = b0 * m


def save(tag, extra=None):
    sd = {k: (v.half() if v.is_floating_point() else v) for k, v in model.state_dict().items()}
    ck = {"model": sd, "arch": a.arch, "res": 384, "lab": LAB, "arm": a.arm, "zoom": a.zoom, "tag": tag,
          "parent": os.path.basename(a.parent), "args": vars(a)}
    ck.update(extra or {})
    torch.save(ck, os.path.join(a.out, f"k2{a.view}_{tag}.pt.tmp"))
    os.replace(os.path.join(a.out, f"k2{a.view}_{tag}.pt.tmp"), os.path.join(a.out, f"k2{a.view}_{tag}.pt"))


# ---------------------------------------------------------------- train
gold_eval("parent")
dl = torch.utils.data.DataLoader(Bags(train), batch_size=1, shuffle=True, num_workers=a.workers,
                                 pin_memory=dev.type == "cuda", persistent_workers=a.workers > 0, drop_last=True)
rng = np.random.default_rng(a.seed)
def train_mode():
    model.train()
    for m in BN:  # bags of 4-window chunks from one study: keep ImageNet BN statistics frozen
        m.eval()


train_mode()
if teacher is not None:
    teacher.eval()
step = micro = 0
t_probe = None
epochs = a.emax
budget_s = a.budget_h * 3600
for ep in range(a.emax):
    if ep >= epochs:
        break
    tot, n, t_ep = 0.0, 0, time.time()
    for tri, cs, y in dl:
        tri, y = tri.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        cs = cs.numpy()
        aug = (rng.random(1) < 0.5, rng.integers(-12, 13, (1, 2)), rng.uniform(0.9, 1.1, 1), rng.uniform(0.9, 1.1, 1))
        with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
            logits, xg = forward(tri, cs, True, aug)
            if teacher is not None:
                with torch.no_grad():
                    t = torch.sigmoid(teacher.head(encode(teacher, xg.half(), False)).float())
                tgt = torch.where(torch.isnan(y), t, a.alpha * y + (1 - a.alpha) * t)
            else:
                tgt = torch.nan_to_num(y, nan=0.0)
        w = (~torch.isnan(tgt if teacher is not None else y)).float()  # unknown label cells carry no loss
        loss = (F.binary_cross_entropy_with_logits(logits.float(), tgt, pos_weight=PW, reduction="none") * w).sum() \
            / w.sum().clamp_min(1) / a.accum
        scaler.scale(loss).backward()
        micro += 1
        tot += loss.item() * a.accum; n += 1
        if micro % a.accum == 0:
            set_lr(step)
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            step += 1
        if micro == 40 and t_probe is None:
            t_probe = time.time()
        if micro == 120 and plan["total"] == 10 ** 9:
            per = (time.time() - t_probe) / 80                      # seconds per study
            ep_s = per * len(train) + 240 * (2 if a.arm == "B" else 1)
            left = budget_s - (time.time() - T0) - 600
            epochs = int(max(1, min(a.emax, (left + per * 120) // ep_s)))
            plan["total"] = epochs * (len(train) // a.accum)
            log(ev="plan", sec_per_study=round(per, 3), epoch_min=round(ep_s / 60, 1), epochs=epochs,
                opt_steps=plan["total"], mem_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2) if dev.type == "cuda" else 0)
        if micro % 200 == 0:
            log(ev="step", ep=ep, micro=micro, step=step, loss=round(tot / max(n, 1), 4),
                lr=[round(g["lr"], 8) for g in opt.param_groups], scale=scaler.get_scale())
        if time.time() - T0 > budget_s:
            break
    log(ev="epoch", ep=ep, loss=round(tot / max(n, 1), 4), min=round((time.time() - t_ep) / 60, 1))
    g = gold_eval(f"ep{ep}")
    save(f"ep{ep}", {"epoch": ep, "gold_macro": g})
    if time.time() - T0 > budget_s:
        log(ev="budget_stop", ep=ep)
        break
save("last", {"epoch": ep})
log(ev="done", epochs_run=ep + 1, planned=epochs, hours=round((time.time() - T0) / 3600, 2))
