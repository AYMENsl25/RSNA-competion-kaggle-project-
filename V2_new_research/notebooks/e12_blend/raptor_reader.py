"""E12 reader: dreaddevelopment Raptor CoAtNet maxspan (dataset dreaddevelopment/raptor-knee-maxspan, CC0).

Model + preprocessing adapted from the public notebook dreaddevelopment/knee-mri-twelve-findings-from-a-single-model
(Apache-2.0), with the maxspan corpus settings used by rjings/rsna-versia-8: IMG 336, span 2-98 %, 64 slices,
62 eval windows, res 384. Writes <E12_OUT>/reader_probs.csv (NaN row = study failed / not reached) and reader_info.json.
Deadline guard: after WINDOW_CHECK_AT studies the projected time is compared with E12_DEADLINE (epoch s); if too slow,
K_EVAL is cut (noted in reader_info); at the deadline the loop stops and the remaining studies stay NaN.
"""
import csv, gc, glob, json, os, sys, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import timm

torch.backends.cudnn.benchmark = True
OUT = os.environ.get("E12_OUT", ".")
DEADLINE = float(os.environ.get("E12_DEADLINE", time.time() + 1e9))
T0 = time.time()
FILE, ARCH, RES = "raptor_ft_coatnet_v5_full_swa.pt", "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", 384
IMG, CROP_MM, SPAN = 336, 140.0, (0.02, 0.98)
SLOTS = [("Sagittal", 1, 18), ("Sagittal", 0, 14), ("Coronal", 1, 12), ("Coronal", 0, 8), ("Axial", -1, 12)]
MAXS = sum(s[2] for s in SLOTS)  # 64
K_EVAL, K_MIN, WINDOW_CHECK_AT = 62, 16, 20
LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
       "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class RaptorClassifier(nn.Module):
    def __init__(self, backbone, F_dim=768, n=12, drop=0.2):
        super().__init__()
        self.backbone = backbone
        self.norm = nn.LayerNorm(F_dim)
        self.att = nn.Sequential(nn.Linear(F_dim, 256), nn.Tanh(), nn.Dropout(drop), nn.Linear(256, n))
        self.clsW = nn.Parameter(torch.zeros(n, F_dim))
        self.clsb = nn.Parameter(torch.zeros(n))

    def forward(self, x):
        B, K = x.shape[:2]
        h = self.norm(self.backbone(x.flatten(0, 1)).view(B, K, -1))
        a = torch.softmax(self.att(h), dim=1)
        return (torch.einsum("bkn,bkf->bnf", a, h) * self.clsW).sum(-1) + self.clsb


def eval_windows(vol, mask, k, res):
    D = vol.shape[0]
    valid = np.where(mask > 0)[0]
    if len(valid) < 3:
        valid = np.arange(min(3, D))
    lo, hi = int(valid.min()), int(valid.max())
    cs = [c for c in range(lo + 1, hi)] or [max(1, min((lo + hi) // 2, D - 2))]
    cs = [cs[i] for i in np.linspace(0, len(cs) - 1, k).round().astype(int)]
    wins = np.empty((len(cs), 3, res, res), np.float32)
    for j, c in enumerate(cs):
        c = max(1, min(c, D - 2))
        t = torch.from_numpy(np.stack([vol[c - 1], vol[c], vol[c + 1]], 0).astype(np.float32) / 255.0)
        if t.shape[-1] != res:
            t = F.interpolate(t[None], size=(res, res), mode="bilinear", align_corners=False)[0]
        wins[j] = t.numpy()
    return (torch.from_numpy(wins) - _MEAN) / _STD


def make_reader():
    import pydicom, cv2
    from pydicom.pixel_data_handlers.util import apply_modality_lut

    def order_and_meta(sdir):
        recs, ps_list = [], []
        for f in glob.glob(sdir + "/*.dcm"):
            try:
                h = pydicom.dcmread(f, stop_before_pixels=True)
                iop, ipp = getattr(h, "ImageOrientationPatient", None), getattr(h, "ImagePositionPatient", None)
                if iop is not None and ipp is not None and len(iop) == 6:
                    pos = float(np.dot(np.array(ipp, float), np.cross(np.array(iop[:3], float), np.array(iop[3:], float))))
                else:
                    pos = float(getattr(h, "InstanceNumber", 0) or 0)
                ps = getattr(h, "PixelSpacing", None)
                ps = float(ps[0]) if ps is not None else 0.5
                ps_list.append(ps); recs.append((pos, f, ps))
            except Exception:
                recs.append((0.0, f, 0.5))
        recs.sort(key=lambda x: x[0])
        return [(f, ps) for _, f, ps in recs], (float(np.median(ps_list)) if ps_list else 0.5)

    def read_px(f):
        d = pydicom.dcmread(f)
        a = apply_modality_lut(d.pixel_array, d).astype(np.float32)
        return a.max() - a if str(getattr(d, "PhotometricInterpretation", "")) == "MONOCHROME1" else a

    def crop(a, ps):
        h, w = a.shape
        cpx = min(int(round(CROP_MM / max(ps, 1e-3))), h, w)
        y0, x0 = (h - cpx) // 2, (w - cpx) // 2
        return cv2.resize(a[y0:y0 + cpx, x0:x0 + cpx], (IMG, IMG), interpolation=cv2.INTER_AREA)
    return order_and_meta, read_px, crop


def pick(rows, plane, fluid, used):
    c = [r for r in rows if r["Anatomical_Plane"] == plane and r["SeriesInstanceUID"] not in used]
    if fluid in (0, 1):
        pref = [r for r in c if int(r.get("Fluid_Sensitive", 0) or 0) == fluid]
        if pref:
            return pref[0]
    return c[0] if c else None


def build_study(sid, ser, tsdir, reader):
    order_and_meta, read_px, crop = reader
    vol, idx, used = np.zeros((MAXS, IMG, IMG), np.uint8), 0, set()
    for plane, fluid, k in SLOTS:
        r = pick(ser.get(sid, []), plane, fluid, used)
        if r is None:
            idx += k; continue
        used.add(r["SeriesInstanceUID"])
        files, med_ps = order_and_meta(f"{tsdir}/{sid}/{r['SeriesInstanceUID']}")
        if not files:
            idx += k; continue
        n = len(files)
        lo, hi = int(n * SPAN[0]), max(int(n * SPAN[1]) - 1, int(n * SPAN[0]))
        picks = np.linspace(lo, hi, k).round().astype(int) if n > 1 else [0] * k
        arrs, pss = [], []
        for p in picks:
            fp, ps = files[min(p, n - 1)]
            try:
                arrs.append(read_px(fp)); pss.append(ps)
            except Exception:
                arrs.append(None); pss.append(med_ps)
        valid = [a for a in arrs if a is not None]
        loq, hiq = np.percentile(np.concatenate([a.ravel() for a in valid]), [2.0, 98.0]) if valid else (0.0, 1.0)
        for a, ps in zip(arrs, pss):
            if idx >= MAXS:
                break
            if a is not None:
                vol[idx] = (crop(np.clip((a - loq) / (hiq - loq + 1e-6), 0, 1), ps if ps > 0 else med_ps) * 255).astype(np.uint8)
            idx += 1
    return vol, (vol.reshape(MAXS, -1).sum(1) > 0).astype(np.uint8)


def find_weights():
    for p in ("/kaggle/input/raptor-knee-maxspan", "/kaggle/input/datasets/dreaddevelopment/raptor-knee-maxspan"):
        hits = glob.glob(f"{p}/**/{FILE}", recursive=True)
        if hits:
            return hits[0]
    for d in glob.glob("/kaggle/input/*/") + glob.glob("/kaggle/input/datasets/*/*/"):
        if "competition" not in d:
            hits = glob.glob(f"{d}**/{FILE}", recursive=True)
            if hits:
                return hits[0]
    raise FileNotFoundError(FILE)


def main():
    import pandas as pd
    root = sys.argv[1]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ids = pd.read_csv(f"{root}/test.csv")["StudyInstanceUID"].astype(str).tolist()
    tser = pd.read_csv(f"{root}/test_series.csv").astype({"StudyInstanceUID": str, "SeriesInstanceUID": str})
    ser = {k: v.to_dict("records") for k, v in tser.groupby("StudyInstanceUID")}
    wpath = find_weights()
    import hashlib
    wsha = hashlib.sha256(open(wpath, "rb").read()).hexdigest()
    ck = torch.load(wpath, map_location="cpu", weights_only=False)
    bb = timm.create_model(ck.get("arch", ARCH), pretrained=False, num_classes=0, in_chans=3, global_pool="avg")
    model = RaptorClassifier(bb, F_dim=bb.num_features)
    model.load_state_dict(ck["model"], strict=True)
    model.eval().to(dev)
    res = int(ck.get("res", RES)); del ck; gc.collect()
    k, notes, fails, t_loop = K_EVAL, [], 0, time.time()
    probs = np.full((len(ids), 12), np.nan)
    for i, sid in enumerate(ids):
        if time.time() > DEADLINE:
            notes.append(f"deadline hit at study {i}; {len(ids) - i} left NaN"); break
        if i == WINDOW_CHECK_AT and len(ids) > 2 * WINDOW_CHECK_AT:
            per = (time.time() - t_loop) / i
            need, left = per * (len(ids) - i), DEADLINE - time.time()
            if need > left:
                k = max(K_MIN, int(k * left / need * 0.9))
                notes.append(f"windows reduced {K_EVAL}->{k} at study {i} (projected {need:.0f}s > left {left:.0f}s)")
        try:
            vol, mask = build_study(sid, ser, f"{root}/test_series", reader)
            x = eval_windows(vol, mask, k, res)[None].to(dev)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
                p = torch.sigmoid(model(x).float())[0].cpu().numpy()
            if not np.isfinite(p).all():
                raise FloatingPointError("nonfinite")
            probs[i] = p
        except Exception as e:
            fails += 1
            print(f"raptor study {i} FAILED {type(e).__name__}: {e}", flush=True)
        if (i + 1) % 100 == 0:
            print(f"raptor {i + 1}/{len(ids)} {time.time() - T0:.0f}s", flush=True)
    pd.DataFrame(probs, columns=LAB).assign(StudyInstanceUID=ids)[["StudyInstanceUID"] + LAB].to_csv(f"{OUT}/reader_probs.csv", index=False)
    json.dump({"weights": wpath, "sha256": wsha, "device": dev, "k_eval_final": k, "failures": fails, "notes": notes, "loop_s": round(time.time() - t_loop, 1), "elapsed_s": round(time.time() - T0, 1)},
              open(f"{OUT}/reader_info.json", "w"))


reader = make_reader()
if __name__ == "__main__":
    main()
