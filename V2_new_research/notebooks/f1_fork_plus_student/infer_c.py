"""Part C: score the hidden test with one S1 student (train_ft.py checkpoint), same input as its training.

python infer_c.py <core_dir> <ckpt> <data_dir> <out_csv> <deadline_epoch_s>
The volume comes from the embedded Raptor reader (96 slices, 384 px, 140 mm) and is cut to the centre 352 px,
exactly the lonnieqin tensors the student was trained on; windows = 48 eval centres, plain + anatomical mirror.
A study that fails or is not reached before the deadline is written as NaN (the caller then keeps plain F0).
"""
import os, sys, time
import numpy as np, pandas as pd, torch, torch.nn.functional as F

core, ckpt, data_dir, out_csv, deadline = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], float(sys.argv[5])
sys.path.insert(0, core)
import zoo_raptor_core as C  # noqa: E402

S, K_EVAL = 352, int(os.environ.get("K_EVAL", 48))
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ck = torch.load(ckpt, map_location="cpu", weights_only=False)
args = ck["args"]
arch, view, zoom = args["arch"], args.get("view", "global"), args.get("zoom", 0.6)
if arch.startswith(("coatnet", "maxvit")):
    model = C.build_model(arch=arch)
else:
    import timm
    bb = timm.create_model(arch, pretrained=False, num_classes=0, global_pool="avg")
    model = C.RaptorClassifier(bb, F_dim=bb.num_features)
model.load_state_dict({k: v.float() for k, v in ck["model"].items()}, strict=True)
model = model.to(dev).eval()
print(f"[partC] {os.path.basename(ckpt)} arch={arch} view={view} epoch={ck.get('epoch')}", flush=True)


def norm(x):
    return (x - C._MEAN.to(x.device).view(1, 1, 3, 1, 1)) / C._STD.to(x.device).view(1, 1, 3, 1, 1)


def views(tri, cs, mirror):
    """tri uint8 [1,T,3,S,S] -> normalised [1,T',3,320,320] for the student's view (train_ft.py, no augmentation)."""
    if view == "global":
        return C.gpu_views(tri, cs, C.MAXS, C.INPUT_RES, mirror=[mirror])
    x = C._mirror_tri(tri[0], cs[0], C.MAXS) if mirror else tri[0]
    if view == "zoom":
        z = int(round(zoom * S)); o = (S - z) // 2
        x = x[:, :, o:o + z, o:o + z]
    else:  # compart: two tiles per window, side 0.5*S at x = S/2 -+ 0.22*S
        z = int(round(0.5 * S)); y0 = (S - z) // 2
        tiles = []
        for sg in (-1, 1):
            x0 = min(max(int(S / 2 + sg * 0.22 * S) - z // 2, 0), S - z)
            tiles.append(x[:, :, y0:y0 + z, x0:x0 + z])
        x = torch.stack(tiles, 1).flatten(0, 1)
    x = x.float().div(255.0)
    x = F.interpolate(x, size=(C.INPUT_RES, C.INPUT_RES), mode="bilinear", align_corners=False)
    return norm(x[None])


@torch.no_grad()
def predict(vol, mask):
    """vol uint8 [96,384,384] (reader output) or [96,352,352]; mask [96]."""
    if vol.shape[-1] != S:
        lo = (vol.shape[-1] - S) // 2
        vol = vol[:, lo:lo + S, lo:lo + S]
    cs = C.eval_centers(mask, C.MAXS, K_EVAL)
    tri = torch.from_numpy(np.ascontiguousarray(vol[np.stack([cs - 1, cs, cs + 1], 1)])).to(dev)[None]
    ps = []
    for mir in (False, True):
        with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
            x = views(tri, cs[None], mir)
            flat = x.flatten(0, 1)
            f = torch.cat([model.backbone(c) for c in flat.split(16)]).view(1, x.shape[1], -1)
            ps.append(torch.sigmoid(model.head(f).float())[0].cpu().numpy())
    return np.mean(ps, 0)


if __name__ == "__main__" and data_dir != "--selftest":
    test = pd.read_csv(f"{data_dir}/test.csv"); test["StudyInstanceUID"] = test.StudyInstanceUID.astype(str)
    ser = pd.read_csv(f"{data_dir}/test_series.csv")
    ser["StudyInstanceUID"] = ser.StudyInstanceUID.astype(str); ser["SeriesInstanceUID"] = ser.SeriesInstanceUID.astype(str)
    SER = {k: v.to_dict("records") for k, v in ser.groupby("StudyInstanceUID")}
    tsdir = f"{data_dir}/test_series" if os.path.isdir(f"{data_dir}/test_series") else f"{data_dir}/test_images"
    reader = C._load_fastread().make_builder(C.FASTREAD_VARIANT, C.IMG, C.SLOTS, hdr_threads=C.FR_HDR, px_threads=C.FR_PX)
    from concurrent.futures import ThreadPoolExecutor
    uids = test.StudyInstanceUID.tolist()
    P = np.full((len(uids), len(C.LAB)), np.nan, np.float32)
    t0, fails = time.time(), 0
    with ThreadPoolExecutor(4) as ex:  # decode ahead of the GPU
        futs = {i: ex.submit(reader, u, SER, tsdir) for i, u in enumerate(uids[:8])}
        for i, u in enumerate(uids):
            if time.time() > deadline:
                print(f"[partC] deadline at study {i}/{len(uids)}", flush=True)
                break
            if i + 8 < len(uids):
                futs[i + 8] = ex.submit(reader, uids[i + 8], SER, tsdir)
            try:
                vol, mask = futs.pop(i).result()
                if mask.sum() >= 3:
                    P[i] = predict(vol, mask)
                else:
                    fails += 1
            except Exception as e:
                fails += 1
                print(f"[partC] {u[:16]} failed: {type(e).__name__}: {e}", flush=True)
            if (i + 1) % 200 == 0:
                print(f"[partC] {i + 1}/{len(uids)} {time.time() - t0:.0f}s", flush=True)
    pd.DataFrame(P, columns=C.LAB).assign(StudyInstanceUID=uids)[["StudyInstanceUID"] + C.LAB].to_csv(out_csv, index=False)
    print(f"[partC] done {len(uids)} studies, {fails} failed, {np.isnan(P).any(1).sum()} NaN rows, {time.time() - t0:.0f}s", flush=True)
