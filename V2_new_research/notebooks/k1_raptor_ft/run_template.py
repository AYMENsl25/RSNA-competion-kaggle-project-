"""K1 Kaggle runner: two fine-tunes of nartaa's public Raptor-384 in parallel, one per T4.

1. find inputs, check the parent checkpoint sha256
2. consensus soft labels from five public LLM report-label tables (gold-58 excluded from training)
3. parity gate: rebuild 3 training studies from DICOM with the reader embedded in the core and compare
   with the tensor rows (centre 352 px); any mismatch stops the run
4. arm A on GPU 0, arm B on GPU 1 (train_ft.py, embedded below); progress in /kaggle/working/status.json
"""
import os, sys, json, time, glob, base64, hashlib, subprocess, shutil
import numpy as np, pandas as pd

T0 = time.time()
W = "/kaggle/working"
HARD_STOP_H = 11.4
BUDGET_H = float(os.environ.get("K1_BUDGET_H", "__BUDGET_H__"))
TRAIN_B64 = "__TRAIN_B64__"
PARENT = "raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt"
PARENT_SHA = "7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef"
LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
       "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
status = {"stage": "start"}


def say(*x):
    print(f"[{time.time() - T0:7.0f}s]", *x, flush=True)


def put(**kw):
    status.update(kw, elapsed_h=round((time.time() - T0) / 3600, 3))
    json.dump(status, open(f"{W}/status.json", "w"), indent=1)


def find(name, root="/kaggle/input"):
    for d, dirs, fs in os.walk(root):
        dirs[:] = [x for x in dirs if "abnormality-detection" not in x and x != "competitions"]
        if name in fs:
            return os.path.join(d, name)
    return None


def comp_root():
    for b in ["/kaggle/input/competitions/rsna-knee-abnormality-detection", "/kaggle/input/rsna-knee-abnormality-detection"]:
        if os.path.exists(b + "/train.csv"):
            return b
    raise RuntimeError("competition data not attached")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(8 << 20), b""):
            h.update(c)
    return h.hexdigest()


# ---------------------------------------------------------------- 1. inputs
put(stage="inputs")
COMP = comp_root()
core_src = find("zoo_raptor_core.py")
parent = find(PARENT)
assert core_src and parent, (core_src, parent)
assert sha(parent) == PARENT_SHA, "parent checkpoint bytes differ"
os.makedirs(f"{W}/core", exist_ok=True)
shutil.copy(core_src, f"{W}/core/zoo_raptor_core.py")
open(f"{W}/core/train_ft.py", "wb").write(base64.b64decode(TRAIN_B64))
say("comp", COMP, "| core", core_src, "| parent ok")

# ---------------------------------------------------------------- 2. labels
tr = pd.read_csv(f"{COMP}/train.csv")
tr["StudyInstanceUID"] = tr.StudyInstanceUID.astype(str)
ids = tr.StudyInstanceUID
is_gold = tr[LAB].notna().all(axis=1).values
gold = ids[is_gold].tolist()


def table(fname, suffix=""):
    p = find(fname)
    if p is None:
        say("label source missing:", fname)
        return None
    try:
        d = pd.read_parquet(p) if p.endswith(".parquet") else pd.read_csv(p)
        if "StudyInstanceUID" not in d.columns:
            d = d.reset_index().rename(columns={d.index.name or "index": "StudyInstanceUID"})
        d["StudyInstanceUID"] = d.StudyInstanceUID.astype(str)
        t = d.set_index("StudyInstanceUID")[[c + suffix for c in LAB]].set_axis(LAB, axis=1).astype(float)
    except Exception as e:
        say("label source unreadable, skipped:", fname, type(e).__name__, str(e)[:120])
        return None
    t = t.where((t >= 0) & (t <= 1))
    return t[~t.index.duplicated()].reindex(ids)


src = {"steven_v4": table("llm_labels_v4_blend.csv"), "pilkwang_v2": table("report_labels_v2.csv"),
       "lixin_gpt56": table("labels_llm_gpt56sol.csv"), "qwen": table("qwen_knee_weak_labels.csv", "__label"),
       "nartaa_v1_blend": table("labels_v1_blend.parquet")}
src = {k: v for k, v in src.items() if v is not None}
stack = np.stack([v.values for v in src.values()])               # [S, N, 12]
cons = np.clip(np.nanmean(stack, 0), 0.01, 0.99)
cover = {k: round(float(v.notna().mean().mean()), 3) for k, v in src.items()}
say("label sources", cover)
assert len(src) >= 3, "need at least three label sources"
y = {u: [None if np.isnan(x) else round(float(x), 4) for x in cons[i]] for i, u in enumerate(ids) if not is_gold[i]}
gold_y = {u: tr.loc[tr.StudyInstanceUID == u, LAB].values[0].astype(float).tolist() for u in gold}
json.dump({"y": y, "gold": gold, "gold_y": gold_y, "sources": list(src)}, open(f"{W}/labels_k1.json", "w"))
put(stage="labels", label_sources=cover, consensus_mean={c: round(float(np.nanmean(cons[~is_gold, j])), 3) for j, c in enumerate(LAB)})

# ---------------------------------------------------------------- 3. parity gate
sys.path.insert(0, f"{W}/core")
import zoo_raptor_core as C  # noqa: E402

ser = pd.read_csv(f"{COMP}/train_series.csv")
ser["StudyInstanceUID"] = ser.StudyInstanceUID.astype(str); ser["SeriesInstanceUID"] = ser.SeriesInstanceUID.astype(str)
SER = {k: v.to_dict("records") for k, v in ser.groupby("StudyInstanceUID")}
tsdir = f"{COMP}/train_series" if os.path.isdir(f"{COMP}/train_series") else f"{COMP}/train_images"
reader = C._load_fastread().make_builder(C.FASTREAD_VARIANT, C.IMG, C.SLOTS, hdr_threads=C.FR_HDR, px_threads=C.FR_PX)
par = []
for s in range(3):
    idx = pd.read_csv(find(f"index_s{s}.csv"))
    vol = np.load(find(f"vol_s{s}.u8.npy"), mmap_mode="r")
    r = idx.iloc[len(idx) // 2]
    v, m = reader(str(r.StudyInstanceUID), SER, tsdir)
    lo = (C.IMG - vol.shape[-1]) // 2
    diff = int(np.abs(v[:, lo:lo + vol.shape[-1], lo:lo + vol.shape[-1]].astype(int) - vol[int(r.row)].astype(int)).max())
    par.append(diff)
say("parity max|diff| per shard sample:", par)
put(stage="parity", parity=par)
assert max(par) == 0, f"tensor/reader mismatch {par}"
del reader, C

# ---------------------------------------------------------------- 4. arms
ARMS = {"A": dict(gpu=0, extra=[]), "B": dict(gpu=1, extra=["--T", "12", "--zoom", "0.6"])}
procs = {}
for arm, cfg in ARMS.items():
    out = f"{W}/arm{arm}"
    os.makedirs(out, exist_ok=True)
    cmd = [sys.executable, f"{W}/core/train_ft.py", "--arm", arm, "--core", f"{W}/core", "--parent", parent,
           "--labels_json", f"{W}/labels_k1.json", "--out", out, "--budget_h", str(BUDGET_H)] + cfg["extra"]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(cfg["gpu"]), PYTHONUNBUFFERED="1")
    procs[arm] = subprocess.Popen(cmd, stdout=open(f"{out}/stdout.txt", "w"), stderr=subprocess.STDOUT, env=env)
    say("launched arm", arm, "gpu", cfg["gpu"])
put(stage="training", arms=list(ARMS))


def tail(arm, n=3):
    try:
        return open(f"{W}/arm{arm}/stdout.txt").read().strip().splitlines()[-n:]
    except Exception:
        return []


while any(p.poll() is None for p in procs.values()):
    time.sleep(120)
    put(stage="training", rc={a: p.poll() for a, p in procs.items()}, tail={a: tail(a, 2) for a in procs})
    for a in procs:
        for line in tail(a, 1):
            say(a, line[:300])
    if time.time() - T0 > HARD_STOP_H * 3600:
        for p in procs.values():
            if p.poll() is None:
                p.terminate()
        say("hard stop")
        break
for a, p in procs.items():
    p.wait(timeout=300)
put(stage="done", rc={a: p.returncode for a, p in procs.items()}, tail={a: tail(a, 6) for a in procs},
    files=sorted(os.path.relpath(f, W) for f in glob.glob(f"{W}/arm*/*.pt")))
say("done", status["rc"])
