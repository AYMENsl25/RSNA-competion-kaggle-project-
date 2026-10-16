"""E12 reader: goodpjw2008 2.5D ConvNeXt reader (dataset goodpjw2008/rsna-knee-2-5d-convnext-reader, Apache-2.0).

Uses the dataset's own knee.py / preprocess.py / infer.py (DicomStudyDataset, load_models logic) unchanged; only the
loop is ours: device-agnostic (CPU preflight), batch 1 so a failure costs one study, fp16 autocast on CUDA only,
deadline guard. 3 folds averaged (as infer.run). pylibjpeg wheels from the dataset go to a private --target dir.
Writes <E12_OUT>/reader_probs.csv (NaN = failed / not reached) and reader_info.json.
"""
import glob, json, os, subprocess, sys, time
import numpy as np

OUT = os.environ.get("E12_OUT", ".")
DEADLINE = float(os.environ.get("E12_DEADLINE", time.time() + 1e9))
T0 = time.time()
K = 12  # infer.run default
DS = next((d for d in ("/kaggle/input/rsna-knee-2-5d-convnext-reader",
                       "/kaggle/input/datasets/goodpjw2008/rsna-knee-2-5d-convnext-reader")
           if os.path.isdir(d)), None) or os.path.dirname(glob.glob("/kaggle/input/**/cnxt_v0_fold0.pt", recursive=True)[0])
DS = os.path.dirname(glob.glob(f"{DS}/**/knee.py", recursive=True)[0])
SITE = os.path.join(OUT, "site")
py = f"cp{sys.version_info.major}{sys.version_info.minor}"
whl = [w for w in glob.glob(f"{DS}/*.whl") if py in w or "py3-none" in w]
pip = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--no-deps", "--target", SITE, *whl],
                     capture_output=True, text=True) if whl else None
sys.path[:0] = [DS, SITE]

import pandas as pd  # noqa: E402
import torch  # noqa: E402
from infer import DicomStudyDataset  # noqa: E402
from knee import LABELS, KneeNet, build_study_table  # noqa: E402
from preprocess import MAX_SLICES  # noqa: E402


def main():
    root = sys.argv[1]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    studies = pd.read_csv(f"{root}/test.csv").StudyInstanceUID.astype(str).tolist()
    se = pd.read_csv(f"{root}/test_series.csv").astype({"StudyInstanceUID": str, "SeriesInstanceUID": str})
    se = se[se.StudyInstanceUID.isin(set(studies))]
    n_files = {}
    for st, sr in zip(se.StudyInstanceUID, se.SeriesInstanceUID):
        d = f"{root}/test_series/{st}/{sr}"
        n_files[sr] = min(len(os.listdir(d)), MAX_SLICES) if os.path.isdir(d) else 0
    ds = DicomStudyDataset(studies, build_study_table(se, n_files), f"{root}/test_series", K)
    dl = torch.utils.data.DataLoader(ds, batch_size=1, num_workers=4, pin_memory=dev.type == "cuda")
    models = []
    for path in sorted(glob.glob(f"{DS}/cnxt_v0_fold*.pt")):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        m = KneeNet(ck["args"]["backbone"].split(".")[0], pretrained=False)
        m.load_state_dict(ck["model"])
        models.append((m.to(dev).to(memory_format=torch.channels_last).eval(), ck["args"]["res"]))
    t_loop = time.time()
    probs, fails, notes = np.full((len(studies), 12), np.nan), 0, []
    for i, (x, mask, pos) in enumerate(dl):
        if time.time() > DEADLINE:
            notes.append(f"deadline hit at study {i}; {len(studies) - i} left NaN"); break
        try:
            x, mask, pos = x.to(dev), mask.to(dev), pos.to(dev)
            if not bool(mask.any()):
                raise ValueError("no usable series")
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
                p = torch.stack([m(x, mask, pos, res=r).float().sigmoid() for m, r in models]).mean(0)[0].cpu().numpy()
            if not np.isfinite(p).all():
                raise FloatingPointError("nonfinite")
            probs[i] = p
        except Exception as e:
            fails += 1
            print(f"gpjw study {i} FAILED {type(e).__name__}: {e}", flush=True)
        if (i + 1) % 100 == 0:
            print(f"gpjw {i + 1}/{len(studies)} {time.time() - T0:.0f}s", flush=True)
    pd.DataFrame(probs, columns=LABELS).assign(StudyInstanceUID=studies)[["StudyInstanceUID"] + LABELS].to_csv(f"{OUT}/reader_probs.csv", index=False)
    import hashlib
    json.dump({"sha256": {os.path.basename(f): hashlib.sha256(open(f, "rb").read()).hexdigest() for f in sorted(glob.glob(f"{DS}/cnxt_v0_fold*.pt"))},
               "device": str(dev), "folds": len(models), "k": K, "failures": fails, "notes": notes,
               "pip_wheels": [os.path.basename(w) for w in whl], "pip_rc": pip.returncode if pip else None,
               "loop_s": round(time.time() - t_loop, 1), "elapsed_s": round(time.time() - T0, 1)}, open(f"{OUT}/reader_info.json", "w"))


if __name__ == "__main__":
    main()
