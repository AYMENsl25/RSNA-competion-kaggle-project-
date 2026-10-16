"""Compare parity-kernel predictions (raw DICOM -> canon -> model, CPU fp32) with train_v3 eval
predictions from the local pilot cache (CPU fp32). Pass: max |diff| < 1e-3.
python notebooks/v3_submission/parity_compare.py <ckpt.pt> <probs_<name>.csv from the parity kernel>"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
import train_v3 as T  # noqa: E402

ck, pred = Path(sys.argv[1]), Path(sys.argv[2])
args = json.loads((ck.parent / "log.json").read_text())["args"]
W = args.get("eval_windows") or args.get("slab") or 16
pilot = [str(REPO / "outputs/cache_v3/pilot")]
studies = T.build_studies(T.read_index(pilot), T.read_split())
uids = sorted(studies)
dev = torch.device("cpu")
model = T.V3(args.get("head", "max"), backbone=args.get("backbone", "resnet18")).eval()
model.load_state_dict(torch.load(ck, map_location="cpu", weights_only=True))
ds = T.CacheDS(pilot, [(u, studies[u]) for u in uids], W, False, mirror_left=args["mirror"])
cache = T.predict(model, T.loader(ds, 2, 0), len(uids), args["res"], dev)
with pred.open(newline="") as f:
    kern = {r["StudyInstanceUID"]: np.array([float(r[t]) for t in T.TARGETS]) for r in csv.DictReader(f)}
diff = np.abs(np.stack([kern[u] for u in uids]) - cache)
print(json.dumps({"studies": len(uids), "max_abs_diff": float(np.nanmax(diff)), "mean_abs_diff": float(np.nanmean(diff)),
                  "nan_in_kernel": int(np.isnan(diff).any(1).sum()), "pass": bool(np.nanmax(diff) < 1e-3 and not np.isnan(diff).any())}))
