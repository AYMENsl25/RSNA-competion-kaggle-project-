"""CPU parity: reload selected heads and reproduce all saved validation predictions."""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from f2_data import sha
from f2_model import Head


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    folder = args.run_dir.resolve()
    cfg = json.loads((folder / "config.json").read_text())
    if json.loads((folder / "summary.json").read_text())["status"] != "complete":
        raise ValueError("main paired run is incomplete")
    torch.set_num_threads(cfg["cpu_threads"])
    with (ROOT / cfg["split"]).open(newline="", encoding="utf-8-sig") as f:
        partition = {r["StudyInstanceUID"]: r["partition"] for r in csv.DictReader(f)}
    uids = json.loads((folder / "private_uid_order.json").read_text())
    va = np.array([i for i, u in enumerate(uids) if partition[u] == "candidate_validation"])
    if len(va) != 863:
        raise ValueError("validation cohort changed")
    data = (torch.from_numpy(np.load(folder / "features.npy").astype(np.float32)),
            torch.from_numpy(np.load(folder / "valid.npy")),
            torch.from_numpy(np.load(folder / "geometry.npy")))
    paths = {"A": folder, "B": folder, "C": folder / "geometry_ablation"}
    report, checksums = {}, {}
    for arm, directory in paths.items():
        ck_path = directory / f"f2_{arm}_best.pt"
        ck = torch.load(ck_path, map_location="cpu", weights_only=True)
        model = Head(width=cfg["head_width"], geometry=arm != "A").eval()
        model.load_state_dict(ck["state_dict"], strict=True)
        prob_path = directory / f"f2_{arm}_val_probabilities.npy"
        expected = np.load(prob_path, allow_pickle=False)
        preds = []
        with torch.inference_mode():
            for lo in range(0, len(va), cfg["head_batch"]):
                ii = va[lo:lo + cfg["head_batch"]]
                f, m, g = (a[ii] for a in data)
                preds.append(model(f, m, torch.zeros_like(g) if arm == "C" else g).sigmoid().numpy())
        actual = np.concatenate(preds)
        maximum = float(np.max(np.abs(actual - expected)))
        if actual.shape != (863, 12) or not np.isfinite(actual).all() or maximum > 1e-6:
            raise ValueError(f"{arm} saved artifact prediction parity failed")
        report[arm] = {"strict_checkpoint_load": True, "max_absolute_prediction_difference": maximum,
                       "validation_shape": list(actual.shape), "epoch": ck["epoch"]}
        checksums[str(ck_path.relative_to(folder))] = sha(ck_path)
        checksums[str(prob_path.relative_to(folder))] = sha(prob_path)
    report.update(status="passed", checkpoint_and_prediction_sha256=checksums)
    path = folder / "artifact_verification.json"
    if path.exists():
        raise FileExistsError("verification already recorded")
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
