"""Build the V3 submission for one or more train_v3 checkpoints (several = rank-average ensemble).

python notebooks/v3_submission/build.py [name=]<ckpt.pt> ... --dataset <slug> [--upload]
  - name defaults to the checkpoint's run dir; staged in the dataset as <name>.pt + <name>_log.json
  - reads head / eval windows / res / mirror per checkpoint from <ckpt dir>/log.json (train_v3 args)
  - stages the private checkpoint dataset in outputs/v3_submission_datasets/<slug>/ and, with
    --upload, creates it (or a new version) via the kaggle CLI (private by default)
  - writes gpu/ (T4 submission kernel), preflight/ (CPU, visible test), parity/ (CPU, 20 pilot
    train studies) each with infer.py + kernel-metadata.json. Never pushes a kernel.
"""
import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SRC = REPO / "src"
USER = "aymenslimani"


def sha(b):
    return hashlib.sha256(b).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoints", nargs="+")
    ap.add_argument("--dataset", required=True, help="checkpoint dataset slug, e.g. rsna-knee-v3-e1-checkpoint")
    ap.add_argument("--kernel", default="rsna-knee-v3-submission", help="kernel slug prefix")
    ap.add_argument("--upload", action="store_true", help="create/version the private dataset")
    a = ap.parse_args()

    cks, models = {}, []
    for item in a.checkpoints:
        name, _, path = item.rpartition("=")
        ck = Path(path).resolve()
        name = name or ck.parent.name
        args = json.loads((ck.parent / "log.json").read_text())["args"]
        cks[name] = ck
        models.append({"name": name, "checkpoint": f"{name}.pt", "sha256": sha(ck.read_bytes()), "source": str(ck),
                       "head": args.get("head", "max"), "backbone": args.get("backbone", "resnet18"), "windows": args.get("eval_windows") or args.get("slab") or 16,
                       "res": args["res"], "mirror": bool(args["mirror"]), "slots": args.get("slots", 3)})
    if len(cks) != len(models):
        raise ValueError("duplicate model names")
    sources = {n: (SRC / n).read_text(encoding="utf-8") for n in ("m00_data.py", "m00_model.py", "canon.py", "v3_model.py")}
    config = {"dataset": a.dataset, "models": models,
              "source_sha256": {n: sha(t.encode("utf-8")) for n, t in sources.items()}}
    config["run"] = [{k: m[k] for k in ("name", "head", "windows", "res", "mirror", "slots")} for m in models]

    ds_dir = REPO / "outputs/v3_submission_datasets" / a.dataset
    ds_dir.mkdir(parents=True, exist_ok=True)
    for old in [*ds_dir.glob("*.pt"), *ds_dir.glob("*log.json")]:
        old.unlink()
    for name, ck in cks.items():
        shutil.copy2(ck, ds_dir / f"{name}.pt")
        shutil.copy2(ck.parent / "log.json", ds_dir / f"{name}_log.json")
    (ds_dir / "dataset-metadata.json").write_text(json.dumps(
        {"title": a.dataset, "id": f"{USER}/{a.dataset}", "licenses": [{"name": "CC0-1.0"}]}, indent=2))

    with (REPO / "outputs/cache_v3/pilot/index.csv").open(encoding="utf-8") as f:
        pilot = sorted({line.split(",", 1)[0] for line in list(f)[1:]})
    template = (HERE / "infer_template.py").read_text(encoding="utf-8")
    for name, gpu, parity in (("gpu", True, None), ("preflight", False, None), ("parity", False, pilot)):
        code = (template.replace("__SOURCES__", repr(sources)).replace("__CONFIG__", repr(config))
                .replace("__PARITY_UIDS__", repr(parity)))
        d = HERE / name
        d.mkdir(exist_ok=True)
        (d / "infer.py").write_text(code, encoding="utf-8")
        slug = a.kernel + ("" if gpu else f"-{name}")
        meta = {"id": f"{USER}/{slug}", "title": slug.replace("-", " "), "code_file": "infer.py",
                "language": "python", "kernel_type": "script", "is_private": "true",
                "enable_gpu": str(gpu).lower(), "enable_tpu": "false", "enable_internet": "false",
                "machine_shape": "NvidiaTeslaT4" if gpu else "",
                "dataset_sources": [f"{USER}/{a.dataset}"],
                "competition_sources": ["rsna-knee-abnormality-detection"], "kernel_sources": [], "model_sources": []}
        (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print("built", d, sha(code.encode()))
    print("config", json.dumps(config["run"]), {m["name"]: m["sha256"][:12] for m in models})

    if a.upload:
        exists = subprocess.run(["kaggle", "datasets", "status", f"{USER}/{a.dataset}"], capture_output=True).returncode == 0
        cmd = (["kaggle", "datasets", "version", "-p", str(ds_dir), "-m", " ".join(f"{m['name']}:{m['sha256'][:12]}" for m in models)]
               if exists else ["kaggle", "datasets", "create", "-p", str(ds_dir)])
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
