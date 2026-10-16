"""Build the E12 blend kernel: python notebooks/e12_blend/build.py

Embeds the Ens4 submission script (notebooks/v3_submission/gpu/infer.py, built for dataset rsna-knee-v3-ens4)
with OUT / STOP_AT_S redirected to the E12 env, plus raptor_reader.py and gpjw_reader.py, into
gpu/infer.py (T4) and preflight/infer.py (CPU, visible test). Never pushes.
"""
import hashlib, json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENS4 = HERE.parent / "v3_submission/gpu/infer.py"
DATASETS = ["aymenslimani/rsna-knee-v3-ens4", "dreaddevelopment/raptor-knee-maxspan",
            "goodpjw2008/rsna-knee-2-5d-convnext-reader"]


def main():
    ens4 = ENS4.read_text(encoding="utf-8")
    assert "'dataset': 'rsna-knee-v3-ens4'" in ens4, "v3_submission/gpu is not the Ens4 build"
    for old, new in (('OUT = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path(".")',
                      'OUT = Path(os.environ.get("E12_OUT", "."))'),
                     ("STOP_AT_S = 8 * 3600", 'STOP_AT_S = float(os.environ.get("E12_DEADLINE", time.time() + 8 * 3600)) - time.time()')):
        assert ens4.count(old) == 1, old
        ens4 = ens4.replace(old, new)
    sources = {"ens4": ens4, "gpjw": (HERE / "gpjw_reader.py").read_text(encoding="utf-8"),
               "raptor": (HERE / "raptor_reader.py").read_text(encoding="utf-8")}
    code = (HERE / "blend_template.py").read_text(encoding="utf-8").replace("__SOURCES__", repr(sources))
    for name, gpu in (("gpu", True), ("preflight", False)):
        d = HERE / name
        d.mkdir(exist_ok=True)
        (d / "infer.py").write_text(code, encoding="utf-8")
        slug = "rsna-knee-v3-e12-blend" + ("-gpu" if gpu else "-preflight")
        meta = {"id": f"aymenslimani/{slug}", "title": slug.replace("-", " "), "code_file": "infer.py",
                "language": "python", "kernel_type": "script", "is_private": "true",
                "enable_gpu": str(gpu).lower(), "enable_tpu": "false", "enable_internet": "false",
                "machine_shape": "NvidiaTeslaT4" if gpu else "", "dataset_sources": DATASETS,
                "competition_sources": ["rsna-knee-abnormality-detection"], "kernel_sources": [], "model_sources": []}
        (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print("built", d, hashlib.sha256(code.encode()).hexdigest()[:12])


if __name__ == "__main__":
    main()
