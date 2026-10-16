"""Generate Kaggle CPU kernels: pilot/ (20 studies + parity) and s0..s{N-1}/ (full shards).
Embeds src/m00_data.py and src/canon.py verbatim. Run: python notebooks/cache_v3/build.py"""
import json
from pathlib import Path

NSHARD = 5
HERE = Path(__file__).resolve().parent
SRC = HERE.parents[1] / "src"
SOURCES = {n: (SRC / n).read_text(encoding="utf-8") for n in ("m00_data.py", "canon.py")}
template = (HERE / "cache_template.py").read_text(encoding="utf-8")

for name, shard, pilot in [("pilot", 0, True)] + [(f"s{k}", k, False) for k in range(NSHARD)]:
    d = HERE / name
    d.mkdir(exist_ok=True)
    code = (template.replace("__SHARD__", str(shard)).replace("__NSHARD__", str(NSHARD))
            .replace("__PILOT__", str(pilot)).replace("__SOURCES__", repr(SOURCES)))
    (d / "cache.py").write_text(code, encoding="utf-8")
    meta = {
        "id": f"aymenslimani/rsna-knee-v3-cache-{name}",
        "title": f"RSNA knee v3 cache {name}",
        "code_file": "cache.py", "language": "python", "kernel_type": "script", "is_private": "true",
        "enable_gpu": "false", "enable_tpu": "false", "enable_internet": "false", "machine_shape": "",
        "dataset_sources": ["aymenslimani/rsna-knee-v2-frozen-dev-split"],
        "competition_sources": ["rsna-knee-abnormality-detection"], "kernel_sources": [], "model_sources": [],
    }
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("built", d)
