"""Build the K1 Kaggle kernel: python build.py [budget_h]  ->  kernel/run.py + kernel/kernel-metadata.json"""
import base64, json, sys
from pathlib import Path

HERE = Path(__file__).parent
out = HERE / "kernel"
out.mkdir(exist_ok=True)
src = (HERE / "run_template.py").read_text(encoding="utf-8")
src = src.replace("__TRAIN_B64__", base64.b64encode((HERE / "train_ft.py").read_bytes()).decode())
src = src.replace("__BUDGET_H__", sys.argv[1] if len(sys.argv) > 1 else "8.0")
(out / "run.py").write_text(src, encoding="utf-8")
json.dump({
    "id": "aymenslimani/rsna-knee-k2-fresh-effv2",
    "title": "RSNA knee K2 fresh EffV2",
    "code_file": "run.py", "language": "python", "kernel_type": "script",
    "is_private": "true", "enable_gpu": "true", "enable_tpu": "false", "enable_internet": "true",
    "machine_shape": "NvidiaTeslaT4",
    "dataset_sources": [f"lonnieqin/rsna-knee-raptor-tensors-s{s}-dataset" for s in range(4)] + [
        "nartaa/rsna-knee-publication-swa-weights-20261007", "nartaa/rsna-knee-hpo-assets",
        "stevenleehans/rsna-knee-llm-report-labels", "pilkwang/rsna-knee-llm-labels",
        "lixin73/rsna-knee-llm-report-labels-sol56", "laymond/rsna-knee-abnormality-qwen3-8b-weak-labels"],
    "competition_sources": ["rsna-knee-abnormality-detection"],
    "kernel_sources": [], "model_sources": [],
}, open(out / "kernel-metadata.json", "w"), indent=2)
print("built", out)
