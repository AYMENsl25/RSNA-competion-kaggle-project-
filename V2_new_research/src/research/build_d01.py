"""Build an isolated P00 sampling comparison without touching its frozen sources."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def once(text: str, old: str, new: str) -> str:
    assert text.count(old) == 1, (old, text.count(old))
    return text.replace(old, new)


def main() -> None:
    src = ROOT / "src"
    original_data = src / "m00_data.py"
    d01_data = src / "d01_data.py"
    d01_text = once(read(original_data),
                    "TRAIN_WINDOWS, INFER_WINDOWS, SIZE = 4, 16, 224",
                    "TRAIN_WINDOWS, INFER_WINDOWS, SIZE = 8, 16, 224")
    d01_data.write_text(d01_text, encoding="utf-8")

    folder = ROOT / "notebooks" / "d01_p00_eight_windows"
    folder.mkdir(parents=True, exist_ok=True)
    base = read(ROOT / "notebooks" / "p00_full_train" / "train_template.py")
    base = once(base, "P00 full training:", "D01 P00 sampling comparison:")
    base = base.replace("p00_full_train_summary", "d01_full_train_summary")
    base = base.replace("p00_best.pt", "d01_best.pt")
    base = base.replace("p00_val_predictions_epoch", "d01_val_predictions_epoch")
    base = once(base,
                '"model": "P00 ImageNet-pretrained ResNet18 2.5D (3 adjacent slices), shared max over valid windows",',
                '"model": "D01 P00 ImageNet-pretrained ResNet18 2.5D, eight training windows per plane; shared max",')
    template = folder / "train_template.py"
    template.write_text(base, encoding="utf-8")

    model = src / "m00_model.py"
    p00 = src / "p00_model.py"
    sources = (d01_data, model, p00, template)
    hashes = {path.name: digest(path) for path in sources}
    code = once(base, "# __SOURCE_SHA256__", f"SOURCE_SHA256 = {hashes!r}")
    p00_body = once(read(p00), "from m00_model import M00\n", "")
    for marker, path, body in (
        ("# __M00_DATA_SOURCE__", d01_data, d01_text),
        ("# __M00_MODEL_SOURCE__", model, read(model)),
        ("# __P00_MODEL_SOURCE__", p00, p00_body),
    ):
        code = once(code, marker, f"# Embedded {path.name} SHA256: {digest(path)}\n{body}")
    (folder / "train.py").write_text(code, encoding="utf-8")
    base_meta = json.loads(read(ROOT / "notebooks" / "p00_full_train" / "kernel-metadata.json"))
    base_meta.update(id="aymenslimani/rsna-knee-v2-d01-full-train",
                     title="RSNA knee V2 D01 full train")
    (folder / "kernel-metadata.json").write_text(json.dumps(base_meta, indent=2) + "\n", encoding="utf-8")

    marker = 'if __name__ == "__main__":\n'
    assert code.count(marker) == 1
    cpu_suffix = read(ROOT / "notebooks" / "p00_cpu_check" / "check.py").split(marker, 1)[1]
    cpu_folder = ROOT / "notebooks" / "d01_cpu_check"
    cpu_folder.mkdir(parents=True, exist_ok=True)
    (cpu_folder / "check.py").write_text(code.split(marker, 1)[0] + marker + cpu_suffix, encoding="utf-8")
    cpu_meta = dict(base_meta)
    cpu_meta.update(id="aymenslimani/rsna-knee-v2-d01-cpu-preflight",
                    title="RSNA knee V2 D01 CPU preflight", code_file="check.py", enable_gpu="false")
    (cpu_folder / "kernel-metadata.json").write_text(json.dumps(cpu_meta, indent=2) + "\n", encoding="utf-8")

    pilot_marker = "# Pilot stops no later than one hour; finish work by 55 minutes."
    pilot_source = read(ROOT / "notebooks" / "p00_gpu_pilot" / "pilot.py")
    assert pilot_source.count(pilot_marker) == 1
    pilot_suffix = pilot_marker + pilot_source.split(pilot_marker, 1)[1]
    pilot_suffix = once(pilot_suffix, 'p00_gpu_pilot_summary.json', 'd01_gpu_pilot_summary.json')
    pilot_folder = ROOT / "notebooks" / "d01_gpu_pilot"
    pilot_folder.mkdir(parents=True, exist_ok=True)
    (pilot_folder / "pilot.py").write_text(code.split(marker, 1)[0] + pilot_suffix, encoding="utf-8")
    pilot_meta = dict(base_meta)
    pilot_meta.update(id="aymenslimani/rsna-knee-v2-d01-gpu-pilot",
                      title="RSNA knee V2 D01 GPU pilot", code_file="pilot.py")
    (pilot_folder / "kernel-metadata.json").write_text(json.dumps(pilot_meta, indent=2) + "\n", encoding="utf-8")
    print(folder / "train.py")
    print(hashes)


if __name__ == "__main__":
    main()
