"""Explicit download of generic Apache-2.0 DINOv2-S/14, never Kaggle weights."""
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
URL = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth"


def main():
    folder = ROOT / "assets/f2_dinov2"
    folder.mkdir(exist_ok=True)
    path = folder / "dinov2_vits14_pretrain.pth"
    if path.exists():
        raise FileExistsError("weights already exist; refusing overwrite")
    partial = path.with_suffix(".partial")
    digest, size = hashlib.sha256(), 0
    with urllib.request.urlopen(URL, timeout=60) as response, partial.open("wb") as f:
        expected = int(response.headers.get("Content-Length", 0))
        while chunk := response.read(1 << 20):
            f.write(chunk)
            digest.update(chunk)
            size += len(chunk)
    if (expected and expected != size) or not 75_000_000 < size < 110_000_000:
        raise ValueError("unexpected DINOv2 download length")
    partial.rename(path)
    manifest = {"source": URL, "sha256": digest.hexdigest(), "bytes": size,
                "license": "Apache-2.0", "kind": "generic self-supervised pretrained encoder"}
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
