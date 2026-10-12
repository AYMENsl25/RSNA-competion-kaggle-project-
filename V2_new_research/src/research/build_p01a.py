"""Embed audited model sources in the P01A Kaggle CPU preflight script."""

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
M00 = ROOT / "src/m00_model.py"
P01A = ROOT / "src/p01a_model.py"
PREFLIGHT_TEMPLATE = ROOT / "notebooks/p01a_cpu_preflight/preflight_template.py"
PREFLIGHT_OUT = ROOT / "notebooks/p01a_cpu_preflight/preflight.py"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    text = PREFLIGHT_TEMPLATE.read_text(encoding='utf-8')
    for marker, path, body in (
        ('# __M00_MODEL_SOURCE__', M00, M00.read_text(encoding='utf-8')),
        ('# __P01A_MODEL_SOURCE__', P01A, P01A.read_text(encoding='utf-8').replace('from m00_model import M00, TARGETS\n', '')),
    ):
        assert text.count(marker) == 1
        text = text.replace(marker, f'# Embedded {path.name} SHA256: {sha(path)}\n{body}')
    PREFLIGHT_OUT.write_text(text, encoding='utf-8')
    print(PREFLIGHT_OUT, sha(PREFLIGHT_OUT))


if __name__ == '__main__':
    main()
