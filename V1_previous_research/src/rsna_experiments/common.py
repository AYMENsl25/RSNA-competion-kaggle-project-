"""Input contracts shared by the standalone experiment notebooks."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import platform

UID = 'StudyInstanceUID'
TARGETS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA',
           'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
YC = [f'y_{i}' for i in range(12)]
WC = [f'w_{i}' for i in range(12)]
ROOTS = ['/kaggle/input/datasets/aymenslimani', '/kaggle/input']


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str, allow_nan=False).encode()).hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, sort_keys=True, indent=2, default=str, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def safe_output(out, identity):
    out = Path(out)
    marker = out / 'run_identity.json'
    if marker.exists():
        if json.loads(marker.read_text()) != {'identity': identity}:
            raise ValueError(f'Output belongs to a different experiment: {out}. Choose a new directory.')
    elif out.exists() and any(out.iterdir()):
        raise ValueError(f'Unrecognized nonempty output directory: {out}')
    out.mkdir(parents=True, exist_ok=True)
    write_json(marker, {'identity': identity})
    return out


def locate(filename, roots=None, override=None):
    if override:
        path = Path(override)
        if not path.is_file():
            raise FileNotFoundError(f'Explicit input does not exist: {path}')
        return path
    roots = ROOTS if roots is None else roots
    hits = sorted({p.resolve() for root in roots if Path(root).exists()
                   for p in Path(root).rglob(filename) if p.is_file()})
    if not hits:
        raise FileNotFoundError(f'Attach the saved output containing {filename}; searched {roots}')
    if len(hits) > 1 and len({sha256(p) for p in hits}) > 1:
        raise ValueError(f'Conflicting {filename} copies; set the explicit path: {hits}')
    print(f'{filename}: {hits[0]}', flush=True)
    return hits[0]


def competition_path():
    for root in ['/kaggle/input/competitions/rsna-knee-abnormality-detection',
                 '/kaggle/input/rsna-knee-abnormality-detection']:
        if (Path(root) / 'train.csv').is_file():
            return Path(root)
    raise FileNotFoundError('Attach the RSNA competition dataset.')


def environment():
    versions = {}
    for name in ['torch', 'torchvision', 'monai', 'timm', 'pydicom', 'numpy', 'pandas', 'scikit-learn']:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {'python': platform.python_version(), 'packages': versions}
