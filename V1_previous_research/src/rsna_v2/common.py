"""Frozen input contracts and provenance. No gold labels are exported for training."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import platform

import numpy as np
import pandas as pd

TARGETS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA',
           'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
SLUGS = ['acl', 'mcl', 'medial_meniscus', 'lateral_meniscus', 'medial_oa',
         'lateral_oa', 'pf_oa', 'effusion', 'synovitis', 'bakers', 'contusion', 'fracture']
PSEUDO_COLS = ['pseudo_' + s for s in SLUGS]
WEIGHT_COLS = ['weight_' + s for s in SLUGS]
SLOTS = ['SAG_FS', 'SAG_NONFS', 'COR_FS', 'COR_NONFS', 'AX_FS', 'AX_NONFS']
UID = 'StudyInstanceUID'
COMPETITION_PATHS = ['/kaggle/input/competitions/rsna-knee-abnormality-detection',
                     '/kaggle/input/rsna-knee-abnormality-detection']


def hash_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def fingerprint(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str,
                                     allow_nan=False).encode()).hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str,
                               allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def environment():
    packages = {}
    for name in ['numpy', 'pandas', 'torch', 'torchvision', 'timm', 'pydicom',
                 'opencv-python-headless', 'scikit-learn', 'pylibjpeg', 'nbformat']:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {'python': platform.python_version(), 'platform': platform.platform(), 'packages': packages}


def resolve_unique(filename, roots, override=None):
    if override:
        p = Path(override)
        if not p.is_file():
            raise FileNotFoundError(f'Explicit {filename} path does not exist: {p}')
        return p
    hits = sorted({p.resolve() for root in roots if Path(root).exists()
                   for p in Path(root).rglob(filename) if p.is_file()})
    if not hits:
        raise FileNotFoundError(f'Attach prior output containing {filename}; searched {roots}')
    if len(hits) > 1 and len({hash_file(p) for p in hits}) != 1:
        raise ValueError(f'Conflicting copies of {filename}; set its explicit path: {hits}')
    return hits[0]


def resolve_comp_dir(explicit=None):
    choices = [str(explicit)] if explicit else COMPETITION_PATHS
    for p in choices:
        if Path(p).is_dir() and (Path(p) / 'train.csv').is_file():
            return Path(p)
    raise FileNotFoundError(f'Add competition data; expected one of {choices}')


def require_columns(frame, columns, context):
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f'{context}: missing columns {sorted(missing)}')


def build_supervision(config):
    """Freeze group roles, reuse v1.2 folds, and save pseudo-only train_manifest.csv."""
    roots = config.get('input_roots', ['/kaggle/input'])
    paths = {key: resolve_unique(name, roots, config.get(key)) for key, name in {
        'slot_plan_path': 'rsna_phase4b2a_six_slot_plan_v1.csv',
        'pseudo_path': 'rsna_pseudo_labels_v1_1_wide.csv',
        'pseudo_meta_path': 'PSEUDO_LABELS_v1_1.json',
    }.items()}
    meta = json.loads(paths['pseudo_meta_path'].read_text(encoding='utf-8-sig'))
    if meta.get('freeze_name') != 'PSEUDO_LABELS_v1_1':
        raise ValueError('Expected unchanged PSEUDO_LABELS_v1_1 freeze')
    roles_raw = pd.read_csv(paths['slot_plan_path'], dtype={UID: str, 'cv_group_id': str})
    role_cols = [UID, 'cv_group_id', 'training_role']
    require_columns(roles_raw, role_cols, 'slot plan')
    roles = roles_raw[role_cols].drop_duplicates()
    if roles.isna().any().any() or roles[UID].duplicated().any():
        raise ValueError('Missing or inconsistent study role/group across series')
    roles = roles[roles.training_role.isin(['PSEUDO_TRAIN', 'GOLD_EVAL_LOCKED',
                                          'GOLD_GROUP_LOCKED_NON_GOLD'])].copy()
    expected = config.get('expected_counts', {'PSEUDO_TRAIN': 4341, 'GOLD_EVAL_LOCKED': 58,
                                             'GOLD_GROUP_LOCKED_NON_GOLD': 6})
    counts = roles.training_role.value_counts().to_dict()
    if expected is not None and counts != expected:
        raise ValueError(f'Frozen cohort mismatch: observed {counts}; expected {expected}')
    locked_groups = set(roles.loc[roles.training_role != 'PSEUDO_TRAIN', 'cv_group_id'])
    train = roles[roles.training_role == 'PSEUDO_TRAIN'].copy()
    if locked_groups.intersection(train.cv_group_id):
        raise ValueError('Gold/group-excluded group found in pseudo training')
    pseudo = pd.read_csv(paths['pseudo_path'], dtype={UID: str})
    require_columns(pseudo, [UID] + PSEUDO_COLS + WEIGHT_COLS, 'pseudo labels')
    if pseudo[UID].duplicated().any():
        raise ValueError('Duplicate study in pseudo table')
    train = train.merge(pseudo[[UID] + PSEUDO_COLS + WEIGHT_COLS], on=UID,
                        how='left', validate='one_to_one')
    values = train[PSEUDO_COLS + WEIGHT_COLS].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError('Missing, nonfinite, or out-of-range pseudo targets/weights')
    if (train[WEIGHT_COLS].sum(axis=0) <= 0).any():
        raise ValueError('A target has no active training supervision')
    folds_file = config.get('folds_path')
    if not config.get('regenerate_folds', False):
        paths['folds_path'] = resolve_unique('rsna_phase4b1b_pseudo_fold_assignments_v1_2.csv',
                                            roots, folds_file)
        folds = pd.read_csv(paths['folds_path'], dtype={UID: str, 'cv_group_id': str})
        require_columns(folds, [UID, 'cv_group_id', 'fold'], 'frozen folds')
        if folds[UID].duplicated().any() or set(folds[UID]) != set(train[UID]):
            raise ValueError('Frozen fold studies do not match the pseudo cohort exactly')
        train = train.merge(folds, on=[UID, 'cv_group_id'], how='left', validate='one_to_one')
    else:
        from sklearn.model_selection import GroupKFold
        train = train.sort_values(UID).reset_index(drop=True)
        train['fold'] = -1
        for fold, (_, val) in enumerate(GroupKFold(n_splits=config.get('n_folds', 5)).split(
                train, groups=train.cv_group_id)):
            train.loc[val, 'fold'] = fold
    if train.fold.isna().any() or (train.groupby('cv_group_id').fold.nunique() != 1).any():
        raise ValueError('Fold missing, group mismatch, or group split across folds')
    if set(train.fold) != set(range(config.get('n_folds', 5))):
        raise ValueError('Unexpected fold IDs')
    train['fold'] = train.fold.astype(int)
    for fold in sorted(train.fold.unique()):
        for part in [train[train.fold == fold], train[train.fold != fold]]:
            if (part[WEIGHT_COLS].sum(axis=0) <= 0).any():
                raise ValueError(f'Fold {fold} has a zero-weight target in train or validation')
    out = Path(config['out_dir'])
    input_hashes = {k: hash_file(v) for k, v in paths.items()}
    identity = fingerprint({'inputs': input_hashes, 'regenerate_folds': config.get('regenerate_folds', False),
                            'n_folds': config.get('n_folds', 5), 'targets': TARGETS})
    if (out / 'supervision_meta.json').exists():
        previous = json.loads((out / 'supervision_meta.json').read_text())
        if previous['supervision_id'] != identity:
            raise ValueError('Output contains different supervision; choose a new out_dir')
    out.mkdir(parents=True, exist_ok=True)
    train.sort_values(UID).to_csv(out / 'train_manifest.csv', index=False)
    roles.sort_values(UID).to_csv(out / 'study_roles.csv', index=False)
    write_json(out / 'supervision_meta.json', {
        'supervision_id': identity, 'freeze': 'PSEUDO_LABELS_v1_1', 'targets': TARGETS,
        'counts': counts, 'group_leakage_check': 'PASS', 'inputs': {k: str(v) for k,v in paths.items()},
        'input_sha256': input_hashes, 'train_manifest_sha256': hash_file(out / 'train_manifest.csv'),
        'study_roles_sha256': hash_file(out / 'study_roles.csv'),
        'folds_regenerated': config.get('regenerate_folds', False), 'environment': environment(),
        'gold_labels_exported': False})
    print(f'Supervision PASS: {counts}; {len(set(train.cv_group_id))} pseudo groups. Saved {out}')
    return out
