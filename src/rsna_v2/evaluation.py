"""Pseudo-validation comparison and explicitly enabled historical gold audit."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from .common import (TARGETS, UID, PSEUDO_COLS, WEIGHT_COLS, hash_file,
                     fingerprint, write_json, resolve_comp_dir)


def probability_table(path):
    frame = pd.read_csv(path, dtype={UID: str})
    if UID not in frame or frame[UID].duplicated().any():
        raise ValueError(f'Missing or duplicate study IDs in {path}')
    missing = set(TARGETS) - set(frame.columns)
    if missing:
        raise ValueError(f'Probability columns missing from {path}: {missing}')
    p = frame[TARGETS].to_numpy(float)
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError('Invalid probabilities')
    return frame.set_index(UID)


def weighted_soft_bce(y, p, w):
    y, p, w = map(lambda x: np.asarray(x, dtype=float), (y, p, w))
    if y.shape != p.shape or y.shape != w.shape or y.ndim != 2:
        raise ValueError('Target/prediction/weight shapes must agree')
    if not all(np.isfinite(a).all() for a in [y,p,w]) or (w < 0).any():
        raise ValueError('Invalid values')
    if (w.sum(axis=0) <= 0).any():
        raise ValueError('A target has zero active supervision')
    p = np.clip(p, 1e-7, 1 - 1e-7)
    raw = -(y * np.log(p) + (1 - y) * np.log1p(-p))
    return (raw * w).sum(axis=0) / w.sum(axis=0)


def compare_runs(run_dirs, supervision_dir, out_dir):
    """Require identical validation IDs for fair comparisons. No gold labels read."""
    if not run_dirs:
        raise ValueError('Set RUN_DIRS to explicit selected training-output directories')
    supervision_dir = Path(supervision_dir)
    meta = json.loads((supervision_dir / 'supervision_meta.json').read_text())
    manifest = supervision_dir / 'train_manifest.csv'
    if hash_file(manifest) != meta['train_manifest_sha256']:
        raise ValueError('Supervision manifest changed')
    labels = pd.read_csv(manifest, dtype={UID: str}).set_index(UID)
    tables, rows = {}, []
    common_ids = None
    for raw_dir in run_dirs:
        directory = Path(raw_dir)
        frame = probability_table(directory / 'validation_predictions.csv')
        if not set(frame.index).issubset(labels.index):
            raise ValueError('Validation predictions include non-pseudo or unknown studies')
        ids = sorted(frame.index)
        if common_ids is None:
            common_ids = ids
        if ids != common_ids:
            raise ValueError('Runs have different validation cohorts. Compare matching folds only.')
        # Every represented group must be held out in the frozen fold assignment.
        folds = set(labels.loc[ids, 'fold'])
        expected_ids = set(labels.index[labels.fold.isin(folds)])
        if set(ids) != expected_ids:
            raise ValueError('Partial fold predictions cannot be compared as complete validation')
        record_path = directory / 'run_meta.json'
        if not record_path.is_file():
            raise FileNotFoundError(f'Missing training provenance: {record_path}')
        record = json.loads(record_path.read_text())
        if not record.get('complete') or record.get('mode') != 'selection':
            raise ValueError('Only completed selection runs are eligible')
        if record.get('supervision_id') != meta['supervision_id']:
            raise ValueError('Run supervision differs from comparison supervision')
        if 'validation_predictions_sha256' in record and hash_file(directory / 'validation_predictions.csv') != record['validation_predictions_sha256']:
            raise ValueError('Validation predictions differ from recorded hash')
        key = str(directory.resolve())
        p = frame.loc[ids, TARGETS]
        y = labels.loc[ids, PSEUDO_COLS].to_numpy(float)
        w = labels.loc[ids, WEIGHT_COLS].to_numpy(float)
        losses = weighted_soft_bce(y, p.to_numpy(), w)
        tables[key] = p
        rows.append({'run': key, 'n_validation': len(ids), 'folds': ','.join(map(str,sorted(folds))),
                     'macro_soft_bce': float(losses.mean()),
                     **{'bce_' + t: float(v) for t,v in zip(TARGETS, losses)}})
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(rows).sort_values('macro_soft_bce')
    result.to_csv(out / 'pseudo_validation_comparison.csv', index=False)
    correlations = []
    keys = list(tables)
    for i,a in enumerate(keys):
        for b in keys[i+1:]:
            for target in TARGETS:
                correlations.append({'run_a': a, 'run_b': b, 'target': target,
                    'spearman': tables[a][target].corr(tables[b][target], method='spearman')})
    pd.DataFrame(correlations).to_csv(out / 'prediction_correlations.csv', index=False)
    # Equal probability mean preserves probability scale for BCE; ranking is descriptive only.
    mean_p = sum(x.to_numpy() for x in tables.values()) / len(tables)
    ensemble = pd.DataFrame(mean_p, index=common_ids, columns=TARGETS)
    ensemble.index.name = UID
    ensemble.to_csv(out / 'equal_mean_validation_predictions.csv')
    report = {'n_runs': len(tables), 'validation_studies': len(common_ids),
              'supervision_id': meta['supervision_id'],
              'equal_mean_macro_soft_bce': float(weighted_soft_bce(y,mean_p,w).mean()),
              'selection_limit': 'Measures agreement with a report teacher; does not prove clinical or gold AUC gains.',
              'gold_used': False}
    write_json(out / 'comparison.json', report)
    return result


def gold_metrics(labels, probabilities, bootstrap=1000, seed=2026):
    """Descriptive historical-gold audit; report undefined targets rather than inflating macro."""
    y = np.asarray(labels, dtype=float)
    p = np.asarray(probabilities, dtype=float)
    if y.shape != p.shape or y.ndim != 2 or y.shape[1] != len(TARGETS):
        raise ValueError('Gold shapes must be [studies,12]')
    if not np.isin(y, [0,1]).all() or not np.isfinite(p).all():
        raise ValueError('Gold labels must be binary and predictions finite')
    rows = []
    for j,t in enumerate(TARGETS):
        valid = len(np.unique(y[:,j])) == 2
        rows.append({'target':t,'n_positive':int(y[:,j].sum()),'n':len(y),
                     'auc':float(roc_auc_score(y[:,j],p[:,j])) if valid else None,
                     'ap':float(average_precision_score(y[:,j],p[:,j])) if valid else None})
    macro = float(np.mean([r['auc'] for r in rows])) if all(r['auc'] is not None for r in rows) else None
    rng = np.random.default_rng(seed)
    draws=[]
    for _ in range(bootstrap):
        idx = rng.integers(0,len(y),len(y))
        if all(len(np.unique(y[idx,j])) == 2 for j in range(len(TARGETS))):
            draws.append(np.mean([roc_auc_score(y[idx,j],p[idx,j]) for j in range(len(TARGETS))]))
    interval = np.quantile(draws,[.025,.975]).tolist() if len(draws) >= 100 else None
    return pd.DataFrame(rows), {'macro_auc':macro,'bootstrap_95_percentile':interval,
        'bootstrap_valid_draws':len(draws),'bootstrap_requested':bootstrap,
        'limitation':'Repeated historical gold audit; interval does not correct adaptive model selection.'}


def run_gold_audit(config):
    if config.get('run_gold_audit') is not True:
        raise ValueError('Gold audit disabled. Freeze the recipe and explicitly enable once ready.')
    from .training import load_trained_model, predict_studies
    import torch
    recipe_path = Path(config['recipe_path'])
    recipe = json.loads(recipe_path.read_text())
    if recipe.get('frozen') is not True or not recipe.get('rationale'):
        raise ValueError('Complete and freeze the recipe before reading gold labels')
    checkpoints = recipe.get('checkpoints', [])
    if not checkpoints:
        raise ValueError('Frozen recipe needs explicit checkpoint paths and sha256')
    roles = pd.read_csv(Path(config['supervision_dir']) / 'study_roles.csv', dtype={UID: str})
    uids = sorted(roles.loc[roles.training_role == 'GOLD_EVAL_LOCKED',UID])
    if len(uids) != config.get('expected_gold',58):
        raise ValueError('Gold cohort size mismatch')
    cache_dir = Path(config['cache_dir'])
    cache = pd.read_csv(cache_dir / 'cache_index.csv', dtype={UID:str})
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    all_preds=[]
    for item in checkpoints:
        path = Path(item['path'])
        if hash_file(path) != item['sha256']:
            raise ValueError('Checkpoint differs from frozen recipe')
        model, trained_config = load_trained_model(path,device)
        pred = predict_studies(model,cache_dir,cache,uids,trained_config,device)
        pred = pred.set_index(UID).loc[uids,TARGETS]
        all_preds.append(pred.to_numpy())
    p = np.mean(all_preds,axis=0)
    labels = pd.read_csv(resolve_comp_dir(config.get('comp_dir')) / 'train.csv',dtype={UID:str}).set_index(UID)
    y = labels.loc[uids,TARGETS].to_numpy(float)
    # Bootstrap by groups if gold includes repeated groups: never mislabel a study bootstrap.
    gold_roles = roles.set_index(UID).loc[uids]
    if gold_roles.cv_group_id.duplicated().any():
        raise ValueError('Repeated gold groups require group bootstrap; resolve before this audit')
    metrics, report = gold_metrics(y,p,config.get('bootstrap',1000))
    out=Path(config['out_dir']); out.mkdir(parents=True,exist_ok=True)
    metrics.to_csv(out / 'gold_metrics.csv',index=False)
    table=pd.DataFrame(p,columns=TARGETS); table.insert(0,UID,uids)
    table.to_csv(out / 'gold_predictions.csv',index=False)
    report.update(recipe_sha256=hash_file(recipe_path),n_studies=len(uids),gold_used_for_training=False)
    write_json(out / 'gold_audit.json',report)
    return metrics,report
