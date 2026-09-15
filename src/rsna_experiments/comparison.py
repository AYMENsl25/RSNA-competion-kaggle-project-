"""Strict, exploratory comparisons of saved report-label validation predictions.

No training imports, ensemble fitting, class filling, or leaderboard inference.
"""
import csv
import hashlib
import io
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .common import TARGETS, UID, WC, YC, fingerprint, safe_output, sha256, write_json

PC = [f'p_{j}' for j in range(12)]
ROLE = 'report_label_exploration'
METRICS = ['macro_bce', 'macro_auc', 'eligible_auc_targets',
           'constant_macro_bce', 'delta_vs_constant']
SUMMARY_KEYS = ['run_id', 'experiment', 'validation_id', 'cohort_id',
                'supervision_id', 'cache_id', 'best_epoch', 'wall_seconds', 'role']
PAIR_COLUMNS = ['reference_run_id', 'run_id', 'delta_macro_auc', 'ci_low', 'ci_high',
                'bootstrap_requested', 'bootstrap_valid', 'bootstrap_discarded']
CORRELATION_COLUMNS = ['run_a', 'run_b', 'target_index', 'target', 'n_rows',
                       'pearson_r', 'reason']


def _load_run(directory):
    directory = Path(directory)
    # Hash precisely the bytes parsed, including when a producer is still running.
    summary_bytes = (directory / 'run_summary.json').read_bytes()
    prediction_bytes = (directory / 'validation_predictions.csv').read_bytes()
    summary = json.loads(summary_bytes.decode('utf-8-sig'))
    if not isinstance(summary, dict):
        raise ValueError(f'{directory}: run_summary must be an object')
    missing = set(SUMMARY_KEYS + ['active_targets', 'metrics']) - summary.keys()
    if missing:
        raise ValueError(f'{directory}: missing summary keys {sorted(missing)}')
    if not isinstance(summary['metrics'], dict) or set(METRICS) - summary['metrics'].keys():
        raise ValueError(f'{directory}: incomplete metrics')
    for key in ['run_id', 'validation_id', 'cohort_id']:
        if not isinstance(summary[key], str) or not summary[key].strip():
            raise ValueError(f'{directory}: invalid {key}')
    active = summary['active_targets']
    if not isinstance(active, list) or len(active) != 12 or any(type(v) is not bool for v in active):
        raise ValueError(f'{directory}: active_targets must contain 12 booleans')
    if summary['role'] != ROLE:
        raise ValueError(f'{directory}: role must be {ROLE}')
    prediction_hash = hashlib.sha256(prediction_bytes).hexdigest()
    if ('validation_predictions_sha256' in summary
            and summary['validation_predictions_sha256'] != prediction_hash):
        raise ValueError(f'{directory}: validation prediction checksum mismatch')

    decoded = prediction_bytes.decode('utf-8-sig')
    header = next(csv.reader(io.StringIO(decoded)), [])
    required = [UID, 'group', 'fold'] + YC + WC + PC
    if len(header) != len(set(header)) or set(required) - set(header):
        raise ValueError(f'{directory}: missing or duplicate prediction columns')
    frame = pd.read_csv(io.StringIO(decoded), dtype={UID: str, 'group': str},
                        keep_default_na=False, float_precision='round_trip')
    if frame.empty or frame[UID].duplicated().any():
        raise ValueError(f'{directory}: validation UIDs must be nonempty and unique')
    if any(frame[key].astype(str).str.strip().eq('').any() for key in [UID, 'group', 'fold']):
        raise ValueError(f'{directory}: missing UID, group, or fold')
    for key in ['fold'] + YC + WC + PC:
        frame[key] = pd.to_numeric(frame[key], errors='raise')
    values = frame[['fold'] + YC + WC + PC].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f'{directory}: nonfinite prediction data')
    if (frame['fold'] < 0).any() or (frame['fold'] % 1 != 0).any():
        raise ValueError(f'{directory}: fold must be a nonnegative integer')
    if (frame.groupby('group')['fold'].nunique() > 1).any():
        raise ValueError(f'{directory}: a group spans multiple folds')
    if (frame[WC].to_numpy() < 0).any():
        raise ValueError(f'{directory}: negative weights')
    for columns, name in [(YC, 'labels'), (PC, 'probabilities')]:
        a = frame[columns].to_numpy()
        if ((a < 0) | (a > 1)).any():
            raise ValueError(f'{directory}: {name} must be in [0, 1]')
    return summary, frame.set_index(UID).sort_index(), {
        'run_id': summary['run_id'],
        'run_summary_sha256': hashlib.sha256(summary_bytes).hexdigest(),
        'validation_predictions_sha256': prediction_hash,
    }


def _macro_auc(y, w, p, eligible):
    """Return None if ANY fixed eligible target degenerates in this sample."""
    aucs = []
    for j in eligible:
        mask = (w[:, j] > 0) & np.isin(y[:, j], [0, 1])
        if np.unique(y[mask, j]).size != 2:
            return None
        # Weights gate observation, as in training's AUC; they do not weight ROC.
        aucs.append(float(roc_auc_score(y[mask, j], p[mask, j])))
    return float(np.mean(aucs)) if aucs else None


def _metrics(y, w, p, active, eligible):
    p_bce = np.clip(p, 1e-7, 1 - 1e-7)
    bces = []
    for j in np.flatnonzero(active):
        mask = w[:, j] > 0
        if mask.any():
            loss = -y[mask, j] * np.log(p_bce[mask, j]) - (1 - y[mask, j]) * np.log1p(-p_bce[mask, j])
            bces.append(float(np.average(loss, weights=w[mask, j])))
    return {'macro_bce': float(np.mean(bces)) if bces else None,
            'macro_auc': _macro_auc(y, w, p, eligible),
            'eligible_auc_targets': [TARGETS[j] for j in eligible],
            'n_auc_targets': len(eligible)}


def _bootstrap_pairs(y, w, predictions, groups, eligible, summaries, measured, bootstrap, seed):
    samples = [[] for _ in predictions[1:]]
    unique_groups = np.unique(groups)
    members = [np.flatnonzero(groups == group) for group in unique_groups]
    rng = np.random.default_rng(seed)
    if eligible:
        for _ in range(bootstrap):
            # Keep every row in each drawn group, including repeated group draws.
            rows = np.concatenate([members[k] for k in rng.integers(len(members), size=len(members))])
            scores = [_macro_auc(y[rows], w[rows], p[rows], eligible) for p in predictions]
            if any(score is None for score in scores):
                continue
            for deltas, score in zip(samples, scores[1:]):
                deltas.append(score - scores[0])
    pairs = []
    for i, deltas in enumerate(samples, start=1):
        interval = np.quantile(deltas, [.025, .975]).tolist() if deltas else [None, None]
        pairs.append(dict(reference_run_id=summaries[0]['run_id'], run_id=summaries[i]['run_id'],
                          delta_macro_auc=(measured[i]['macro_auc'] - measured[0]['macro_auc']) if eligible else None,
                          ci_low=interval[0], ci_high=interval[1], bootstrap_requested=bootstrap,
                          bootstrap_valid=len(deltas), bootstrap_discarded=bootstrap - len(deltas)))
    return pairs


def compare_runs(run_dirs, out_dir, bootstrap=200, seed=42) -> Path:
    """Write CSV reports, an equal-probability ensemble, and comparison.json.

    Requires at least two distinct runs with identical validation/cohort IDs,
    active targets, UIDs and aligned labels, weights, groups and folds. Training
    supervision/cache IDs may differ and are retained in the summary. All inputs
    are checked before output creation. Returns the output directory.

    Correlations are Pearson correlations on all validation rows, separately for
    each active target. Inactive/constant vectors produce None (blank in CSV).
    AUC uses positive-weight binary labels and a fixed eligible target set.
    Intervals are paired group-bootstrap 95% percentile intervals for each run
    minus the first run; a draw losing any eligible class is discarded in full.
    Zero bootstrap draws disables intervals. Ensemble probabilities are the
    arithmetic mean, with no ranking, fitting, or validation-prior estimation.
    """
    for value, name in [(bootstrap, 'bootstrap'), (seed, 'seed')]:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
            raise ValueError(f'{name} must be a nonnegative integer')
    bootstrap, seed = int(bootstrap), int(seed)
    loaded = [_load_run(path) for path in run_dirs]
    if len(loaded) < 2:
        raise ValueError('At least two runs are required')
    summaries, frames, inputs = map(list, zip(*loaded))
    if len({s['run_id'] for s in summaries}) != len(summaries):
        raise ValueError('Duplicate run_id')
    reference = frames[0]
    for summary, frame in zip(summaries[1:], frames[1:]):
        for key in ['validation_id', 'cohort_id', 'active_targets']:
            if summary[key] != summaries[0][key]:
                raise ValueError(f"{summary['run_id']}: {key} mismatch")
        if not reference.index.equals(frame.index):
            raise ValueError(f"{summary['run_id']}: validation UIDs mismatch")
        for key in ['group', 'fold'] + YC + WC:
            if not np.array_equal(reference[key].to_numpy(), frame[key].to_numpy()):
                raise ValueError(f"{summary['run_id']}: aligned {key} mismatch")

    y, w = (reference[columns].to_numpy(dtype=float) for columns in [YC, WC])
    active = np.asarray(summaries[0]['active_targets'], dtype=bool)
    eligible = [int(j) for j in np.flatnonzero(active)
                if np.unique(y[(w[:, j] > 0) & np.isin(y[:, j], [0, 1]), j]).size == 2]
    predictions = [frame[PC].to_numpy(dtype=float) for frame in frames]
    measured = [_metrics(y, w, p, active, eligible) for p in predictions]
    rows = []
    for summary, metrics in zip(summaries, measured):
        row = {key: summary[key] for key in SUMMARY_KEYS}
        row.update({key: summary['metrics'][key] for key in METRICS})
        row.update(reported_macro_bce=summary['metrics']['macro_bce'],
                   reported_macro_auc=summary['metrics']['macro_auc'])
        row.update(metrics)
        rows.append(row)
    correlations = []
    for a, b in combinations(range(len(predictions)), 2):
        for j, target in enumerate(TARGETS):
            x, z = predictions[a][:, j], predictions[b][:, j]
            reason = 'inactive_target' if not active[j] else (
                'constant_predictions' if np.ptp(x) == 0 or np.ptp(z) == 0 else None)
            correlation = None
            if reason is None:
                # Scaling prevents underflow for very small but nonconstant probabilities.
                correlation = float(np.corrcoef((x - x.min()) / np.ptp(x),
                                               (z - z.min()) / np.ptp(z))[0, 1])
            correlations.append(dict(run_a=summaries[a]['run_id'], run_b=summaries[b]['run_id'],
                                     target_index=j, target=target, n_rows=len(x),
                                     pearson_r=correlation, reason=reason))
    pairs = _bootstrap_pairs(y, w, predictions, reference['group'].to_numpy(), eligible,
                             summaries, measured, bootstrap, seed)
    ensemble_p = np.mean(predictions, axis=0)
    ensemble = {'method': 'equal_probability_mean', 'exploratory_only': True,
                'run_ids': [s['run_id'] for s in summaries],
                'metrics': _metrics(y, w, ensemble_p, active, eligible)}
    identity_payload = dict(schema_version=1, kind='experiment_comparison', inputs=inputs,
                            bootstrap=bootstrap, seed=seed, role=ROLE)
    identity = fingerprint(identity_payload)
    receipt = dict(identity_payload, comparison_id=identity, exploratory_only=True,
                   validation_id=summaries[0]['validation_id'], cohort_id=summaries[0]['cohort_id'],
                   active_targets=active.tolist(), eligible_auc_targets=[TARGETS[j] for j in eligible],
                   validation_rows=len(reference), validation_groups=reference['group'].nunique(),
                   summary=rows, correlations=correlations, pairwise=pairs, ensemble=ensemble,
                   methodology={
                       'delta_direction': 'run_minus_first_run', 'confidence_level': .95,
                       'interval': 'paired_group_bootstrap_percentile',
                       'degenerate_replicates': 'discard_entire_macro_replicate_no_class_filling',
                       'auc': 'unweighted_ROC_on_positive_weight_binary_labels_fixed_active_targets',
                       'bce': 'mean_of_per_active_target_weighted_BCE_including_soft_labels',
                       'correlation': 'per_active_target_Pearson_all_validation_rows',
                       'summary_metrics': 'BCE_and_AUC_recomputed_reported_values_retained',
                       'constant_baselines': 'reported_per_run_training_baselines_no_ensemble_baseline',
                       'claims': 'report_label_exploration_only_no_tuning_or_leaderboard_claim',
                   })
    # Validate serializability before claiming an output directory.
    fingerprint(receipt)
    out = safe_output(out_dir, identity)
    summary_csv = pd.DataFrame(rows)
    summary_csv['eligible_auc_targets'] = summary_csv['eligible_auc_targets'].map(json.dumps)
    summary_csv.to_csv(out / 'summary.csv', index=False)
    pd.DataFrame(pairs, columns=PAIR_COLUMNS).to_csv(out / 'pairwise.csv', index=False)
    pd.DataFrame(correlations, columns=CORRELATION_COLUMNS).to_csv(out / 'pairwise_correlations.csv', index=False)
    ensemble_frame = reference[['group', 'fold'] + YC + WC].copy()
    ensemble_frame[PC] = ensemble_p
    ensemble_frame.reset_index().to_csv(out / 'ensemble_predictions.csv', index=False)
    receipt['outputs'] = {name: sha256(out / name) for name in
                          ['summary.csv', 'pairwise.csv', 'pairwise_correlations.csv', 'ensemble_predictions.csv']}
    write_json(out / 'comparison.json', receipt)
    return out
