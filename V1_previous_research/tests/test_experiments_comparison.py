"""Small synthetic comparisons; no competition data or training imports."""
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from rsna_experiments.common import TARGETS, UID, WC, YC, sha256
from rsna_experiments.comparison import PC, compare_runs


def make_run(tmp_path, name, frame=None, **overrides):
    if frame is None:
        # Unequal group sizes expose bootstrap implementations that sample rows.
        frame = pd.DataFrame({UID: [f'{i:03d}' for i in range(6)],
                              'group': ['a', 'a', 'a', 'b', 'c', 'd'], 'fold': [0] * 6})
        for j in range(12):
            frame[f'y_{j}'] = [0, 1, 0, 1, 0, 1]
            frame[f'w_{j}'] = 1.
            frame[f'p_{j}'] = [.1, .7, .3, .9, .2, .8]
    summary = dict(run_id=name, experiment=name, validation_id='validation', cohort_id='cohort',
                   supervision_id=f'supervision-{name}', cache_id=f'cache-{name}', best_epoch=2,
                   metrics=dict(macro_bce=.2, macro_auc=1., eligible_auc_targets=[TARGETS[0]],
                                constant_macro_bce=.7, delta_vs_constant=-.5),
                   active_targets=[True] + [False] * 11, wall_seconds=10.,
                   role='report_label_exploration')
    summary.update(overrides)
    directory = tmp_path / name
    directory.mkdir()
    (directory / 'run_summary.json').write_text(json.dumps(summary), encoding='utf-8')
    frame.to_csv(directory / 'validation_predictions.csv', index=False)
    return directory


def read_frame(directory):
    return pd.read_csv(directory / 'validation_predictions.csv', dtype={UID: str}, float_precision='round_trip')


def read_receipt(out):
    return json.loads((out / 'comparison.json').read_text())


def test_alignment_summary_correlations_and_probability_ensemble(tmp_path):
    a = make_run(tmp_path, 'a')
    frame = read_frame(a)
    frame['p_0'] = [.8, .1, .7, .2, .6, .3]
    b = make_run(tmp_path, 'b', frame.iloc[::-1])
    constant = frame.copy()
    constant['p_0'] = .5
    c = make_run(tmp_path, 'c', constant)
    out = compare_runs([a, b, c], tmp_path / 'out', bootstrap=15)
    receipt = read_receipt(out)
    assert receipt['exploratory_only'] and receipt['role'] == 'report_label_exploration'
    assert receipt['eligible_auc_targets'] == [TARGETS[0]]
    assert receipt['ensemble']['method'] == 'equal_probability_mean'
    predicted = pd.read_csv(out / 'ensemble_predictions.csv', dtype={UID: str}, float_precision='round_trip')
    np.testing.assert_allclose(predicted[PC], (read_frame(a)[PC] + frame[PC] + constant[PC]) / 3)
    assert predicted[UID].tolist() == read_frame(a)[UID].tolist()
    active = [r for r in receipt['correlations'] if r['target_index'] == 0]
    assert len(active) == 3  # All pairs, including b versus c.
    assert active[0]['pearson_r'] == pytest.approx(np.corrcoef(read_frame(a)['p_0'], frame['p_0'])[0, 1])
    assert active[1]['pearson_r'] is None and active[1]['reason'] == 'constant_predictions'
    assert receipt['correlations'][1]['reason'] == 'inactive_target'
    expected_auc = roc_auc_score(predicted['y_0'], predicted['p_0'])
    assert receipt['ensemble']['metrics']['macro_auc'] == expected_auc
    expected_bce = np.mean(-predicted['y_0'] * np.log(predicted['p_0'])
                           - (1 - predicted['y_0']) * np.log1p(-predicted['p_0']))
    assert receipt['ensemble']['metrics']['macro_bce'] == pytest.approx(expected_bce)
    assert len(receipt['pairwise']) == 2
    assert receipt['pairwise'][0]['delta_macro_auc'] == -1.
    summary = pd.read_csv(out / 'summary.csv')
    assert summary['macro_auc'].tolist() == [1., 0., .5]
    assert summary['reported_macro_auc'].tolist() == [1., 1., 1.]
    for name, digest in receipt['outputs'].items():
        assert sha256(out / name) == digest


@pytest.mark.parametrize('key', ['validation_id', 'cohort_id', 'active_targets'])
def test_identity_mismatch_rejected_without_output(tmp_path, key):
    a = make_run(tmp_path, 'a')
    value = [True, True] + [False] * 10 if key == 'active_targets' else 'different'
    b = make_run(tmp_path, 'b', **{key: value})
    with pytest.raises(ValueError, match=key):
        compare_runs([a, b], tmp_path / 'out')
    assert not (tmp_path / 'out').exists()


@pytest.mark.parametrize('key,value', [(UID, 'other'), ('group', 'other'), ('fold', 1),
                                      ('y_0', .25), ('y_11', .25), ('w_0', .1), ('w_11', .1)])
def test_actual_aligned_data_tampering_rejected(tmp_path, key, value):
    a = make_run(tmp_path, 'a')
    frame = read_frame(a)
    frame[key] = frame[key].astype(object)
    frame.loc[5, key] = value
    b = make_run(tmp_path, 'b', frame.iloc[::-1])
    with pytest.raises(ValueError, match='mismatch'):
        compare_runs([a, b], tmp_path / 'out', bootstrap=1)
    assert not (tmp_path / 'out').exists()


def test_fixed_targets_soft_labels_weights_and_group_bootstrap(tmp_path):
    base = make_run(tmp_path, 'base')
    frame = read_frame(base)
    frame['y_1'] = [.25, .75, .5, 0, 0, 1]  # Only d carries the positive class.
    frame['w_1'] = [.1, .5, 1., 1., 0., 1.]
    frame['y_2'] = 1  # Active but globally ineligible.
    active = [True] * 3 + [False] * 9
    a = make_run(tmp_path, 'a', frame, active_targets=active)
    other = frame.copy()
    other['p_0'] = [.7, .3, .2, .6, .9, .1]
    other['p_1'] = [.6, .2, .5, .9, .3, .1]
    b = make_run(tmp_path, 'b', other, active_targets=active)
    out = compare_runs([a, b], tmp_path / 'out', bootstrap=60, seed=7)
    receipt = read_receipt(out)
    assert receipt['eligible_auc_targets'] == TARGETS[:2]
    # Independent literal grouped resampling oracle, with both models sharing rows.
    rng = np.random.default_rng(7)
    groups = [np.flatnonzero(frame['group'].to_numpy() == g) for g in sorted(frame['group'].unique())]
    deltas = []
    for _ in range(60):
        rows = np.concatenate([groups[i] for i in rng.integers(4, size=4)])
        left, right = [], []
        for j in [0, 1]:
            block = frame.iloc[rows]
            mask = (block[f'w_{j}'].to_numpy() > 0) & np.isin(block[f'y_{j}'], [0, 1])
            labels = block[f'y_{j}'].to_numpy()[mask]
            if len(np.unique(labels)) != 2:
                break
            left.append(roc_auc_score(labels, frame.iloc[rows][f'p_{j}'].to_numpy()[mask]))
            right.append(roc_auc_score(labels, other.iloc[rows][f'p_{j}'].to_numpy()[mask]))
        if len(left) == 2:
            deltas.append(np.mean(right) - np.mean(left))
    pair = receipt['pairwise'][0]
    assert 0 < len(deltas) < 60
    assert pair['bootstrap_valid'] == len(deltas)
    assert pair['bootstrap_discarded'] == 60 - len(deltas)
    np.testing.assert_allclose([pair['ci_low'], pair['ci_high']], np.quantile(deltas, [.025, .975]))
    p = (frame[PC].to_numpy() + other[PC].to_numpy()) / 2
    y, w = frame[YC].to_numpy(), frame[WC].to_numpy()
    expected_bce = np.mean([np.average(-y[:, j] * np.log(p[:, j])
                                      - (1 - y[:, j]) * np.log1p(-p[:, j]), weights=w[:, j])
                            for j in range(3)])
    assert receipt['ensemble']['metrics']['macro_bce'] == pytest.approx(expected_bce)
    assert read_receipt(compare_runs([a, b], out, bootstrap=60, seed=7)) == receipt


def test_identical_runs_have_zero_paired_interval(tmp_path):
    a, b = make_run(tmp_path, 'a'), make_run(tmp_path, 'b')
    pair = read_receipt(compare_runs([a, b], tmp_path / 'out', bootstrap=20))['pairwise'][0]
    assert pair['bootstrap_valid'] > 0
    assert pair['delta_macro_auc'] == pair['ci_low'] == pair['ci_high'] == 0.


@pytest.mark.parametrize('bootstrap', [0, 10])
def test_no_auc_eligible_targets(tmp_path, bootstrap):
    base = make_run(tmp_path, 'base')
    frame = read_frame(base)
    frame['y_0'] = .5
    a, b = make_run(tmp_path, 'a', frame), make_run(tmp_path, 'b', frame)
    result = read_receipt(compare_runs([a, b], tmp_path / 'out', bootstrap=bootstrap))
    assert result['eligible_auc_targets'] == []
    assert result['ensemble']['metrics']['macro_auc'] is None
    assert result['ensemble']['metrics']['macro_bce'] is not None
    pair = result['pairwise'][0]
    assert pair['bootstrap_valid'] == 0
    assert pair['delta_macro_auc'] is pair['ci_low'] is pair['ci_high'] is None


def test_safe_output_and_content_fingerprint(tmp_path):
    a, b = make_run(tmp_path, 'a'), make_run(tmp_path, 'b')
    unrelated = tmp_path / 'unrelated'
    unrelated.mkdir()
    (unrelated / 'summary.csv').write_text('keep me')
    with pytest.raises(ValueError, match='Unrecognized'):
        compare_runs([a, b], unrelated, bootstrap=0)
    assert (unrelated / 'summary.csv').read_text() == 'keep me'
    out = compare_runs([a, b], tmp_path / 'out', bootstrap=0)
    original = (out / 'comparison.json').read_bytes()
    for runs, kwargs in [([b, a], {}), ([a, b], {'seed': 1}), ([a, b], {'bootstrap': 1})]:
        with pytest.raises(ValueError, match='different experiment'):
            compare_runs(runs, out, **({'bootstrap': 0} | kwargs))
    frame = read_frame(b)
    frame.loc[0, 'p_0'] = .25
    frame.to_csv(b / 'validation_predictions.csv', index=False)
    with pytest.raises(ValueError, match='different experiment'):
        compare_runs([a, b], out, bootstrap=0)
    assert (out / 'comparison.json').read_bytes() == original


@pytest.mark.parametrize('case', ['duplicate_uid', 'empty', 'missing_column', 'duplicate_column',
                                 'nan', 'probability', 'negative_weight', 'group_folds', 'checksum'])
def test_malformed_predictions(tmp_path, case):
    a = make_run(tmp_path, 'a')
    frame = read_frame(a)
    if case == 'duplicate_uid':
        frame.loc[1, UID] = frame.loc[0, UID]
    elif case == 'empty':
        frame = frame.iloc[:0]
    elif case == 'missing_column':
        frame = frame.drop(columns='p_11')
    elif case == 'duplicate_column':
        frame = pd.concat([frame, frame[['p_0']]], axis=1)
    elif case == 'nan':
        frame.loc[0, 'p_0'] = np.inf
    elif case == 'probability':
        frame.loc[0, 'p_0'] = 1.1
    elif case == 'negative_weight':
        frame.loc[0, 'w_0'] = -1.
    elif case == 'group_folds':
        frame.loc[0, 'fold'] = 1
    b = make_run(tmp_path, 'b', frame, **({'validation_predictions_sha256': 'wrong'} if case == 'checksum' else {}))
    with pytest.raises(ValueError):
        compare_runs([a, b], tmp_path / 'out', bootstrap=0)
    assert not (tmp_path / 'out').exists()


@pytest.mark.parametrize('kwargs', [{'bootstrap': -1}, {'bootstrap': 1.5}, {'bootstrap': True},
                                   {'seed': -1}, {'seed': False}])
def test_invalid_arguments(tmp_path, kwargs):
    with pytest.raises(ValueError, match='nonnegative integer'):
        compare_runs([], tmp_path / 'out', **kwargs)


def test_duplicate_run_ids_and_insufficient_runs(tmp_path):
    a = make_run(tmp_path, 'a')
    with pytest.raises(ValueError, match='At least two'):
        compare_runs([a], tmp_path / 'out')
    with pytest.raises(ValueError, match='Duplicate run_id'):
        compare_runs([a, a], tmp_path / 'out')
