"""Two bounded CPU hypotheses using immutable F2 features; no Kaggle execution.

Ranking uses a simple pairwise logistic surrogate, not the cited DAM algorithm.
Dropout removes whole series only during training, retaining at least one series.
Both are exploratory on the previously used proxy fold, not F0 comparisons.
"""
import argparse
import csv
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import torch
from f2_data import sha
from f2_model import Head, masked_bce
spec = importlib.util.spec_from_file_location('f2_runner', ROOT / 'notebooks/f2_dino_geometry/run.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def ranking_loss(logits, target, observed):
    """Equal weight per supported finding; unobserved cells never form pairs."""
    losses = []
    for j in range(logits.shape[1]):
        pos = logits[observed[:, j] & (target[:, j] == 1), j]
        neg = logits[observed[:, j] & (target[:, j] == 0), j]
        if len(pos) and len(neg):
            losses.append(torch.nn.functional.softplus(neg[None, :] - pos[:, None]).mean())
    return torch.stack(losses).mean() if losses else logits.sum() * 0


def drop_series(valid, probability, generator):
    present = valid.any(-1)
    if not present.any(-1).all():
        raise ValueError('empty study')
    keep = (torch.rand(present.shape, generator=generator) >= probability) & present
    # Uniform fallback among available slots avoids favoring one plane.
    empty = ~keep.any(-1)
    if empty.any():
        choices = torch.multinomial(present[empty].float(), 1, generator=generator)
        rows = torch.where(empty)[0]
        keep[rows, choices[:, 0]] = True
    return valid & keep[..., None]


def worker(arm, out, trial):
    start = time.monotonic()
    torch.set_num_threads(trial['cpu_threads_per_worker'])
    parent = ROOT / trial['parent']
    cfg = json.loads((parent / 'config.json').read_text())
    complete = json.loads((parent / 'features_complete.json').read_text())
    if complete['source_sha256'] != runner.source_hashes() or complete['feature_sha256'] != sha(parent / 'features.npy'):
        raise ValueError('parent source or features changed')
    prepared = runner.prepare(cfg)
    uids, partitions, values = prepared[:3]
    if not prepared[-1]['failure_gate_passed'] or uids != json.loads((parent / 'private_uid_order.json').read_text()):
        raise ValueError('cohort/QC mismatch')
    data = tuple(torch.from_numpy(np.load(parent / f'{name}.npy').astype(dtype))
                 for name, dtype in [('features', np.float32), ('valid', bool), ('geometry', np.float32)])
    y = torch.from_numpy(np.stack([values[u][0] for u in uids]))
    obs = torch.from_numpy(np.stack([values[u][1] for u in uids]))
    tr = np.array([i for i, u in enumerate(uids) if partitions[u] == 'candidate_train'])
    va = np.array([i for i, u in enumerate(uids) if partitions[u] == 'candidate_validation'])
    torch.manual_seed(cfg['seed'])
    base = Head(width=cfg['head_width'])
    model = Head(width=cfg['head_width'], geometry=True)
    model.load_state_dict(base.state_dict(), strict=False)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg['epochs'])
    history, best = [], None
    runner.write_json(out / 'provenance.json', {
        'arm': arm, 'parent_config': cfg, 'trial_config': trial,
        'sources': {**runner.source_hashes(), 'f4_run.py': sha(__file__)},
        'feature_sha256': complete['feature_sha256'], 'inputs': prepared[-1],
        'mask_sha256': sha(parent / 'valid.npy'), 'geometry_sha256': sha(parent / 'geometry.npy'),
        'python': sys.version, 'torch': str(torch.__version__), 'numpy': np.__version__,
        'platform': platform.platform(), 'device': 'cpu', 'dtype': 'float32',
        'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()})
    for epoch in range(1, cfg['epochs'] + 1):
        torch.manual_seed(cfg['seed'] + epoch)
        generator = torch.Generator().manual_seed(cfg['seed'] + epoch + 10000)
        order = np.random.default_rng(cfg['seed'] + epoch).permutation(tr)
        model.train()
        for offset in range(0, len(order), cfg['head_batch']):
            if time.monotonic() - start > trial['hard_wall_seconds'] - 60:
                raise TimeoutError('CPU trial deadline')
            ii = order[offset:offset + cfg['head_batch']]
            f, m, g = (x[ii] for x in data)
            if arm == 'series_dropout':
                m = drop_series(m, trial['series_drop_probability'], generator)
            logits = model(f, m, g)
            loss = masked_bce(logits, y[ii], obs[ii])
            if arm == 'ranking':
                loss = loss + trial['ranking_weight'] * ranking_loss(logits, y[ii], obs[ii])
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite loss')
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            opt.step()
        sched.step()
        p, metric = runner.evaluate(model, data, va, y, obs, cfg['head_batch'])
        record = {'epoch': epoch, **metric}
        history.append(record)
        key = (-metric['macro_auc'], metric['masked_bce'], epoch)
        if best is None or key < best[0]:
            best = key, record, p.copy()
            torch.save({'state_dict': model.state_dict(), 'epoch': epoch, 'arm': arm,
                        'config': cfg, 'trial': trial}, out / 'best.pt')
        runner.write_json(out / 'history.json', history)
        print(json.dumps({'arm': arm, 'epoch': epoch, 'auc': metric['macro_auc']}), flush=True)
    p = best[2]
    np.save(out / 'validation_probabilities.npy', p, allow_pickle=False)
    model.load_state_dict(torch.load(out / 'best.pt', weights_only=True)['state_dict'])
    reproduced, _ = runner.evaluate(model, data, va, y, obs, cfg['head_batch'])
    if not np.array_equal(p, reproduced):
        raise ValueError('checkpoint parity failed')
    baseline = np.load(parent / 'f2_B_val_probabilities.npy')
    target, observed = y[va].numpy(), obs[va].numpy()
    with (ROOT / cfg['split']).open(newline='', encoding='utf-8-sig') as f:
        groups = {r['StudyInstanceUID']: r['group_id'] for r in csv.DictReader(f)}
    gu = np.array([groups[uids[i]] for i in va])
    members = [np.flatnonzero(gu == g) for g in sorted(set(gu))]
    rng, differences = np.random.default_rng(trial['seed']), []
    for _ in range(trial['bootstrap_draws']):
        ii = np.concatenate([members[j] for j in rng.integers(len(members), size=len(members))])
        try:
            differences.append(runner.metrics(target[ii], observed[ii], p[ii])['macro_auc'] - runner.metrics(target[ii], observed[ii], baseline[ii])['macro_auc'])
        except ValueError:
            continue
    baseline_metric = runner.metrics(target, observed, baseline)
    result = {'status': 'complete', 'arm': arm, 'selected': best[1],
              'baseline': baseline_metric, 'delta': best[1]['macro_auc'] - baseline_metric['macro_auc'],
              'exploratory_group_bootstrap_95pct': np.percentile(differences, [2.5, 97.5]).tolist(),
              'bootstrap_valid_draws': len(differences), 'checkpoint_exact_parity': True,
              'checkpoint_sha256': sha(out / 'best.pt'), 'predictions_sha256': sha(out / 'validation_probabilities.npy'),
              'elapsed_seconds': time.monotonic() - start, 'kaggle_gpu_hours_used': 0,
              'limitation': 'reused proxy fold and epoch selection; not independent confirmation or incremental F0 evidence'}
    runner.write_json(out / 'result.json', result)
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', choices=['ranking', 'series_dropout'])
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    trial = json.loads(Path(__file__).with_name('config.json').read_text())
    if args.worker:
        worker(args.worker, args.out, trial)
        return
    out = ROOT / 'outputs/f4_cpu_trials' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out.mkdir(parents=True, exist_ok=False)
    runner.write_json(out / 'config.json', trial)
    print(str(out), flush=True)
    processes = []
    start = time.monotonic()
    try:
        for arm in trial['arms']:
            folder = out / arm
            folder.mkdir()
            log = (folder / 'stdout.log').open('w', encoding='utf-8')
            command = [sys.executable, str(Path(__file__).resolve()), '--worker', arm, '--out', str(folder)]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', PYTHONUNBUFFERED='1')
            proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
            processes.append((arm, proc, log))
        codes = {}
        for arm, proc, _ in processes:
            codes[arm] = proc.wait(timeout=max(1, trial['hard_wall_seconds'] - (time.monotonic() - start)))
        runner.write_json(out / 'supervisor.json', {'exit_codes': codes, 'elapsed_seconds': time.monotonic() - start})
        if any(codes.values()):
            raise SystemExit(1)
    finally:
        for _, proc, log in processes:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
            log.close()


if __name__ == '__main__':
    main()
