"""Study-level training with provenance, fixed validation, priors and resume."""
from pathlib import Path
import json
import math
import random
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              brier_score_loss, balanced_accuracy_score, f1_score)
from .common import UID, TARGETS, YC, WC, fingerprint, sha256, write_json, safe_output, environment
from .models import StudyModel
from .losses import MaskedObjective


class CacheDataset(Dataset):
    def __init__(self, root, frame, index, cfg, training=False):
        self.root, self.frame, self.cfg, self.training = Path(root), frame, cfg, training
        self.files = index.set_index(UID)['file'].to_dict()
        self.gamma = None
        if training and cfg['augmentation'] == 'intensity':
            from monai.transforms import RandAdjustContrast
            self.gamma = RandAdjustContrast(prob=.5, gamma=(.85, 1.15))
            self.gamma.set_random_state(seed=cfg['seed'])

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, i):
        row = self.frame.iloc[i]
        with np.load(self.root/self.files[row[UID]], allow_pickle=False) as data:
            images, mask = data['images'].astype(np.float32)/255, data['mask'].astype(bool)
        n = self.cfg['train_slices' if self.training else 'eval_slices']
        selected = np.zeros((3, n, *images.shape[-2:]), np.float32)
        selected_mask = np.zeros((3, n), bool)
        for plane in range(3):
            if self.cfg['planes'] == 'sagittal' and plane != 0:
                continue
            valid = np.flatnonzero(mask[plane])
            if len(valid) == 0:
                continue
            # Uniform bins with random within-bin sampling during training; fixed eval anchors.
            count = min(n, len(valid))
            if self.training:
                bounds = np.linspace(0, len(valid), count+1).astype(int)
                ids = np.array([np.random.randint(bounds[j], bounds[j+1]) for j in range(count)])
            else:
                ids = np.linspace(0, len(valid)-1, count).round().astype(int)
            selected[plane, :count] = images[plane, valid[ids]]
            selected_mask[plane, :count] = True
        x = torch.from_numpy(selected)
        if self.gamma is not None:
            # Shared intensity transform preserves slice order; no flips without laterality audit.
            x = torch.as_tensor(self.gamma(x)).float().clamp(0, 1)
            x *= torch.from_numpy(selected_mask)[..., None, None]
        return x, torch.from_numpy(selected_mask), torch.tensor(row[YC].to_numpy(float), dtype=torch.float32), torch.tensor(row[WC].to_numpy(float), dtype=torch.float32)


def metrics(y, w, p, active, prior=None):
    p = np.asarray(p, dtype=np.float64)
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError('Nonfinite or out-of-range predictions')
    p = np.clip(p, 1e-7, 1-1e-7)
    per_target, bces, aucs, aps, briers, balanced_accs, f1s = [], [], [], [], [], [], []
    for j, name in enumerate(TARGETS):
        observed = w[:, j] > 0
        binary = observed & np.isin(y[:, j], [0, 1])
        eligible = bool(active[j] and len(np.unique(y[binary, j])) == 2)
        bce = float(((-y[:, j]*np.log(p[:, j])-(1-y[:, j])*np.log1p(-p[:, j]))*w[:, j]).sum()/w[:, j].sum()) if active[j] and observed.any() else None
        auc = float(roc_auc_score(y[binary, j], p[binary, j])) if eligible else None
        ap = float(average_precision_score(y[binary, j], p[binary, j])) if eligible else None
        brier = float(brier_score_loss(y[binary, j], p[binary, j])) if eligible else None
        balanced_accuracy = (float(balanced_accuracy_score(y[binary, j], p[binary, j] >= 0.5)) if eligible else None)
        f1 = float(f1_score(y[binary, j], p[binary, j] >= 0.5, zero_division=0)) if eligible else None
        if bce is not None:
            bces.append(bce)
        if auc is not None:
            aucs.append(auc)
        if ap is not None: aps.append(ap)
        if brier is not None: briers.append(brier)
        if balanced_accuracy is not None: balanced_accs.append(balanced_accuracy)
        if f1 is not None: f1s.append(f1)
        per_target.append(dict(target=name, n_positive=int(((y[:, j] == 1)&observed).sum()),
                               n_negative=int(((y[:, j] == 0)&observed).sum()),
                               n_supervised=int(observed.sum()), train_active=bool(active[j]), bce=bce, auc=auc, ap=ap))
        per_target[-1].update(brier=brier, balanced_accuracy=balanced_accuracy, f1_at_05=f1)
    result = {'macro_bce': float(np.mean(bces)) if bces else None,
              'macro_auc': float(np.mean(aucs)) if aucs else None,
              'macro_ap': float(np.mean(aps)) if aps else None,
              'macro_brier': float(np.mean(briers)) if briers else None,
              'macro_balanced_accuracy': float(np.mean(balanced_accs)) if balanced_accs else None,
              'macro_f1_at_05': float(np.mean(f1s)) if f1s else None,
              'n_metric_targets': len(aucs),
              'eligible_auc_targets': [r['target'] for r in per_target if r['auc'] is not None],
              'n_auc_targets': len(aucs), 'per_target': per_target}
    if prior is not None:
        baseline = metrics(y, w, np.broadcast_to(prior, p.shape), active)
        result['constant_macro_bce'] = baseline['macro_bce']
        result['delta_vs_constant'] = result['macro_bce']-baseline['macro_bce'] if bces else None
    return result


def _save_checkpoint(path, obj):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    torch.save(obj, temp)
    temp.replace(path)


def run_training(cfg):
    cfg = dict(cfg)
    for field in ['epochs', 'batch_size', 'accumulate', 'train_slices', 'eval_slices']:
        if not isinstance(cfg[field], int) or isinstance(cfg[field], bool) or cfg[field] < 1:
            raise ValueError(f'{field} must be a positive integer')
    if cfg['workers'] != 0:
        raise ValueError('Use workers=0 for reproducible notebook resume and MONAI transform RNG.')
    if cfg['selection'] not in ['macro_bce', 'macro_auc']:
        raise ValueError('selection must be macro_bce or macro_auc')
    cache = Path(cfg['cache_dir'])
    meta = json.loads((cache/'cache_meta.json').read_text())
    if not meta['complete'] or meta.get('config', {}).get('max_studies', 0) > 0:
        counts = meta.get('counts', {})
        if meta.get('config', {}).get('max_studies', 0) > 0 or counts.get('unprocessed', 0):
            raise ValueError(
                'This is a preprocessing PILOT/truncated cache and cannot be trained. '
                'Rerun notebook 02 with PILOT=False, Save & Run All, and attach that full output. '
                f"Processed={counts.get('processed', '?')}, unprocessed={counts.get('unprocessed', '?')}."
            )
        raise ValueError(
            'Incomplete cache: inspect cache_audit.json and decode_errors.csv, fix the failures, '
            'then rebuild the full cache before training.'
        )
    frame = pd.read_csv(cache/'cache_manifest.csv', dtype={UID: str, 'group': str})
    index = pd.read_csv(cache/'cache_index.csv', dtype={UID: str})
    if sha256(cache/'cache_manifest.csv') != meta['manifest_sha256'] or sha256(cache/'cache_index.csv') != meta['index_sha256']:
        raise ValueError('Cache manifest/index hash mismatch')
    if frame[UID].duplicated().any() or index[UID].duplicated().any() or set(frame[UID]) != set(index[UID]):
        raise ValueError('Cache studies do not match the fixed cohort exactly')
    if frame[[UID, 'group', 'fold']].isna().any().any() or (frame.groupby('group').fold.nunique() != 1).any():
        raise ValueError('Missing identifiers or split group')
    values = frame[YC+WC].to_numpy(float)
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError('Labels and weights must be finite and within [0,1]')
    for _, row in index.iterrows():
        path = (cache/row['file']).resolve()
        if not path.is_relative_to(cache.resolve()) or not path.is_file() or sha256(path) != row['sha256']:
            raise ValueError(f'Missing or changed cache file for {row[UID]}')
        with np.load(path, allow_pickle=False) as data:
            images, mask = data['images'], data['mask']
            if images.ndim != 4 or images.shape[0] != 3 or mask.shape != images.shape[:2] or not mask.any():
                raise ValueError(f'Invalid cache shape/mask for {row[UID]}')
            if cfg['planes'] == 'sagittal' and not mask[0].any():
                raise ValueError('Sagittal-only comparison needs sagittal data for every fixed cohort study; cohort will not be silently filtered.')
    tr = frame[frame.fold != meta['fold']].sort_values(UID).reset_index(drop=True)
    va = frame[frame.fold == meta['fold']].sort_values(UID).reset_index(drop=True)
    if tr.empty or va.empty or set(tr.group)&set(va.group):
        raise ValueError('Empty train/validation or overlapping groups')
    if cfg['model'] == 'monai3d' and min(cfg['train_slices'], cfg['eval_slices']) < 16:
        raise ValueError('MONAI 3D needs at least 16 retained slice positions; set train_slices=16.')
    random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg['seed'])
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = StudyModel(cfg).to(device)
    train_y, train_w = torch.tensor(tr[YC].to_numpy(float), dtype=torch.float32), torch.tensor(tr[WC].to_numpy(float), dtype=torch.float32)
    objective = MaskedObjective(cfg, train_y, train_w).to(device)
    active = objective.active.cpu().numpy()
    prior = (train_y*train_w).sum(0)/train_w.sum(0).clamp_min(1e-8)
    prior[~torch.tensor(active)] = .5
    prior = prior.numpy().clip(1e-7, 1-1e-7)
    y, w = va[YC].to_numpy(float), va[WC].to_numpy(float)
    constant = metrics(y, w, np.broadcast_to(prior, y.shape), active)
    if constant['macro_bce'] is None:
        raise ValueError('No validation supervision for active training targets')
    if cfg['selection'] == 'macro_auc' and constant['n_auc_targets'] != 12:
        raise ValueError('AUC selection requires positive and negative validation examples for all 12 targets. Use BCE for exploratory sparse labels.')
    source_id = fingerprint({p.name: sha256(p) for p in Path(__file__).parent.glob('*.py')})
    identity_config = {k: v for k, v in cfg.items() if k not in ['out_dir', 'resume_path', 'cache_dir', 'weights_path', 'allow_downloads']}
    run_id = fingerprint({'config': identity_config, 'source_id': source_id,
                          'cache_id': meta['cache_id'], 'weights': model.weight_hash})
    out = safe_output(cfg['out_dir'], run_id)
    if (out/'run_summary.json').exists():
        print('Completed identical run already exists:', out)
        return out
    write_json(out/'config.json', cfg)
    validation_id = fingerprint(va[[UID, 'group', 'fold']+YC+WC].to_dict('records'))
    optimizer = torch.optim.AdamW([
        {'params': model.encoder.parameters(), 'lr': cfg['encoder_lr']},
        {'params': [p for name, p in model.named_parameters() if not name.startswith('encoder.')], 'lr': cfg['lr']}], weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['epochs'])
    scaler = torch.amp.GradScaler('cuda', enabled=cfg['amp'] and device.type == 'cuda')
    train_ds = CacheDataset(cache, tr, index, cfg, True)
    val_ds = CacheDataset(cache, va, index, cfg, False)
    loader_rng = torch.Generator().manual_seed(cfg['seed'])
    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True, generator=loader_rng, num_workers=0, pin_memory=device.type == 'cuda')
    val_loader = DataLoader(val_ds, batch_size=cfg['batch_size'], num_workers=0, pin_memory=device.type == 'cuda')
    start_epoch, stale, best, best_epoch, history = 0, 0, float('inf'), 0, []
    elapsed_before = 0.0
    resume = cfg.get('resume_path') or (str(out/'last.pt') if (out/'last.pt').exists() else None)
    best_predictions = None
    if resume:
        # Full optimizer/RNG checkpoint is trusted only from this user's experiment output.
        ck = torch.load(resume, map_location='cpu', weights_only=False)
        if ck['run_id'] != run_id:
            raise ValueError('Resume config/cache/source/weights mismatch. Use matching inputs or a new experiment.')
        model.load_state_dict(ck['model']); optimizer.load_state_dict(ck['optimizer'])
        scheduler.load_state_dict(ck['scheduler']); scaler.load_state_dict(ck['scaler'])
        start_epoch, stale, best, best_epoch, history = ck['epoch'], ck['stale'], ck['best'], ck['best_epoch'], ck['history']
        elapsed_before = ck['elapsed']
        torch.set_rng_state(ck['torch_rng']); np.random.set_state(ck['numpy_rng']); random.setstate(ck['python_rng'])
        loader_rng.set_state(ck['loader_rng'])
        if device.type == 'cuda' and ck['cuda_rng'] is not None:
            torch.cuda.set_rng_state_all(ck['cuda_rng'])
        if train_ds.gamma is not None:
            train_ds.gamma.R.set_state(ck['augmentation_rng'])
        best_predictions = ck['best_predictions']
        # A resume must carry the selected best model as well as last state.
        _save_checkpoint(out/'best.pt', ck['best_checkpoint'])
    else:
        ck = {}
    best_checkpoint = ck.get('best_checkpoint')
    print('Device:', device, 'train:', len(tr), 'validation:', len(va), 'constant macro BCE:', constant['macro_bce'], flush=True)
    print('Eligible AUC targets:', constant['n_auc_targets'], '/12. Labels remain report-derived.', flush=True)
    wall_start = time.perf_counter()
    for epoch in range(start_epoch, cfg['epochs']):
        if stale >= cfg['patience'] and epoch >= cfg['min_epochs']:
            break
        model.train(); model.set_epoch(epoch)
        optimizer.zero_grad(set_to_none=True)
        train_total = 0.0
        group_denominator = 1
        for step, (x, mask, target, weight) in enumerate(train_loader):
            if step % cfg['accumulate'] == 0:
                group_start = step*cfg['batch_size']
                group_denominator = min(cfg['accumulate']*cfg['batch_size'], len(tr)-group_start)
            x, mask, target, weight = x.to(device), mask.to(device), target.to(device), weight.to(device)
            with torch.autocast(device_type=device.type, enabled=cfg['amp'] and device.type == 'cuda'):
                logits = model(x, mask)
                raw_loss = objective(logits, target, weight)
                loss = raw_loss * len(x)/group_denominator
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            scaler.scale(loss).backward()
            train_total += raw_loss.item()*len(x)
            if (step+1) % cfg['accumulate'] == 0 or step+1 == len(train_loader):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        model.eval(); predictions = []
        with torch.no_grad():
            for x, mask, _, _ in val_loader:
                with torch.autocast(device_type=device.type, enabled=cfg['amp'] and device.type == 'cuda'):
                    pred = model(x.to(device), mask.to(device))
                predictions.append(pred.float().sigmoid().cpu().numpy())
        predictions = np.concatenate(predictions)
        measured = metrics(y, w, predictions, active, prior)
        criterion = measured['macro_bce'] if cfg['selection'] == 'macro_bce' else -measured['macro_auc']
        elapsed = elapsed_before + time.perf_counter()-wall_start
        record = dict(epoch=epoch+1, train_objective=train_total/len(tr), val_macro_bce=measured['macro_bce'],
                      val_macro_auc=measured['macro_auc'], constant_macro_bce=constant['macro_bce'],
                      delta_vs_constant=measured['delta_vs_constant'], seconds=elapsed)
        history.append(record); print(record, flush=True)
        if criterion < best:
            best, best_epoch, stale = criterion, epoch+1, 0
            best_predictions = predictions.copy()
            best_checkpoint = {'model': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                               'config': cfg, 'run_id': run_id, 'cache_metadata': meta, 'targets': TARGETS, 'epoch': best_epoch}
            _save_checkpoint(out/'best.pt', best_checkpoint)
        else:
            stale += 1
        _save_checkpoint(out/'last.pt', dict(run_id=run_id, model=model.state_dict(), optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(), scaler=scaler.state_dict(), epoch=epoch+1,
            best=best, best_epoch=best_epoch, stale=stale, history=history, elapsed=elapsed,
            best_predictions=best_predictions, best_checkpoint=best_checkpoint,
            torch_rng=torch.get_rng_state(), numpy_rng=np.random.get_state(), python_rng=random.getstate(),
            cuda_rng=torch.cuda.get_rng_state_all() if device.type == 'cuda' else None, loader_rng=loader_rng.get_state(),
            augmentation_rng=train_ds.gamma.R.get_state() if train_ds.gamma is not None else None))
        pd.DataFrame(history).to_csv(out/'history.csv', index=False)
    if best_predictions is None:
        raise ValueError('No completed epoch/checkpoint')
    result = va[[UID, 'group', 'fold']+YC+WC].copy()
    for j in range(12):
        result[f'p_{j}'] = best_predictions[:, j]
    result.to_csv(out/'validation_predictions.csv', index=False)
    measured = metrics(y, w, best_predictions, active, prior)
    pd.DataFrame(measured['per_target']).to_csv(out/'target_metrics.csv', index=False)
    summary = dict(run_id=run_id, experiment=cfg['experiment'], validation_id=validation_id,
        cohort_id=meta['cohort_id'], supervision_id=meta['supervision_id'], cache_id=meta['cache_id'],
        best_epoch=best_epoch, metrics=measured, active_targets=active.tolist(),
        wall_seconds=elapsed_before+time.perf_counter()-wall_start, role='report_label_exploration',
        source_id=source_id, pretrained_sha256=model.weight_hash, environment=environment(),
        validation_predictions_sha256=sha256(out/'validation_predictions.csv'),
        best_checkpoint_sha256=sha256(out/'best.pt'), train_prior=prior.tolist())
    write_json(out/'run_summary.json', summary)
    return out
