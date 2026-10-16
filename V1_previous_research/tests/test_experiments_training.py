import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
from rsna_experiments.common import UID, YC, WC, sha256, write_json
from rsna_experiments.registry import experiment
from rsna_experiments.models import StudyModel
from rsna_experiments.losses import MaskedObjective
from rsna_experiments.training import metrics, run_training


def test_masked_losses_ignore_unknown_and_have_finite_gradients():
    y = torch.tensor([[1., 0.]*6, [0., 1.]*6, [1., 1.]*6])
    w = torch.ones_like(y); w[2] = 0
    for loss in ['bce', 'pos_weight', 'focal', 'asl', 'bce_rank', 'smooth']:
        cfg = experiment('s00_tiny_smoke', loss=loss)
        objective = MaskedObjective(cfg, y, w)
        z = torch.zeros_like(y, requires_grad=True)
        value = objective(z, y, w)
        value.backward()
        assert torch.isfinite(value) and torch.isfinite(z.grad).all()
        assert (z.grad[2] == 0).all()
        if loss == 'bce':
            assert value.item() == pytest.approx(np.log(2))


@pytest.mark.parametrize('pooling', ['meanmax', 'attention'])
def test_padding_mask_cannot_change_logits(pooling):
    model = StudyModel(experiment('s00_tiny_smoke', pooling=pooling)).eval()
    images = torch.rand(2, 3, 2, 24, 24)
    mask = torch.ones(2, 3, 2, dtype=torch.bool); mask[:, 2] = False
    result = model(images, mask)
    images[:, 2] = 10000
    assert torch.allclose(model(images, mask), result)
    assert result.shape == (2, 12)
    with pytest.raises(ValueError, match='no valid'):
        model(images, torch.zeros_like(mask))


def test_monai_3d_backward_and_intensity_transform():
    model = StudyModel(experiment('m02_monai_3d')).train()
    model.set_epoch(0)
    image = torch.rand(1, 3, 16, 32, 32)
    mask = torch.ones(1, 3, 16, dtype=torch.bool)
    logits = model(image, mask)
    logits.sum().backward()
    assert logits.shape == (1, 12)
    assert torch.isfinite(logits).all()


def test_macro_metrics_do_not_fill_undefined_auc():
    y = np.zeros((4, 12)); y[:, 0] = [0, 0, 1, 1]
    w = np.zeros_like(y); w[:, :2] = 1
    p = np.full_like(y, .5); p[:, 0] = [.1, .2, .8, .9]
    result = metrics(y, w, p, np.ones(12, dtype=bool), np.full(12, .5))
    assert result['macro_auc'] == 1
    assert result['n_auc_targets'] == 1
    assert result['per_target'][1]['auc'] is None
    assert result['delta_vs_constant'] < 0


def make_cache(root):
    root.mkdir()
    rows, index = [], []
    for i in range(8):
        uid = str(i)
        row = {UID: uid, 'group': f'g{i}', 'fold': i % 2}
        row.update({c: float((i//2) % 2) for c in YC})
        row.update({c: 1. for c in WC}); rows.append(row)
        path = root/f'{uid}.npz'
        np.savez_compressed(path, images=np.random.default_rng(i).integers(0, 255, (3, 2, 24, 24), dtype=np.uint8), mask=np.ones((3, 2), bool))
        index.append({UID: uid, 'file': path.name, 'sha256': sha256(path)})
    pd.DataFrame(rows).to_csv(root/'cache_manifest.csv', index=False)
    pd.DataFrame(index).to_csv(root/'cache_index.csv', index=False)
    write_json(root/'cache_meta.json', dict(complete=True, fold=0, cache_id='synthetic-cache', cohort_id='cohort', supervision_id='supervision',
               manifest_sha256=sha256(root/'cache_manifest.csv'), index_sha256=sha256(root/'cache_index.csv')))
    return root


def test_training_checkpoint_and_identical_rerun(tmp_path):
    torch.set_num_threads(1)
    cache = make_cache(tmp_path/'cache')
    cfg = experiment('s00_tiny_smoke', cache_dir=str(cache), out_dir=str(tmp_path/'run'),
                     train_slices=2, eval_slices=2, amp=False, augmentation='intensity')
    out = run_training(cfg)
    summary = json.loads((out/'run_summary.json').read_text())
    assert summary['metrics']['n_auc_targets'] == 12
    assert summary['pretrained_sha256'] is None
    assert (out/'last.pt').exists()
    old_hash = sha256(out/'best.pt')
    assert run_training(cfg) == out
    assert sha256(out/'best.pt') == old_hash
    # Resume from saved complete epochs into another directory, preserving the selected model.
    resumed = run_training({**cfg, 'out_dir': str(tmp_path/'resumed'), 'resume_path': str(out/'last.pt')})
    a = pd.read_csv(out/'validation_predictions.csv'); b = pd.read_csv(resumed/'validation_predictions.csv')
    pd.testing.assert_frame_equal(a, b)
    broken = dict(cfg, lr=.02)
    with pytest.raises(ValueError, match='different experiment'):
        run_training(broken)
