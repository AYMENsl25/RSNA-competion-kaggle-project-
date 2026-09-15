"""One-factor experiments. Each run is selected explicitly; no automatic sweep."""
BASE = dict(model='resnet18', pooling='meanmax', loss='bce', pretrained=True,
            strategy='finetune', epochs=8, batch_size=2, accumulate=4,
            lr=3e-4, encoder_lr=3e-5, weight_decay=1e-4, dropout=0.2,
            seed=42, workers=0, train_slices=8, eval_slices=16,
            augmentation='none', amp=True, patience=3, min_epochs=4,
            selection='macro_bce', gamma=2.0, gamma_pos=0.0, gamma_neg=4.0,
            asl_clip=0.05, ranking_weight=0.1, label_smoothing=0.05,
            planes='all', allow_downloads=False, weights_path=None)

EXPERIMENTS = {
    'r00_resnet_control': {},
    'r01_target_attention': {'pooling': 'attention'},
    'r02_single_sagittal': {'planes': 'sagittal'},
    'r03_dense_slices': {'train_slices': 16},
    'r04_frozen_encoder': {'strategy': 'frozen'},
    'r05_staged_unfreeze': {'strategy': 'staged'},
    'r06_monai_intensity_aug': {'augmentation': 'intensity'},
    'l01_pos_weight_bce': {'loss': 'pos_weight'},
    'l02_monai_focal': {'loss': 'focal'},
    'l03_asymmetric': {'loss': 'asl'},
    'l04_bce_rank': {'loss': 'bce_rank'},
    'l05_label_smoothing': {'loss': 'smooth'},
    'h01_lower_lr': {'lr': 1e-4, 'encoder_lr': 1e-5},
    'h02_higher_decay': {'weight_decay': 1e-3},
    'h03_longer_budget': {'epochs': 16, 'min_epochs': 8, 'patience': 5},
    'm01_dinov2_frozen': {'model': 'dinov2', 'strategy': 'frozen'},
    'm02_monai_3d': {'model': 'monai3d', 'pretrained': False, 'train_slices': 16},
    's00_tiny_smoke': {'model': 'tiny', 'pretrained': False, 'epochs': 2,
                      'min_epochs': 1, 'batch_size': 2, 'accumulate': 1},
}


def experiment(name, **overrides):
    if name not in EXPERIMENTS:
        raise ValueError(f'Unknown experiment {name}; choose from {list(EXPERIMENTS)}')
    unknown = set(overrides) - set(BASE) - {'cache_dir', 'out_dir', 'resume_path'}
    if unknown:
        raise ValueError(f'Unknown settings: {sorted(unknown)}')
    return {**BASE, **EXPERIMENTS[name], **overrides, 'experiment': name}
