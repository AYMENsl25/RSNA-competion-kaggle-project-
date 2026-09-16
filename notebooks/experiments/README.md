# Kaggle experiments: start here

This suite builds on the **fresh preparation output**, not the old v1 pseudo-label dataset. Keep your successful three fresh notebooks and their baseline outputs as history. These new three notebooks contain the full runtime package: no repository upload is needed.

## Inputs and order

| Notebook | Attach | Settings | Output to save |
|---|---|---|---|
| `01_Experiment_Readiness.ipynb` | Saved **fresh preparation notebook 01** output containing `fresh_manifest.csv`, `fresh_preparation.json`, preferably `label_audit.csv` | CPU; fold 0; max_train 512, max_val 240 | `rsna_experiment_preparation` |
| `02_Preprocessing_Experiments.ipynb` | Original **RSNA competition** + saved **experiment readiness 01** output | CPU; `RECIPE='p0'`; run `PILOT=True`, inspect, then `PILOT=False` | `rsna_experiment_p0_cache/p0` |
| `03_Training_Experiments.ipynb` | Saved **full experiment preprocessing 02** output | GPU; default `EXPERIMENT='r00_resnet_control'`; Internet for setup/weights | `rsna_r00_resnet_control` |

Paths search `/kaggle/input/datasets/aymenslimani/` and then `/kaggle/input/`, including Kaggle notebook-output mounts. Competition data stays under `/kaggle/input/competitions/rsna-knee-abnormality-detection`. If two differing cache/preparation copies are attached, paste the desired filename's copied Kaggle path in `MANIFEST_PATH` or `CACHE_META_PATH`; ambiguity is an error, not arbitrary selection. The earlier `rsna_fresh_cache` cannot be used by the new training notebook because it contains only one series and a different schema/cohort.

`INSTALL_MISSING=True` installs absent dependencies. Enable Kaggle Internet for this setup, or set `WHEEL_DIR` to an attached wheel directory. Missing compressed-DICOM decoder plugins surface during the pilot; install the plugin identified by pydicom for that transfer syntax before a full cache. Package versions are recorded. Local tests use MONAI 1.5.1; an already installed MONAI version is preserved and reported.

## First session

1. Import experiment 01, attach your saved **fresh preparation** output and run. Inspect `target_status_audit_sample.csv`, coverage, warnings and `preparation_meta.json`. Source folds/labels are preserved, and unknown labels remain masked. No training report text is sent to external services.
2. Import experiment 02, attach the competition and experiment 01 output. Run the 12-study timing pilot. Inspect `montage.png` and `cache_audit.json`, especially missing planes, rejected geometry, series selection, crop failures and decode errors. Full-byte source checksums read the original files even though only selected slices are decoded; time the actual pilot rather than assuming preprocessing is cheap.
3. Set `PILOT=False` and run the full **same prepared cohort**. Training refuses truncated pilot caches or caches with unusable studies. In notebook 03, attach the saved **full** preprocessing output; attaching the pilot output produces an intentional “preprocessing PILOT/truncated cache” error. Save & Run All and attach the saved full output to experiment 03.
4. Run `r00_resnet_control` on GPU first. Then run the matched short screening experiments before `h04_200epoch_early_stop`. The short runs use the same eight-epoch ceiling and early stopping so their differences are easier to attribute. The long run is reserved for the strongest candidate. Generic ImageNet weights download when `ALLOW_WEIGHT_DOWNLOADS=True`. For offline use, attach a standard torchvision ResNet18 state dict and set `WEIGHTS_PATH`; the old MRI-trained `best.pt` is incompatible.
5. Run `r04_frozen_encoder` on the same full cache. Then use the staged shortlist in the [research review](../../docs/RSNA_EXPERIMENT_RESEARCH.md). Do not run every model × preprocessing × loss combination.

The registry control uses batch size two, accumulation four, eight training/16 validation slice positions, eight epochs maximum, cosine LR and mixed precision on GPU. The notebook selects the new 200-epoch ceiling; early stopping usually ends it sooner, but this is not guaranteed. The cosine schedule spans the selected epoch ceiling, so this run also changes the learning-rate trajectory. Timings and disk estimates are outputs, not promises. If GPU memory is insufficient, lower batch size and adjust accumulation for the desired effective batch; rerun the matched control with that setting. No automatic batch-size reductions silently change an experiment.

## Which experiment to select

All 19 named settings are shown in notebook 03 before training. Changes are made in `EXPERIMENT`; use `CONFIG.update(...)` before the training cell for a named hyperparameter override, and choose a new `out_dir` for every changed configuration.

| IDs | Purpose |
|---|---|
| `r00_resnet_control` | Pretrained ResNet18 with masked mean/max pooling |
| `r01_target_attention`, `r02_single_sagittal`, `r03_dense_slices` | Pooling, number of planes, training slice coverage |
| `r04_frozen_encoder`, `r05_staged_unfreeze`, `r06_monai_intensity_aug` | Freeze strategy and MONAI contrast augmentation |
| `l01_pos_weight_bce`, `l02_monai_focal`, `l03_asymmetric`, `l04_bce_rank`, `l05_label_smoothing` | Five alternative losses against the same BCE control |
| `h01_lower_lr`, `h02_higher_decay`, `h03_longer_budget`, `h04_200epoch_early_stop` | LR, regularization, 16-epoch and 200-epoch training ceilings |
| `m01_dinov2_frozen`, `m02_monai_3d` | Generic frozen SSL features and small 3D DenseNet |
| `s00_tiny_smoke` | Two-epoch execution check |

Select `p0`–`p5` in notebook 02 for preprocessing variations. Each is saved independently; rerun the same training control on each. Do not compare runs that silently dropped different failed studies. Three-plane selection is geometry-checked and prefers explicit fluid/FS evidence; it is not a complete six-semantic-slot model. Sagittal-only training refuses a cohort with any missing sagittal series so it cannot gain an advantage by filtering difficult cases.

For DINO: set `PREPARE_DINO_WEIGHTS=True` in the optional setup cell with Internet enabled. It saves `generic_weights/dinov2_vits14_timm.pth` and a checksum. Select `m01_dinov2_frozen`; the setup cell assigns the downloaded path in the current run. Offline, attach that saved weight file and set `WEIGHTS_PATH`. The encoder stays frozen but is evaluated each batch: precomputed feature caching is not implemented. The MONAI 3D branch needs 16 depth positions and is randomly initialized; compare against `r03_dense_slices` and treat this as an architecture-family comparison, not a one-parameter change.

## Compare and resume

Notebook 03 saves `run_summary.json`, `history.csv`, `target_metrics.csv`, `validation_predictions.csv`, `best.pt`, `last.pt` and the full configuration. `delta_vs_constant < 0` means lower validation macro BCE than a prediction based only on training label frequencies. Per-target loss normalization differs from the original fresh notebook's pooled BCE, so do not compare the new number directly with 0.581.

To compare without retraining: import notebook 03 into a separate Kaggle session, attach saved training outputs, set `RUN_TRAINING=False`, then put their exact folders in `RUN_DIRS`, control first. It checks validation IDs, groups, labels and active target coverage before producing comparisons, paired group-bootstrap intervals and an exploratory equal-probability ensemble. Undefined target AUCs remain undefined. Confirm selected improvements on another fold; repeated validation selection is not an independent test.

Resume an interrupted training run by attaching its full output and pointing `RESUME_PATH` to `last.pt`. Config, source runtime, generic weights and cache must match. The checkpoint carries optimizer, scheduler, scaler, RNG and selected best model; recovery is at an epoch boundary. It does not recover a partially completed epoch. Changing epochs or a hyperparameter is a new experiment, not an identical resume.

Preprocessing reuses verified per-study NPZs in its output directory. A killed kernel can leave `.build.lock`; confirm no writer is still running before removing only that cache's stale lock and retrying. Do not delete source inputs or entire working directories. On Kaggle a new session may require importing the previous output into a fresh writable cache location to resume; the current implementation is optimized for same-session/writable-directory cache reuse, not automatic read-only output migration.

## Reviewed training-label corrections

Use a separate CSV with columns `StudyInstanceUID,target,label,weight,reviewer,reason`, with exact target names from the notebook and numeric label/weight in [0,1]. Only selected **training** studies may be corrected; duplicate entries, unknown IDs and validation edits are rejected. Attach the CSV, set `CORRECTIONS_PATH` in notebook 01 and a new output directory, then rebuild the prepared output. This is an explicit supervised data change. The suite does not automatically generate a new multilingual teacher or turn unmentioned findings into negatives.

## Validation status

The suite is designed for synthetic local checks plus real-data Kaggle pilot inspection. The final validation receipt is [RSNA_EXPERIMENT_VALIDATION.md](../../docs/RSNA_EXPERIMENT_VALIDATION.md). Creating notebooks does not establish a new competition score. Research rationale, primary references, historical score caveats and deferred extensions are in [RSNA_EXPERIMENT_RESEARCH.md](../../docs/RSNA_EXPERIMENT_RESEARCH.md).
