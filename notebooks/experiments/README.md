# Kaggle experiments: start here

This suite builds on the **fresh preparation output**, not the old v1 pseudo-label dataset. Keep your successful three fresh notebooks and their baseline outputs as history. These new three notebooks contain the full runtime package: no repository upload is needed.

## Inputs and order

| Notebook | Attach | Settings | Output to save |
|---|---|---|---|
| `01_Experiment_Readiness.ipynb` | Saved **fresh preparation notebook 01** output containing `fresh_manifest.csv`, `fresh_preparation.json`, preferably `label_audit.csv` | CPU; fold 0; max_train 512, max_val 240 | `rsna_experiment_preparation` |
| `02_Preprocessing_Experiments.ipynb` | Original **RSNA competition** + saved **experiment readiness 01** output | CPU; `RECIPE='p0'`; run `PILOT=True`, inspect, then `PILOT=False` | `rsna_experiment_p0_cache/p0` |
| `03_Training_Experiments.ipynb` | Saved **full experiment preprocessing 02** output | GPU; default `EXPERIMENT='r00_resnet_control'`; Internet for setup/weights | `rsna_r00_resnet_control` |
| `04_Full_Cohort_Readiness.ipynb` | Saved **fresh preparation 01** output with `fresh_manifest.csv` | CPU; source fold 1; all eligible studies | `rsna_full_fold1_preparation` |
| `05_Full_Cohort_Preprocessing.ipynb` | RSNA competition + saved **04** output | CPU; `p0`, 16 slices, 192 pixels, complete cohort | `rsna_full_fold1_p0_cache/p0` |
| `06_Long_Attention_Training_and_Evaluation.ipynb` | Saved complete **05** cache | GPU; `r01_target_attention`, 200 maximum epochs, BCE early stopping | `rsna_full_fold1_r01_target_attention_long_v1` |

Paths search `/kaggle/input/datasets/aymenslimani/` and then `/kaggle/input/`, including Kaggle notebook-output mounts. Competition data stays under `/kaggle/input/competitions/rsna-knee-abnormality-detection`. If two differing cache/preparation copies are attached, paste the desired filename's copied Kaggle path in `MANIFEST_PATH` or `CACHE_META_PATH`; ambiguity is an error, not arbitrary selection. The earlier `rsna_fresh_cache` cannot be used by the new training notebook because it contains only one series and a different schema/cohort.

`INSTALL_MISSING=True` installs absent dependencies. Enable Kaggle Internet for this setup, or set `WHEEL_DIR` to an attached wheel directory. Missing compressed-DICOM decoder plugins surface during the pilot; install the plugin identified by pydicom for that transfer syntax before a full cache. Package versions are recorded. Local tests use MONAI 1.5.1; an already installed MONAI version is preserved and reported.

## First session

1. Import experiment 01, attach your saved **fresh preparation** output and run. Inspect `target_status_audit_sample.csv`, coverage, warnings and `preparation_meta.json`. Source folds/labels are preserved, and unknown labels remain masked. No training report text is sent to external services.
2. Import experiment 02, attach the competition and experiment 01 output. Run the 12-study timing pilot. Inspect `montage.png` and `cache_audit.json`, especially missing planes, rejected geometry, series selection, crop failures and decode errors. Full-byte source checksums read the original files even though only selected slices are decoded; time the actual pilot rather than assuming preprocessing is cheap.
3. Set `PILOT=False` and run the full **same prepared cohort**. Training refuses truncated pilot caches or caches with unusable studies. In notebook 03, attach the saved **full** preprocessing output; attaching the pilot output produces an intentional “preprocessing PILOT/truncated cache” error. Save & Run All and attach the saved full output to experiment 03.
4. Run `r00_resnet_control` on GPU first. Then run the matched short screening experiments before `h04_200epoch_early_stop`. The short runs use the same eight-epoch ceiling and early stopping so their differences are easier to attribute. The long run is reserved for the strongest candidate. Generic ImageNet weights download when `ALLOW_WEIGHT_DOWNLOADS=True`. For offline use, attach a standard torchvision ResNet18 state dict and set `WEIGHTS_PATH`; the old MRI-trained `best.pt` is incompatible.
5. Run `r04_frozen_encoder` on the same full cache. Then use the staged shortlist in the [research review](../../docs/RSNA_EXPERIMENT_RESEARCH.md). Do not run every model × preprocessing × loss combination.

The registry control uses batch size two, accumulation four, eight training/16 validation slice positions, eight epochs maximum, cosine LR and mixed precision on GPU. Notebook 06 explicitly selects a 200-epoch ceiling; early stopping usually ends it sooner, but this is not guaranteed. The cosine schedule spans the selected epoch ceiling, so this run also changes the learning-rate trajectory. Timings and disk estimates are outputs, not promises. If GPU memory is insufficient, lower batch size and adjust accumulation for the desired effective batch; rerun the matched control with that setting. No automatic batch-size reductions silently change an experiment.

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

## Full-cohort confirmation round

Use notebooks 04, 05, then 06. Notebook 04 needs the original fresh-manifest output, not the bounded screening preparation. Its `max_train=0` and `max_val=0` settings retain all eligible source studies, with fold 1 as the confirmation split. Notebook 05 needs the RSNA competition files and the saved output of 04. Notebook 06 needs only the complete new cache from 05 and a GPU. Run `r01_target_attention` first; save its full output, then change only `EXPERIMENT` to `r00_resnet_control` and run again as a matched control. For a comparison-only Kaggle session, attach both saved long-run outputs, set `RUN_TRAINING=False`, fill `RUN_DIRS` with the exact directories (control first), and run notebook 06. The cache and GPU are unnecessary in that comparison session.

The existing saved screening cache has 752 studies (512 train, 240 validation). The fresh source manifest has 4,407 studies. Source fold 1 has examples of both classes for all 12 targets, but only two negatives each for MCL and synovitis. Report their individual scores with this support; a 12-target average is mathematically defined but uncertain. Report-derived labels and report-hash grouping do not establish clinical accuracy or patient-independent validation. Review the label audit before making such claims. Local screening ZIPs can be compared with `python scripts/compare_saved_experiment_zips.py`; its output is exploratory and must not be ranked directly against the new cohort.

## Validation status

The suite is designed for synthetic local checks plus real-data Kaggle pilot inspection. The final validation receipt is [RSNA_EXPERIMENT_VALIDATION.md](../../docs/RSNA_EXPERIMENT_VALIDATION.md). Creating notebooks does not establish a new competition score. Research rationale, primary references, historical score caveats and deferred extensions are in [RSNA_EXPERIMENT_RESEARCH.md](../../docs/RSNA_EXPERIMENT_RESEARCH.md).
