# RSNA Knee v2 — Kaggle run guide

Nine standalone notebooks embed the same reviewed source modules. Upload any `.ipynb` directly to Kaggle; no GitHub checkout is needed in the notebook. Generated notebooks are rebuilt with `python scripts/build_v2_notebooks.py` after source changes.

## Run order and required attachments

| Notebook | Accelerator | Attach | Saves |
|---|---|---|---|
| 00 Inputs/Splits/Weights | CPU | Competition, pseudo labels, slot plan, v1.2 fold CSV | `rsna_v2_inputs/supervision`, optional pretrained weights |
| 01 Preprocessing Pilot | CPU | Competition and old slot plan | 8-study pilot, montage, decode/geometry audits |
| 02 Preprocessing Full | CPU | Competition and old slot plan | Full uint8 cache, index, hashes, audits |
| 03 Train ResNet18 | GPU | Full 02 cache, 00 supervision and ImageNet weights | `best.pt`, `last.pt`, history, validation predictions |
| 04 Train Frozen DINOv2 | GPU | Full 02 cache, 00 supervision and DINO weights | Same outputs, frozen encoder |
| 05 Compare Experiments | CPU | Matching-fold 03/04 outputs and 00 supervision | Macro/per-target pseudo BCE and prediction correlations |
| 06 Refit Frozen Recipe | GPU | Selected checkpoint, original pretrained weights, full cache, supervision | Final pseudo-only refit checkpoint |
| 07 Historical Gold Audit | GPU | Frozen recipe/checkpoints, full cache, supervision, competition | Descriptive 58-case AUC/AP and bootstrap interval |
| 08 Submission Inference | GPU | Final checkpoints and exact preprocessing metadata, competition | `submission.csv` |

Save Version with **Save & Run All** for each completed stage. Attach the saved notebook output as an input to the next stage. Inputs are read-only under `/kaggle/input`; all new outputs go under `/kaggle/working`.

## Original paths retained

Competition root: `/kaggle/input/competitions/rsna-knee-abnormality-detection`. The older `/kaggle/input/rsna-knee-abnormality-detection` mount is also recognized. The real series directories in the prior version are `train_series` and `test_series`; `*_images` is recognized only when that is the sole available directory.

Previous known attachment roots:

- `/kaggle/input/datasets/aymenslimani/phase-4a-3/` — `rsna_pseudo_labels_v1_1_wide.csv`, `PSEUDO_LABELS_v1_1.json`.
- `/kaggle/input/datasets/aymenslimani/pseudo-labels-v1-1-dataset/` — `rsna_phase4b2a_six_slot_plan_v1.csv`.
- `/kaggle/input/datasets/aymenslimani/4b-1a/` — historical indexing outputs.
- `/kaggle/input/datasets/aymenslimani/result-phase-4-b/` — `rsna_phase4b1b_pseudo_fold_assignments_v1_2.csv`. Confirm the copied path in Kaggle; discovery also handles a different dataset slug.

For notebook 00 attach exactly the competition, **Phase 4A-3**, **PSEUDO_LABELS_v1_1 dataset**, and **result phase 4 b**. The 4B-1A indexing dataset and advanced series inventory are not required. Notebook 00 checks all four required artifact files before processing, prints their actual paths, and lists available fold CSVs when the expected fold file is missing. A path change cannot supply an unattached file: upload the original v1.2 fold CSV if the result dataset does not contain it. Set `INPUT_OVERRIDES['folds_path']` to its copied path if its filename differs, only after confirming it is the correct v1.2 assignment file.

The dataset search starts at `/kaggle/input/datasets/aymenslimani/` and falls back to `/kaggle/input` for saved notebook outputs and legacy mounts. Keep the competition under its separate `/kaggle/input/competitions/` mount. The slot plan is read using only three string columns and cleaned into a separate working CSV; the original series-level CSV remains the preprocessing input.

Resolvers search the exact previous filenames. If several attachments contain different versions of the same name, set the explicit path in the configuration cell. Identical copies are accepted. New cache/run attachment slugs are not known until you save them; later notebooks discover metadata by filename or accept explicit overrides.

The old `/kaggle/input/models/aymenslimani/phase4b2b1-selection-best-v1-/other/default/1/phase4b2b1_selection_best_v1` is a legacy selection archive. It is not a v2 checkpoint and cannot initialize these new models.

## First runs

1. Run notebook 00 and inspect its PASS counts (4341 training, 58 gold, 6 exclusions). Enable `PREPARE_WEIGHTS=True` only in an Internet-enabled setup session to save generic ResNet18 and DINOv2 weights.
2. Run notebook 01 with the default control. Inspect every pilot montage and geometry/decode audit. Cold decoding and preprocessing time count toward runtime; extrapolate pilot cost with headroom before a full run.
3. Run notebook 02 for `p0_control`, save its output.
4. Train `r0_resnet_meanmax` on fold 0, then `d0_dino_frozen_meanmax` on the exact same cache/fold.
5. Attach both run outputs to notebook 05 and set explicit `RUN_DIRS`.
6. Compare promising preprocessing variants with the same training configuration, split and seed. Use new names/output directories for every changed recipe.

The default cache budget is 24 slices × 224² uint8 pixels per selected series, approximately 25.7 GB uncompressed for 21,334 series plus metadata/reserve. NPZ compression often reduces disk use, but the preflight budgets uncompressed size. Free space is checked before the full run. Pilot caches deliberately fail production-training checks. Up to 10% malformed instances per series may be skipped; failures are always recorded and valid slices remain masked. A series with no valid slices or rejected geometry fails the production cache.

## Controlled experiment matrix

| ID | Change | Keep fixed | Why |
|---|---|---|---|
| p0/r0 | 130-mm crop, percentile normalization, broad slice band, repeat channels, ResNet18 mean/max | Frozen teacher/folds | Establish the new v2 control |
| p1 | Full field of view | Encoder, head, anchors, teacher/folds | Test crop loss of peripheral pathology |
| p2 | 150-mm crop | Same | Test anatomical context versus resolution |
| p3 | CLAHE | Crop, model, folds | Test local contrast |
| p4 | Z-score mapping | Same | Test normalization sensitivity |
| p5 | Narrower .12–.88 slice band | Same | Test sparse band coverage |
| r1 | 16 train anchors instead of 8 | p0 cache, model, head, teacher/folds | More anatomy exposure |
| r2 | Neighboring retained slices as three channels | Same | Test 2.5D input; channels may not be physically adjacent source slices |
| r3 | Compact target attention | Same | Test pooling capacity under fixed representations |
| d0 | Frozen generic DINOv2-S/14 | Same cache and split, mean/max head | Strong representation with limited trainable capacity |
| d1 | DINO target attention | Same | Test head complexity independently |

The defaults use legacy plane/FS slots so training and inference have a common tested contract. Semantic fluid/T1 slots, measured laterality mirroring, patch-level focal pooling, precomputed frozen features, supervision v2, ranking loss, full OOF stack, and final-block unfreezing are further development stages; they are **not implemented experiments** in this release. The preprocessing module has an optional geometry-axis standardization and a metadata/description consistency audit; these are not anatomical left/right mirroring or a fluid/T1 slot system. Do not enable them as if validated semantic preprocessing were complete.

For supervision-weight experiments, preserve the unchanged validation weights. Reducing validation supervision can produce a misleading lower loss. Current training deliberately uses the original teacher only. Five folds can be run individually by changing `FOLD` to 0–4; notebook 05 compares identical cohorts, not unmatched folds. Full OOF assembly/stack selection is a later feature.

## Resume and refit

Set `resume_path` to the prior `last.pt` and attach the complete prior run output, including `best.pt` and `validation_predictions.csv`. The epoch budget and model configuration must match the saved run identity. Use a new run for an extended epoch budget. Each completed epoch saves its state, optimizer/scaler and RNG; no checkpoint can recover an interrupted mid-epoch step.

Notebook 06 is disabled until you specify the selected checkpoint, original generic pretrained backbone file, and fixed epoch budget. It starts a fresh model on all pseudo-training studies. Notebook 07 is disabled until the recipe and checkpoint hashes are frozen. Notebook 08 is disabled until final checkpoint and preprocessing hashes are specified.

## Evaluation interpretation

The 58 gold cases have informed historical teacher/model comparisons. Group separation prevents training leakage but does not restore an untouched test set after adaptive inspection. Gold results are descriptive historical diagnostics. Pseudo BCE measures teacher agreement, not clinical correctness or a guaranteed gold-AUC gain. The report teacher AUC is not a mathematical ceiling for an MRI student.

The archived v1.2 handoff lists `0.6993578670`, while the stored JSON and CSV mean give approximately `0.6994008394`. Preserve both sources and recompute from saved predictions for exact comparisons. The new v2 control changes preprocessing/training and should not be described as an exact reproduction.

The archived public approximately 0.94 ensemble is a historical reference whose original notebooks are absent here. No score close to or above it is guaranteed by this preparation.

## Dependencies and sources

Kaggle commonly provides NumPy, pandas, PyTorch, torchvision and scikit-learn. `timm`, pydicom and image/decoder packages may require setup. The dependency cell defaults to no installation; set `INSTALL_MISSING=True` in setup or attach compatible wheels via `WHEEL_DIR`. Compressed DICOM may need `pylibjpeg`, `pylibjpeg-libjpeg`, `pylibjpeg-openjpeg` or GDCM. See [official pydicom decoder guidance](https://pydicom.github.io/pydicom/stable/guides/user/image_data_handlers.html).

The DINO model is `vit_small_patch14_dinov2.lvd142m`; setup stores the full timm state dictionary and inference rebuilds the same architecture without downloading. See [official timm model configuration](https://huggingface.co/timm/vit_small_patch14_dinov2.lvd142m/blob/main/config.json). Matching architecture is mandatory for externally attached weights; arbitrary Hugging Face or public competition weights are not interchangeable.

Synthetic tests validate software behavior only. You must verify real decoder support, anatomy retention, GPU memory, full cache completion and hidden-test runtime on Kaggle.
