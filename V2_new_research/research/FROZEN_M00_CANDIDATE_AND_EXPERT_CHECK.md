# Frozen M00 candidate and predeclared one-time expert check — 2026-10-11

**Status update, October 11:** The user declined M00 as a competition-submission candidate and requested stronger methods. Preserve this record as the frozen M00 development artifact, but do **not** run its one-time 58-study expert check now. Keep those labels sealed while the new experiment plan is decided. A later candidate must have its own frozen inference/evaluation record before the one-time check.

This record freezes the development winner **before reading the 58 expert target values**. M00 epoch 3 was selected solely from the frozen public-report-proxy validation. M01, a blend, thresholds and new architectures will not be selected using the expert check while calling it a final check.

- Candidate checkpoint: `outputs/m00_full_train_v1/m00_best.pt`, SHA-256 `783743cf59b6acea67b1fd3cf1b2ed11a9b818c615a69af9b6d61f73f785c721`.
- Model: random-init ResNet18 2.5D, shared max pooling, twelve sigmoid outputs. Source hashes in the checkpoint: `m00_data.py` `84190f34e7f4f25518f2f0855104c1c8514bb6e0f4789c1e6e9d12d4803811c5`; `m00_model.py` `e8410ff5ebcfcf2036c5da18538fc194fe6db24eb88f274981bfba9de3ed531e`; trainer template `9a376d9709249d8497af44db2bf6d665e39166d25819e96bc252fa6ef0577952`.
- Preprocessing and inference policy: use exactly the same series selection, geometry guards, pixel scaling, 1st/99th percentile normalization and `build_study(train=False)` from the hashed `m00_data.py`. This yields up to 16 three-slice windows for each sagittal, coronal and axial role, resized to 224×224. No augmentation, test-time flip, blending, calibration or report input. Use `model.eval()` and sigmoid on the 12 logits.
- Weak development evidence: best epoch 3 macro AUC 0.739719 on 863 public-report-proxy validation studies. This was compared with M01 before the expert check. Both models used one training seed. This score is not expert image-label accuracy.

## One-time expert-check protocol

1. Finish and hash the inference and metric scripts before opening the 58 expert target values. Verify image decoding, output order, finite probabilities and metric behavior on synthetic/nonexpert data first.
2. Run this **M00 checkpoint once** on the 58 excluded expert-labeled MRI studies. Do not train on them or use their results to adjust model, preprocessing, checkpoint, blend, thresholds or labeler.
3. For each of the twelve targets, report positive and negative support and tie-aware ROC-AUC when both classes occur. Mark AUC undefined if a class is absent. Report the twelve-target macro only if all twelve AUCs are defined; otherwise report the number of defined targets and their mean with an explicit non-comparability warning. Report a study/group bootstrap interval only where resamples contain both classes, plus the valid-draw count and seed.
4. Report study failures and full inference runtime. Any failed study makes the check incomplete. Record exact checkpoint, code and data hashes.

The public report-label source was reportedly selected using the 58 expert cases by its publisher. Consequently, this is a held-out check for **our image-model fitting**, but it is not independent of every supervision-source choice. The final Kaggle competition submission requires separate explicit user approval.
