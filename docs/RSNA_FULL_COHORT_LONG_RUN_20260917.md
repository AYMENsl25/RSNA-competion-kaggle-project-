# Full-cohort long training and evaluation plan — 2026-09-17

## Screening result and decision

The eight saved ZIPs in `results/` were compared using the same 240 validation studies,
235 report-hash groups, and the same cohort/cache/supervision IDs. The strict local
comparison receipt is `results/screening_comparison/comparison.json`. Best checkpoints
were chosen by validation macro BCE, and ROC AUC averages only the nine targets with
both observed classes. These scores are report-label exploration, not the official
12-target competition score.

| Configuration | Best macro BCE ↓ | Nine-target macro AUC ↑ | Decision |
|---|---:|---:|---|
| `r01_target_attention` | 0.354079 | 0.794998 | Primary long-run candidate |
| `r03_dense_slices` | 0.380974 | 0.748595 | Secondary candidate |
| `r00_resnet_control` | 0.408866 | 0.695371 | Matched control |
| `r05_staged_unfreeze` | 0.411725 | 0.702440 | No priority |
| `r06_monai_intensity_aug` | 0.412138 | 0.700557 | No priority |
| `r04_frozen_encoder` | 0.453696 | 0.566072 | No priority |
| `l03_asymmetric` | 0.468694 | 0.679122 | No priority |
| `l02_monai_focal` | 0.474588 | 0.767884 | Ranking reasonable, BCE worse than prior |

The paired 1,000-draw report-hash-group bootstrap yielded 908 valid draws.
`r01` minus `r00` AUC was +0.099626 (95% percentile interval +0.023757 to
+0.172747). `r03` minus `r00` was +0.053224 (interval -0.014439 to +0.119336).
These intervals describe the selected screening fold and do not remove the bias from
choosing a candidate after viewing that fold repeatedly.

## Why the full cohort must be a new experiment

The saved `02 RSNA preprocessing FULL.zip` is complete for **752 studies** (512 train,
240 validation); `max_studies=0` means it processed all rows in that bounded manifest,
not all 4,407 eligible source studies. The source fold-0 validation has **zero MCL
negatives** even when expanded, so it cannot produce the official 12-target average.
Source fold 1 has both classes for all 12 targets, but only **two negatives each** for
MCL and synovitis. Its 12-target AUC is defined but uncertain. The official Kaggle
metric averages ROC AUC across 12 findings; do not compare it directly with the
screening nine-target average. See the [official Kaggle evaluation page](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview/evaluation).

Source patient identifiers were unavailable to the fresh grouping step. It groups
identical normalized reports, which does not prove patient independence. Labels are
heuristically extracted from multilingual reports. Review `target_status_audit_sample.csv`
and the original report examples before treating fold-1 performance as trustworthy.
No claim of clinical validation or leaderboard improvement follows from this workflow.

## Kaggle execution sequence

1. Import `notebooks/experiments/04_Full_Cohort_Readiness.ipynb`. Attach the saved
   *fresh preparation* output containing `fresh_manifest.csv`. Run on CPU. Confirm
   **3,525 training / 882 validation** source studies, fold 1, no zero-class target,
   and inspect sparse targets, warnings and audit samples. Save its entire output.
2. Import `05_Full_Cohort_Preprocessing.ipynb`. Attach the RSNA competition and the
   saved output from 04 only. Run on CPU and save the complete `p0` cache. Confirm
   `complete=true`, `failed=0`, `processed=4407`. The bounded cache cannot be used
   for this step. Based on the observed 752-study preprocessing time (~46 minutes),
   the full run may take several hours; measure the actual Kaggle session.
3. Import `06_Long_Attention_Training_and_Evaluation.ipynb`. Attach the saved
   complete cache from 05 and enable GPU. Run the default `r01_target_attention`
   with **200 maximum epochs**, **12 minimum epochs**, **12-epoch early-stopping
   patience**, batch size **2** and gradient accumulation **4** (effective batch
   approximately 8). Checkpoints are selected by validation macro BCE. Save all
   outputs (`best.pt`, `last.pt`, predictions, metrics and configuration).
4. Run notebook 06 again on the same cache after changing only `EXPERIMENT` to
   `r00_resnet_control`. Save this matched full-cohort control separately.
5. For comparison without retraining, attach both saved long-run outputs in a new
   notebook-06 session, set `RUN_TRAINING=False`, set `RUN_DIRS` with the control
   directory first, and run. Inspect per-target counts, 12-target AUC, BCE/AP/Brier,
   paired group-bootstrap interval and failure cases. A negative or uncertain
   comparison is a valid result; do not repeatedly adjust to fold 1.

## Go/no-go checks

- Preparation: all 4,407 source studies retained, no zero-class target in fold 1;
  explicitly flag MCL and synovitis with only two negatives.
- Preprocessing: every selected study has a usable cache entry, with zero decoding
  failures and `complete=true`; inspect `cache_audit.json` before training.
- Training: GPU is active, source/cache/configuration IDs are saved, `best.pt` and
  `validation_predictions.csv` exist, all 12 per-target rows report support.
- Comparison: same validation IDs, labels, groups, and cache across `r01` and `r00`;
  report paired intervals and uncertainty, not just a single macro score.

The local machine has CPU-only PyTorch and no configured Kaggle CLI session. The
new notebooks were generated and validated locally; the full MRI preprocessing
and GPU training require a Kaggle session and are not yet executed.
