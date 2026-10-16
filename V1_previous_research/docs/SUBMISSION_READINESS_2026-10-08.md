# RSNA submission readiness — 2026-10-08

## Verified results

These numbers use different labels and cohorts and must not be compared as one leaderboard series.

| Run | Scope | Macro AUC | Status |
|---|---|---:|---|
| r01 target attention | 240-study screening validation, 9 evaluable report-label targets | 0.794998 | Best saved screening run; macro BCE 0.354079 |
| r03 dense slices | Same screening validation | 0.748595 | Secondary |
| l02 MONAI focal | Same screening validation | 0.767884 | Highest AUC among the six locally saved checkpoint ZIPs, but BCE 0.474588 is worse than the constant baseline 0.470891 |
| r00 ResNet control | Same screening validation | 0.695371 | Control |
| 4B-1B v1.2 | Historical 58-case gold audit, 12 targets | 0.699358 | Descriptive historical baseline |
| 4B-2B1 | Historical gold audit, 12 targets | 0.644729 | Descriptive historical result |

The remembered ~0.82 appears in the old ledger as **Effusion target AUC 0.816149** for 4B-2B1 (v1.2 Effusion: 0.814907). It is not an overall score. No verified Kaggle leaderboard score or submitted file is present locally.

## Current blockers

1. The saved screening ZIPs include validation checkpoints for six runs, but not r01 or r03; their validation predictions and metrics are in the comparison receipt and executed notebooks. Find/download the r01 run output if using it. The best locally available checkpoint by AUC is l02, but it has worse BCE than a constant baseline and should not be chosen merely because it is available.
2. No `submission.csv`, test images, competition `sample_submission.csv`, or final refit checkpoint is present locally. Validation predictions cannot be submitted as test predictions.
3. No Kaggle authentication was detected. The Kaggle CLI is not installed; package installation failed because this shell cannot resolve PyPI.
4. The v2 submission notebook deliberately requires a complete production preprocessing metadata file, matching checkpoint hashes, and GPU inference. It is not configured with final artifacts yet.
5. `04_Full_Cohort_Readiness.zip` contains preparation for 4,407 source studies (3,525 train, 882 validation) and a `manual_review_required` status. No full-cohort preprocessing, training, or test-inference output was found among the local ZIPs.

## Ordered path to a submission

1. Verify the current competition rules and the actual sample submission schema. This is a notebook-submission competition: the notebook must run offline within nine hours and write `submission.csv`.
2. Retrieve the saved r01 notebook output and its `best.pt`, run configuration, and matching preprocessing cache metadata. If the full cohort experiment was completed on Kaggle, retrieve that output instead and compare it against r00 using the same fold.
3. Freeze the chosen recipe and run the final refit. Preserve checkpoint and preprocessing SHA-256 hashes.
4. Run `notebooks/v2/08_Submission_Inference.ipynb` with the competition data attached and its configuration explicitly populated. It validates UIDs, target columns, finite probabilities, and range before saving `submission.csv`.
5. Inspect the generated CSV against `sample_submission.csv` and inspect `test_decode_audit.csv` and `submission_provenance.json`.
6. Save and run an internet-off Kaggle notebook version, then submit that notebook version through Kaggle. Record the submission ID, public score, runtime, notebook version and checkpoint hashes. A direct local CSV CLI submission does not satisfy this competition's notebook requirement.

Sources: `results/screening_comparison/summary.csv`, `docs/RSNA_FULL_COHORT_LONG_RUN_20260917.md`, historical `EXPERIMENT_LEDGER.md`, and `src/rsna_v2/submission.py`.
