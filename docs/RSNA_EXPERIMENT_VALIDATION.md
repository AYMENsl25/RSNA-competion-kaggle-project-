# RSNA experiment-suite validation

Validation date: 2026-09-15.

## Verified locally

- `scripts/build_experiment_notebooks.py` generated all three experiment notebooks with valid notebook schema and Python syntax.
- `scripts/validate_experiment_notebooks.py` verified each embedded runtime package and imported preparation, preprocessing, registry, models, losses, training and comparison modules.
- The complete local suite passed **116 tests** in 53.49 seconds with MONAI 1.5.1 installed.
- Synthetic checks exercise report-label/fold preservation, training-only corrections, validation-edit rejection, target coverage and prior diagnostics.
- Synthetic DICOM checks exercise spatial ordering, plane/missing-slice masks, physical crop failures, source/config mismatch, corrupt-cache detection, resume checks and incomplete pilot metadata.
- Training checks exercise masked BCE, positive-weight BCE, MONAI focal loss, asymmetric loss, BCE plus ranking, label smoothing, attention/mean-max padding masks, MONAI 3D backward propagation, per-target AUC eligibility, checkpoint output and epoch-boundary resume.
- Comparison checks exercise aligned validation identities/data, mismatched-cohort rejection, prediction tampering detection, per-target correlations, paired group bootstrap and exploratory equal-probability averaging.
- The older v2 notebooks still passed their separate syntax/schema/runtime-import validator.

These tests use synthetic data. They do not demonstrate real competition DICOM compatibility, GPU memory fit, preprocessing runtime, clinical label correctness, validation improvement or a Kaggle leaderboard score.

## Required Kaggle gates

1. Run experiment readiness and manually inspect target/status examples, class coverage and grouping warnings.
2. Run the 12-study p0 pilot; inspect montage, selected series, masks and every failure. Confirm dependency and compressed-DICOM decoding support.
3. Run p0 on the entire fixed bounded cohort. Training requires `complete=true` and rejects a pilot or silently reduced cohort.
4. Run `s00_tiny_smoke`, then `r00_resnet_control`. Confirm finite losses, all intended active targets, checkpoint hashes and improvement over the constant training-prior BCE.
5. Run every promoted candidate against the same cache/cohort and compare with paired results. Confirm on another frozen fold before treating the result as stable.

If p0 cannot complete because studies have no usable series under the strict geometry rules, review `cache_audit.json`. Resolve the parsing rule or create a documented image-QC exclusion cohort before all recipe comparisons; do not drop a different set of studies independently for each preprocessing recipe.
