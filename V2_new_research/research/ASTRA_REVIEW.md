# Astra scientific review — 2026-10-09

## Decision

**Conditional GO** for fresh data/report/DICOM audits, an offline runtime pilot and one compact baseline after gates. **NO-GO** for running the entire 33-item backlog or claiming a leaderboard gain. This is an independent scientific critique, not an experimental score. The user requested to see the decision before confirming execution of the proposed model study.

## Agreed priorities

1. Protect the 58 expert studies and detected linked groups from fitting. Use one grouped development split and a fixed binary/masked weak-label evaluation reference. ROC-AUC requires observed binary targets; uncertain and unaddressed cells are excluded. Do not redefine evaluation labels with each new report teacher.
2. Audit a prespecified language-stratified sample of reports without viewing model outputs. This checks text extraction only, not MRI truth. Fit learned teachers within training partitions.
3. Measure available GPU hours, decode/cache time, seconds per study and offline notebook runtime before a full model run.
4. Run M00: compact three-plane 2.5D image baseline with a fixed max-pooling head. After a complete baseline, prioritize M01: 12 finding-specific attention queries with all other factors matched.
5. Keep N09 in the backlog. Consider hard versus soft targets only if the soft values have a defensible, audited meaning and the observed-cell masks/weights are identical. Otherwise M02, more training-window coverage, is the stronger alternative when resources allow.

## Dissent and limitations

- Tokenizer fragmentation and truncation **measurement** is useful now; custom tokenizer training and text-domain adaptation are lower priority before the deadline.
- A public GitHub implementation's numerical results and runtime are author-reported. Its CoAtNet, crop and learning-rate settings are candidate hypotheses, not validated settings for this study.
- The review independently confirmed organizer context from [RSNA](https://www.rsna.org/artificial-intelligence/ai-image-challenge/knee-mri-ai-challenge), prior art for [attention MIL](https://proceedings.mlr.press/v80/ilse18a.html) and available [MONAI transforms](https://monai.readthedocs.io/en/stable/transforms.html). Kaggle's official page text was unavailable to the reviewer through its web tool; competition-specific dates and rules rely on the authenticated CLI and source checks in `competition_constraints.md` and must be captured again before training/submission.
- The backlog is a menu of isolated tests. Calendar and compute likely permit a baseline and one full comparison, not a broad architecture sweep.

## Reversal conditions

Repair report supervision first if coverage or severity-aware extraction is poor; increase window coverage before changing pooling if anatomy is often missed; optimize decoding if runtime is too high; defer laterality/position modeling if DICOM geometry is unresolved; keep the simpler model if gains vanish across seeds or contradict the protected expert check.

## Confirmation requested

Confirm the compact 2.5D baseline, one protected 58-study check, M01 as the sole budgeted matched full comparison, and the proposed compute caps in `PLAN.md`. N09, M02 and the remaining registry stay a research backlog.
