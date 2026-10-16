# RSNA research decision and updated plan — 2026-10-09

## Decision

**Go:** provenance/label audit, fixed-split design, DICOM/runtime pilot, and an offline submission smoke test.

**No-go for now:** expensive architecture expansion or a claim that our next model can beat the current public leaderboard. On 2026-10-09, `kaggle competitions leaderboard rsna-knee-abnormality-detection --show --page-size 10` returned a public top score of **0.964**. That is a moving public score, measured on roughly 30% of hidden test cases. Our best saved **0.794998** is a nine-target *report-label* screening AUC, so the two numbers do not measure the same outcome. We have no completed full-cohort MRI model or scored competition submission from this project.

The independent Astra review agrees with the no-go decision and recommends one matched `r00` versus `r01` experiment after the label/split gates below. It did not certify a leaderboard-winning strategy. Council opinion is a critique, not performance evidence.

Review provenance: this checkpoint used Sol's direct artifact/source checks plus one independent Astra model review. It was **not** a three-model council, and no Luna research agents or GPU training runs were launched. The two attached prompt documents and workflow image were treated as proposals to evaluate, not as evidence that their claims had been executed.

## Grounded facts and corrections

| Claim | Status | Evidence and consequence |
|---|---|---|
| Competition objective is mean ROC AUC over 12 findings | Verified | [Official evaluation](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview/evaluation). Select experiments for this outcome, while using BCE as a prespecified checkpoint rule if needed. |
| Submissions require an offline Kaggle Notebook, `submission.csv`, and <=9 hours | Verified | [Official code requirements](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview/code-requirements). Three visible test studies cannot validate hidden-test runtime. |
| Training data has 4,407 studies, 24,371 series, and 58 fully observed expert-labeled studies | Verified locally | Downloaded competition `train.csv` and `train_series.csv` in ignored `results/competition_metadata/`. `test.csv` contains UID only; reports are unavailable at inference. |
| Expert labels are independent image readings, not report transcriptions | Verified host statement | [Host clarification](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733826). Report-derived labels are imperfect proxies for the target outcome. |
| Best saved local run is `r01` 0.794998 AUC | Verified with scope | `results/screening_comparison/summary.csv`: 240-study validation, nine eligible report-label targets. No verified Kaggle score follows. |
| `l02` is unusable because BCE exceeds a constant baseline | Incorrect as a blanket conclusion | `l02` has 0.767884 nine-target AUC. Its 0.474588 BCE versus constant 0.470891 is a separate diagnostic; it does not nullify AUC ranking. Keep the endpoint and checkpoint-selection rule distinct. |
| The 0.949 public single model is independently validated on Gold58 | Unsupported | [Published notebook](https://www.kaggle.com/code/nartaa/rsna-knee-0949-anatomical-mirror) says Gold58 was reused for selection/development. Its 0.949 is author-reported *public leaderboard* performance. |
| Public checkpoint availability proves external-data compliance | Unresolved | The model credits OAI training data. The [host's conditional OAI clarification](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/741819) hinges on generally accessible, free access without institutional review. Verify the exact checkpoint's training-data provenance and terms before using it as an eligible competition baseline. |
| A novel attention/region module will beat 0.964 | Hypothesis only | The pasted proposals specify directions, not measured improvements or established novelty. No GPU run is justified by that claim alone. |

## New artifact finding: expert-study overlap in prepared weak-label folds

`04_Full_Cohort_Readiness.zip` contains an `experiment_manifest.csv` with all 4,407 source studies and all 58 expert-labeled UIDs. The expert UIDs are distributed across folds 0–4 (10, 15, 16, 9, 8 respectively; fold 1 has 15). The manifest's 12 `y_*`/`w_*` columns are not simply the expert labels: only 503 of 696 `y_*` values numerically match the competition labels, and many corresponding weights are zero. For example, only 3/58 MCL and 2/58 Lateral OA labels have positive manifest weight. This is consistent with report-derived, masked supervision and needs a source-code/provenance audit before any expert-label evaluation. A model trained on these images—even with weak labels—cannot be called independent of those 58 images.

The preparation metadata reports `manual_review_required` and uses normalized report grouping. That prevents exact-report cross-fold collisions, but it does not prove patient independence or rule out duplicated MRI studies under different reports.

## Prioritized execution

1. **Freeze the data and validation contract (now).** Record hashes of the competition CSVs, prepared manifest, cache metadata, and source code. Audit how the 58 expert studies enter training and validation, whether their expert labels were accessed by any label-generation step, and whether duplicate images/patients cross folds. Define a fixed weak-label development split and a separately labeled, explicitly historical 58-case diagnostic. No result from the reused 58 can be called an untouched test set.
2. **Pilot inference/runtime before full training.** Use representative train DICOM studies through the exact test-time preprocessing path, including transfer-syntax coverage, geometric ordering, laterality, missing slots, GPU precision, and output validation. Measure seconds per study and extrapolate with substantial headroom to the hidden cohort; the three visible test studies are just a smoke test. Use a supported T4 precision mode and inspect fallback counts.
3. **Complete one matched full-cohort comparison.** Finish a production `p0` cache, then train `r00_resnet_control` and `r01_target_attention` with the same cache, fold, weak-label source, seed, training budget and checkpoint-selection rule. Save checkpoints, predictions, per-target support, hashes and runtime. Compare paired 12-target weak-label AUC/BCE and report uncertainty; do not rename it expert AUC.
4. **Test supervision quality before model complexity.** Review the public label sources and terms. Compare labelers on the 58 expert cases only as a historical diagnostic, with uncertainty and per-target errors. Freeze a teacher before MRI experiments. A report labeler must handle absence, negation, uncertainty and the stricter image-derived definitions stated by the host.
5. **Only then run one-factor MRI ablations.** Pilot semantic sequence slots from DICOM headers, laterality normalization, full field of view and denser slice coverage separately. Keep source labels, fold, architecture and budget fixed. Promote only changes that improve a prespecified endpoint and satisfy inference runtime.
6. **Consider a stronger backbone after a confirmed gain.** Compare a single pretrained DINOv2 or compact CoAtNet reader against the matched control. Treat target attention, region routing and ensembling as existing ideas unless a specific new mechanism survives prior-art review and isolated ablation. Fit blend weights only on out-of-fold predictions; the public leaderboard is a secondary check.
7. **Submit an eligible notebook.** Freeze the chosen recipe, attach pinned weights, run offline on Kaggle within nine hours, assert exact UID/schema/probability validity, save `submission.csv`, commit and submit the notebook version. Record public score, runtime, full provenance and fallback count. The final submission deadline is [October 22, 2026 at 23:59 UTC](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview/evaluation).

## Reversal condition

The no-go on expensive innovation changes if a matched full-cohort result, valid expert-label diagnostic, and end-to-end inference pilot show a reproducible gain with a plausible remaining gap to the current leaderboard. A public notebook's score or an untested architecture sketch alone does not satisfy that condition.
