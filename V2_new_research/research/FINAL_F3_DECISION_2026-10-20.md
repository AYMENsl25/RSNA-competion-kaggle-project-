# Final candidate: F3 weight shrinkage

## Scoring submission update

Following the user's explicit instruction to continue and submit after disclosure of runtime uncertainty, F3 version 1 was submitted on October 20 at 19:58:37 UTC. Kaggle submission reference **57451681** is verified **PENDING**, with no score yet. This supersedes the earlier not-submitted status below; it does not establish that the original runtime gate passed. No changes to F0 or final-submission selections were made.

## Completed preview check

Live CLI reports F3 version 1 COMPLETE. Downloaded receipts confirm three visible studies, zero Raptor/R6 failures, successful Raptor-224 and small-reader stages, and 547.23 seconds elapsed inside the pipeline. Submission checksum, unique study IDs, twelve finite probability columns and [0,1] range passed local verification. This is execution validation, not an AUC result. The log contains failed optional pyjpegls installation attempts; the visible reader nevertheless completed. Unseen transfer-syntax coverage remains untested.

Quota after preview: 3.89 h remaining, 26.11 h used (verified CLI display). No leaderboard submission has been made. Full-runtime feasibility and whether the 1,800-second commit timeout applies to hidden scoring remain unresolved. Do not submit this version while claiming the original 30% full-run reserve has been demonstrated. Next deployment action is a single version-pinned submission only after these gates are resolved; otherwise retain the already-scored F0.

Predeclared 2026-10-20, before any F3 leaderboard result. The user's current instruction authorizes starting the experiment. F0 remains the incumbent.

## Decision

Stop Kaggle training. Test one inference-only candidate: shrink F0's per-finding Raptor-family weights halfway toward 0.5. Keep all checkpoint families, resolutions, TTA, within-Raptor weights and the small reader contribution unchanged. This may reduce overfitting of hand-selected finding weights; it may also dilute genuinely superior specialists. No positive gain, probability of success or private-rank improvement is established. A new architecture or toolkit is not a justified last-minute replacement.

| Finding | F0 Raptor weight (verified source) | F3 (predeclared design) |
|---|---:|---:|
| Lateral OA | 0.20 | 0.35 |
| PF OA | 0.60 | 0.55 |
| Synovitis | 0.80 | 0.65 |
| Baker's | 0.35 | 0.425 |
| Others | 0.50 | 0.50 |

The competing family gets the complement. The shrinkage coefficient is fixed, not selected from gold-58 or LB trials. Additional assertions reject missing main families and invalid output; they do not change valid-case predictions.

## Four lenses

1. **Senior action:** preserve the scored F0 artifact, distinguish score evidence from notebook titles, audit checkpoint lineage, stop selecting on burned gold-58, and separate weekly quota from competition runtime. Do not spend the last quota on a weak new member simply because a parallel slot exists.
2. **Raw evidence:** exact submitted kernel versions, raw submission history, per-family prediction CSVs with aligned study IDs, actual worker coverage/fallback receipts, checkpoint hashes, and original notebook source. The three visible test studies cannot measure hidden-test throughput. No hidden labels are available.
3. **Specific methods:** `kaggle quota`, `kaggle kernels push --timeout`, source-cell SHA-256 comparison, pandas percentile ranks, a fixed shrinkage coefficient, and analytical synthetic checks of the blend delta. A patient-group paired bootstrap would require honest held-out predictions; we do not have honest F0 OOF. Prediction correlation alone is not a quality gate.
4. **External arbiters:** Kaggle run logs establish execution and coverage only; a single predeclared public score establishes public performance only; private scoring is the final generalization arbiter. Source agreement verifies implementation, not accuracy. Independent confirmation data cannot be manufactured by splitting data already used by public parents.

## Targeted search: decision-changing findings and limits

- Live CLI confirmed F0 public 0.958 and F1 public 0.954 (verified account records in this session). Latest score-sorted public-code inventory was inspected, with source checks of the leading Versia and Helios variants. Versia reuses the same stack. Helios varies per-finding weights and other inference details; its title's 0.9586 is not an isolated validated weight ablation. Do not copy several changes at once. Sources: [Versia](https://www.kaggle.com/code/evgendvorkin/rsna-versia-7), [Helios](https://www.kaggle.com/code/heliosli/this-is-best-public-dataset-new-version-0-9586).
- Scope limitation: CLI notebook listings do not expose numerical scored-version history. This is NOT a verified exhaustive audit of every notebook above 0.950. Existing earlier research and fresh source inspections found no verified independent drop-in member; that is not proof none exists.
- [MONAI model zoo](https://github.com/Project-MONAI/model-zoo), [TorchIO augmentation](https://docs.torchio.org/latest/transforms/augmentation/), and [timm](https://huggingface.co/docs/timm/en/index) supply useful infrastructure, not evidence of an offline, ready-to-use classifier matching this competition's labels. No new packages are necessary for F3. MRI artifact augmentation without retraining is a distribution change, not an established improvement.
- [MRNet](https://pmc.ncbi.nlm.nih.gov/articles/PMC6258509/) targets abnormality, ACL and meniscus tasks; it is not a direct replacement for this competition's output schema.
- Prior solutions support complementary trained representations and anatomy-aware processing, not a guaranteed final-week blend trick: [2019 winner](https://github.com/SeuTao/RSNA2019_Intracranial-Hemorrhage-Detection), [2022 third place](https://github.com/darraghdog/RSNA22), [2023 winner](https://github.com/Nischaydnk/RSNA-2023-1st-place-solution), [2024 fifth place](https://github.com/siwooyong/RSNA-2024-Lumbar-Spine-Degenerative-Classification). Exact final-week chronology and second-slot decisions were not established; do not invent them.

## Budget and runtime gate

CLI quota on Oct 20: **4.05 h remaining**, **25.95 h used**, reset Oct 24 (verified displayed values). With 30% reserve, total permitted remaining expenditure is at most **2.835 h** (derived from rounded CLI values).

Start only a **1,800-second capped preview** (design cap; 0.5 session-hours). This leaves at least approximately 3.55 h before any other account activity, subject to Kaggle accounting. Preview expectation is minutes, not a measured hidden runtime. All model inference code stays unchanged. Existing F0 receipts cover only three studies.

The plan's FULL hidden-inference budget gate is **NOT PASSED**. A [community answer](https://www.kaggle.com/questions-and-answers/453105) says hidden scoring does not consume weekly GPU quota, but this is not sufficient verification of current account behavior. The [official GPU documentation](https://www.kaggle.com/docs/efficient-gpu-usage) explains quota but does not establish this exemption in the material inspected. Do not represent a short preview as a full-test budget test. Do not submit the 30-minute-capped version for hidden scoring without resolving timeout semantics and full-runtime feasibility.

Before scoring: establish a conservative full-runtime bound and whether hidden scoring is charged; include measured preview cost and any necessary further commit. If charged, total must fit 2.835 h; if separate, still honor the nine-hour competition ceiling and the plan's runtime reserve. In either interpretation, no measured full-runtime upper bound currently exists. Retain F0 if this cannot be resolved in time. No claim that every plan constraint has been met.

## Predeclared decision and kill rules

- At most one F3 scored submission, only after runtime gate passes. No weight search after seeing its score.
- Kill on missing family, nonfinite predictions, UID mismatch, timeout, unverified runtime feasibility, or public score below F0's displayed 0.958.
- At exactly displayed 0.958, F3 may be a second-slot hedge only if technically valid; no claim of a gain. Above 0.958, select F0 and F3, while acknowledging public selection bias. Below it, keep F0; do not choose known-worse F1 merely to fill a second slot. Check the site's actual final-selection state; this document does not change selections.
- F2 excluded unless it obtains independent expert-label evidence of incremental benefit over the full F0 with matched provenance and passes deployment/runtime checks. Weak-label agreement or parent decorrelation cannot pass this gate.

## Calendar (UTC; planned, not completed)

| Time | Work | New Kaggle session budget | Scored submissions |
|---|---|---:|---:|
| Oct 20 19:00-20:00 | Research, fixed recipe, build and analytical checks; launch bounded preview | <=0.5 h | 0 |
| Oct 20 20:00-21:00 | Inspect receipts/coverage, resolve quota and hidden-runtime evidence | 0 h unless separately justified within total cap | 0 |
| Oct 20 21:00-22:00 | If runtime gate passes, one F3 submission; otherwise hold F0 | Must fit remaining derived cap | <=1 |
| Oct 20 22:00-Oct 21 09:00 | No training or parameter search; allow accepted scoring to finish | No new runs | 0 |
| Oct 21 09:00-10:00 | Read score and apply fixed kill/selection rule | 0 h | 0 |
| Oct 21 10:00-Oct 22 12:00 | Preserve artifacts; resolve only concrete execution failures, no new hypotheses | 0 h planned | 0 |
| Oct 22 12:00-13:00 | Verify final-submission selections in Kaggle | 0 h | 0 |
| Oct 22 13:00-23:59 | Freeze; no new launches | 0 h | 0 |

No unattended polling agent is created. Local work is source validation/integration; existing F2 work is untouched. No new laptop training is started.

## Artifacts and checks

- `scripts/build_f3_final.py`: deterministic builder preserving original F0.
- `notebooks/f3_weight_shrinkage/f3.ipynb`, metadata and SHA-256 manifest.
- `outputs/f3_synthetic_check/check_result.json`: analytical blend delta and incomplete-family rejection passed using the existing project environment. These synthetic outputs are not competition submissions.
- No toolkit installation needed. Source checks assert all original inference cells are unchanged and parse Python cells; GPU execution remains the preview's responsibility.
