# D01 and S01 three-epoch results; two next research paths

Date: 2026-10-13. These are 12-finding AUCs against the frozen **report-proxy development labels**, not Kaggle leaderboard or expert-label scores. No competition submission was made.

## Matched result

Kaggle marks both [D01](https://www.kaggle.com/code/aymenslimani/rsna-knee-v2-d01-full-train) and [S01](https://www.kaggle.com/code/aymenslimani/rsna-knee-v2-s01-full-train) COMPLETE. Output summaries say `status: complete`. The saved checkpoint SHA-256 values, label/split hashes, and prediction UID order were checked against the downloaded files. Each run completed three epochs on the same 3,428 training and 863 validation studies; there were zero failed studies. Validation consists of 839 groups. The 58 expert-labeled cases were not used.

| Three-epoch run | Change from P00 | Macro-AUC | Difference from P00 | Paired group-bootstrap 95% CI | Masked BCE | GPU-script time | Peak reserved VRAM |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| P00 | reference: one series/plane, 4 train and 16 inference windows/plane | 0.801735 | — | — | 0.483132 | 2.39 h | 760 MiB |
| D01 | 8 training windows/plane | 0.793934 | -0.007801 | [-0.014713, -0.001354] | 0.495228 | 3.24 h | 982 MiB |
| S01 | up to 2 series/plane sharing P00's window budget | 0.792733 | -0.009002 | [-0.017304, -0.001241] | 0.493847 | 4.09 h | 760 MiB |

Intervals use 1,000 valid paired bootstrap draws over the 839 groups. Both variants underperform P00 at the same epoch count. Neither is promoted. The S01 result tests **two-series selection under the same window budget**; it does not rule out a separately tested sequence-aware or larger-budget design.

| Finding | P00 | D01 | S01 |
| --- | ---: | ---: | ---: |
| ACL | 0.729496 | 0.727312 | 0.695581 |
| MCL | 0.685112 | 0.700168 | 0.694120 |
| Medial Meniscus | 0.825929 | 0.794106 | 0.777711 |
| Lateral Meniscus | 0.749731 | 0.739204 | 0.745717 |
| Medial OA | 0.874987 | 0.854396 | 0.849427 |
| Lateral OA | 0.825246 | 0.814409 | 0.818584 |
| PF OA | 0.796872 | 0.795695 | 0.795962 |
| Effusion | 0.847167 | 0.833748 | 0.849903 |
| Synovitis | 0.955435 | 0.950617 | 0.944595 |
| Baker's | 0.768036 | 0.755268 | 0.749995 |
| Contusion | 0.765573 | 0.757481 | 0.773488 |
| Fracture | 0.797242 | 0.804806 | 0.817716 |

Checkpoint SHA-256: P00 `5dba7e9eceda646edcde3dc37f5371dadf20fb186cd999319d834bb7996656f4`; D01 `86e6f9296b3e34cc514cc0453ace8ee74ce9a94301c95152432a3602a77209c6`; S01 `99560aaef2db288671c223f39378569814877c9338206ac61cbabad120980538`.

The [E01 six-epoch result](E01_SIX_EPOCH_RESULTS.md) is 0.821538 on the same proxy validation and remains the artifact-verified development leader. It is a duration result. Do not use E01 as the comparator to declare whether D01 or S01 improved P00 at three epochs. E01's final summary-status caveat also remains documented.

## External evidence and limits

The [public 0.949 CoAtNet recipe](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/746792) combines 16 epochs, denser windows, finding-level attention, soft report targets, image crops, checkpoint averaging and some external OAI data. It is a useful design inventory, but those changes are confounded and its public leaderboard score cannot be compared numerically with our report-proxy AUC. Its own tests found fewer inference windows and unqualified flips harmful. [A separate public CoAtNet implementation](https://github.com/mathischmp/challenge_rsna) reports physical-field-of-view cropping and target-specific attention, while also reporting negative results for mismatched training/inference slice counts and arbitrary flips. [A public ResNet ensemble implementation](https://github.com/NTejas-1/RSNA-Knee-Abnormality-Detection) uses laterality-normalized physical crops and RadImageNet weights; its own score improvement from adding the extra models was small, so the claimed weights benefit is only a hypothesis for us. The [RadImageNet weights source](https://github.com/BMEII-AI/RadImageNet) is publicly inspectable; provenance, license, exact normalization and offline availability require a local audit before use. A [Kaggle soft-label discussion](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/734105) reports improvement in one modest controlled setup and an inconclusive stronger-model replication. Reddit threads located for this competition were largely recruitment or broad model suggestions; none supplied a reproducible, controlled improvement claim we can promote to evidence.

## Plan A: improve supervision and optimization, preserving the P00 model

This is the lower-cost path and the first recommendation because E01 was still improving through epoch six.

1. **A0: label semantics audit, no GPU.** Verify probability range, missingness masks, `p=0.5` uncertainty, report-language distribution and source hashes. Freeze exact train/validation labels and split. Keep the 58 expert cases sealed.
2. **A1: soft-target ablation.** Retain P00's ResNet18, geometry, sampling, augmentation, validation labels, seed, optimizer, schedule, and three epochs. Change only the training target from thresholded observed 0/1 to observed report probability in masked BCE. Keep missing and `p=0.5` uncertainty masked. Compare directly to three-epoch P00 on the same 863 cases, report all 12 AUCs, BCE and paired group-bootstrap interval. This proxy-label comparison can favor calibration to the report source; any gain remains provisional.
3. **A2: optimizer ablation, separately.** If A1 is sound, freeze the better label choice and test separate encoder/head learning rates, with P00-style single rate as its **same-epoch, same-label** control. Choose rates before looking at validation results, and keep model/geometry/sampling fixed. Do not attribute a combined change to either factor.
4. **A3: duration confirmation.** Only for a promising candidate, run a matched six-epoch version and compare with a six-epoch control built from the same label and learning-rate recipe. E01 gives the existing six-epoch hard-label/single-rate reference when those settings are unchanged. Freeze the winner on weak validation before the single predeclared expert-label check.

Before a full run, require CPU data/label/inference checks, one bounded GPU pilot, a measured runtime projection under the approved per-run stop cap, and a verified offline weight path. Preserve the nine-hour final inference window. No new run is launched by this document.

## Plan B: improve image representation and anatomy handling

This has greater possible upside and greater implementation/runtime risk. The first two tests are separate three-epoch ablations, both against P00.

1. **B0: physical-geometry audit, no GPU.** Recheck DICOM pixel spacing, orientation, laterality, 140 mm field-of-view feasibility, interpolation, and whether all three planes remain anatomically valid. Save examples and per-series coverage statistics. Reject crops that lose clinically relevant anatomy. Confirm identical preprocessing in offline inference.
2. **B1: physical-FOV crop.** Retain P00 ResNet18, ImageNet weights, 4/16 windows, hard-label training, optimization, and three epochs. Change only the 224-pixel input crop from the present pixel-based geometry to a documented 140 mm physical-FOV crop. Compare with P00 at three epochs. If 140 mm is unsuitable for a plane, determine the rule from training-set geometry before the validation run.
3. **B2: medical pretraining, controlled pair.** On the selected frozen crop geometry, run RadImageNet ResNet50 and ImageNet ResNet50 with the same architecture, input statistics, head, labels, split, seed, optimizer, and three-epoch budget. The comparison isolates weight provenance; comparison to P00 additionally mixes architecture and is descriptive only. Gate on license, exact weight hash, load coverage, CPU forward/backward and T4 pilot runtime. If a 3-epoch ResNet50 run cannot fit the approved cap, stop rather than quietly change the protocol.
4. **B3: finding-aware pooling or longer schedule, only after B1/B2 evidence.** Test one choice at a time with a same-architecture, same-duration control. A 6-epoch candidate is compared with a 6-epoch control, not a 3-epoch result. Validate any laterality mirroring separately; do not apply arbitrary flips.

For either path, publish macro and per-target report-proxy AUC, paired interval, GPU time, peak VRAM, failed studies and checkpoint hash after each full experiment. A higher public recipe score is no guarantee against this validation. Prepare an offline Kaggle notebook and make a final submission only after separate user approval.
