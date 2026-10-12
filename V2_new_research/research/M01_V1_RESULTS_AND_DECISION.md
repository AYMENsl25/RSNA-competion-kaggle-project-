# M01 GPU version 1 and matched decision — 2026-10-11

Private Kaggle kernel `aymenslimani/rsna-knee-v2-m01-three-epoch-training` finished COMPLETE. Its best epoch was 3. All inputs, group split, label hash, preprocessing, training settings, three-epoch/eight-hour cap and validation study IDs match M00; the planned architecture difference is the twelve-query finding-specific attention pool in place of shared max pooling. These are **public report-proxy validation** results, not gold MRI or leaderboard scores.

| Model | Best epoch | 12-target macro AUC | Masked BCE | Script time | Peak reserved VRAM | Failed train / validation studies |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| M00 shared max | 3 | **0.739719** | **0.541696** | 9444.7 s (2.62 h) | 760 MiB | 0 / 0 |
| M01 finding attention | 3 | 0.690701 | 0.572107 | 9929.0 s (2.76 h) | 758 MiB | 0 / 0 |

M01 epoch macro AUCs: 0.635196, 0.671856, 0.690701. Every epoch accounted for 3,428 usable training and 863 eligible validation studies. Three AMP updates skipped over the run, below the 1% gate. The 65 excluded studies remained out. The downloaded M01 checkpoint SHA-256 matches the summary: `830e6df965c55028edb6ed78f2d2e3979df5985ae40eac1f0c612b45557ceda1`.

| Finding | M00 AUC | M01 AUC | M01 − M00 |
| --- | ---: | ---: | ---: |
| ACL | 0.6653 | 0.6260 | −0.0393 |
| MCL | 0.6670 | 0.6433 | −0.0237 |
| Medial Meniscus | 0.6992 | 0.6504 | −0.0488 |
| Lateral Meniscus | 0.6984 | 0.6431 | −0.0553 |
| Medial OA | 0.7738 | 0.7391 | −0.0347 |
| Lateral OA | 0.7612 | 0.7312 | −0.0301 |
| PF OA | 0.7159 | 0.6800 | −0.0358 |
| Effusion | 0.7890 | 0.7114 | −0.0776 |
| Synovitis | 0.9145 | 0.8299 | −0.0846 |
| Baker's | 0.7050 | 0.6723 | −0.0327 |
| Contusion | 0.7383 | 0.6806 | −0.0576 |
| Fracture | 0.7492 | 0.6812 | −0.0680 |

`compare_m00_m01.py` checked both source-file hashes, checkpoint hashes, identical ordered UIDs and finite predictions, reproduced the saved AUCs, and resampled the **839 validation groups** with replacement for 1,000 paired draws (seed 20261011). All draws had class support. M01 minus M00 macro delta was **−0.049019**, with a percentile 95% group-bootstrap interval **[−0.060970, −0.038909]**. This quantifies sampling variation on this proxy-labeled fold, not run-to-run training variation or expert-label uncertainty. Only one seed per architecture was trained.

**Development decision:** Select and freeze M00 as the candidate from these two budgeted models. Do not launch extra model experiments from this comparison alone. Next, prepare a reproducible offline Kaggle inference notebook using the frozen M00 checkpoint, verify visible test schema and runtime, then perform the one predeclared expert-label check after the model and evaluation script are frozen. The public report-label publisher reportedly used the 58 expert studies while choosing its label version; even an untouched local one-time check is not fully independent of that source. Final competition submission still requires explicit user approval.
