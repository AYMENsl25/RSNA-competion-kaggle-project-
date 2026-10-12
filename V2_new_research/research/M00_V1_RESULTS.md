# M00 GPU version 1 — 2026-10-11 audit

Private Kaggle kernel: `aymenslimani/rsna-knee-v2-m00-three-epoch-training`, status COMPLETE. All three epochs finished. The metric is **12-target macro ROC-AUC against thresholded public report-derived labels** on the frozen weak-validation split. It is not a score on the 58 expert MRI labels or the competition leaderboard.

| Epoch | Macro AUC | Masked BCE | Wall seconds | Train / validation failures | Peak reserved VRAM |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.681255 | 0.596772 | 3167.6 | 0 / 0 | 758 MiB |
| 2 | 0.724415 | 0.563340 | 3172.7 | 0 / 0 | 758 MiB |
| 3 | **0.739719** | **0.541696** | 3103.2 | 0 / 0 | 760 MiB |

Total script time 9444.7 seconds (2.62 hours), below the eight-hour cap. GPU Tesla T4 (one used, two visible). Each epoch accounted for 3,428 usable training studies and 863 eligible weak-validation studies. The 65 excluded studies and 29 all-masked training studies were omitted. Two AMP updates were skipped over the run, below the declared 1% gate. Split and public label SHA-256 checks passed inside Kaggle. The best checkpoint is epoch 3, SHA-256 `783743cf59b6acea67b1fd3cf1b2ed11a9b818c615a69af9b6d61f73f785c721`, independently verified after download. All three prediction CSVs have exactly 863 unique UIDs in the frozen order, 12 finite probabilities in [0,1] per study.

## Best epoch per-target AUC

| Finding | AUC | Positive / negative support |
| --- | ---: | ---: |
| ACL | 0.6653 | 192 / 601 |
| MCL | 0.6670 | 137 / 630 |
| Medial Meniscus | 0.6992 | 469 / 359 |
| Lateral Meniscus | 0.6984 | 247 / 539 |
| Medial OA | 0.7738 | 327 / 356 |
| Lateral OA | 0.7612 | 254 / 396 |
| PF OA | 0.7159 | 429 / 310 |
| Effusion | 0.7890 | 534 / 244 |
| Synovitis | 0.9145 | 123 / 27 |
| Baker's | 0.7050 | 223 / 239 |
| Contusion | 0.7383 | 214 / 436 |
| Fracture | 0.7492 | 72 / 426 |

M01 may now run as the preapproved matched comparison if its architecture and standalone CPU gates pass. Keep the same split, proxy labels, preprocessing, training recipe, and three-epoch/eight-hour limits. Do not use the 58 expert labels for selection.
