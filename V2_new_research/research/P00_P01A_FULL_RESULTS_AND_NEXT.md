# P00 and P01A full-run audit — 2026-10-12

Both private Kaggle kernels completed three epochs: [P00](https://www.kaggle.com/code/aymenslimani/rsna-knee-v2-p00-full-train) and [P01A](https://www.kaggle.com/code/aymenslimani/rsna-knee-v2-p01a-full-train). We downloaded their checkpoints, summaries, logs and all epoch prediction CSVs to `outputs/p00_full_train_v1` and `outputs/p01a_full_train_v1` and verified SHA-256 hashes. These are **frozen report-proxy development AUCs**, not Kaggle leaderboard or expert-label results.

| Model | Epoch 1 | Epoch 2 | Epoch 3 / best | Best masked BCE | Script GPU seconds | Peak reserved VRAM | Failed studies |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M00 random-init ResNet18 | 0.681255 | 0.724415 | 0.739719 | 0.541696 | 9444.7 | 760 MiB | 0 |
| P00 pretrained ResNet18 | 0.758714 | 0.786871 | **0.801735** | **0.483132** | 8601.6 (2.39 h) | 760 MiB | 0 |
| P01A pretrained CoAtNet-0 | 0.668360 | 0.675099 | 0.687420 | 0.578095 | 9865.2 (2.74 h) | 2948 MiB | 0 |

P00/P01A each processed 3,428 training and 863 eligible weak-validation studies per epoch, zero failures. Both stayed below their eight-hour GPU caps. P00 had 2/5,142 skipped AMP updates; P01A had 5/5,142, each below the declared 1% gate. Best checkpoint epoch is 3 for both. P00 SHA-256 `5dba7e9eceda646edcde3dc37f5371dadf20fb186cd999319d834bb7996656f4`; P01A SHA-256 `c7b58cf14706756a7e65197488199dcd8f50f0a2675de8e2030c573d06fc60cf`.

## Per-target weak-validation AUC, best epoch

| Target | M00 | P00 | P01A |
| --- | ---: | ---: | ---: |
| ACL | 0.6653 | 0.7295 | 0.6500 |
| MCL | 0.6670 | 0.6851 | 0.6583 |
| Medial Meniscus | 0.6992 | 0.8259 | 0.6326 |
| Lateral Meniscus | 0.6984 | 0.7497 | 0.6538 |
| Medial OA | 0.7738 | 0.8750 | 0.7222 |
| Lateral OA | 0.7612 | 0.8252 | 0.7196 |
| PF OA | 0.7159 | 0.7969 | 0.6628 |
| Effusion | 0.7890 | 0.8472 | 0.6748 |
| Synovitis | 0.9145 | 0.9554 | 0.8475 |
| Baker's | 0.7050 | 0.7680 | 0.6564 |
| Contusion | 0.7383 | 0.7656 | 0.6778 |
| Fracture | 0.7492 | 0.7972 | 0.6933 |

P00 beat M00 and P01A on all twelve targets. Prediction files have exactly the same 863 unique ordered study UIDs, finite probabilities in [0,1], and the same verified split/label hashes. The paired **839-group, 1,000-draw** bootstrap in `outputs/pretrained_run_comparison.json` gives P00 minus M00 macro delta **+0.062016**, 95% percentile interval **[+0.049379, +0.075216]**; P00 minus P01A **+0.114315**, interval **[+0.094583, +0.135443]**. These intervals quantify sampling variability on this weak-label validation set; they do not account for label noise, a new cohort, or the Kaggle leaderboard.

## Decision and next gate

Retain P00 epoch 3 as the leading V2 development checkpoint. Do not pursue P01A further under the current recipe. P00's macro AUC and BCE improved in every epoch, so a fresh **E01 six-epoch P00** run is a reasonable next controlled experiment, with the same data, model and preprocessing but a newly planned six-epoch cosine schedule. The three-epoch checkpoint cannot be treated as equivalent to continuing a six-epoch schedule because its learning rate was annealed for three epochs and it does not save optimizer state. The observed 2.39-hour three-epoch run suggests about 4.8 hours for six, with a 20% margin about 5.7 hours, below an eight-hour cap; a fresh pilot/estimate and separate GPU-budget authorization are required before launch. Keep denser windows and finding-specific pooling separate and later. In parallel, prepare an offline Kaggle inference notebook for P00 and measure full test runtime/schema; do not submit or open the 58 expert labels. The public-report-proxy validation cannot establish leaderboard competitiveness.
