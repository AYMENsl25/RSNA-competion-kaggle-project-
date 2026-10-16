# E01 renamed six-epoch result — October 12, 2026

The completed Kaggle kernel is [`aymenslimani/rsna-knee-v2-e01-six-epochs`](https://www.kaggle.com/code/aymenslimani/rsna-knee-v2-e01-six-epochs). The original singular-slug kernel was canceled; its state was not the renamed run's result.

The renamed run saved all six full training and validation epochs. Best epoch 6 reached **0.821538** macro-AUC and **0.467434** masked BCE on the frozen 863-study public report-proxy validation. Epoch macro-AUCs were 0.748988, 0.764744, 0.794022, 0.808040, 0.817159 and 0.821538. This is not an expert-image or competition leaderboard score.

| Finding | P00 three epochs | E01 six epochs |
| --- | ---: | ---: |
| ACL | 0.729496 | 0.728244 |
| MCL | 0.685112 | 0.698193 |
| Medial Meniscus | 0.825929 | 0.854571 |
| Lateral Meniscus | 0.749731 | 0.761798 |
| Medial OA | 0.874987 | 0.886863 |
| Lateral OA | 0.825246 | 0.846795 |
| PF OA | 0.796872 | 0.813204 |
| Effusion | 0.847167 | 0.849761 |
| Synovitis | 0.955435 | 0.969587 |
| Baker's | 0.768036 | 0.858801 |
| Contusion | 0.765573 | 0.783310 |
| Fracture | 0.797242 | 0.807332 |

The matched E01-minus-P00 macro-AUC gain is **+0.019803**. A 1,000-draw paired bootstrap over 839 validation groups gave a 95% percentile interval of **[+0.012323, +0.026682]**. Eleven targets improved and ACL declined by 0.001252. Identical validation UIDs, label and split hashes, 3,428 train and 863 validation studies, and the saved checkpoint SHA-256 `6b357912031969c267d670df1db304a0f554bb83ea31a6247ea1aef1dd062a60` were verified. There were zero failed studies in any epoch. The GPU script consumed 18,373 s (5.10 h), below the eight-hour hard cap; peak reserved VRAM was 760 MiB.

**Summary-status caveat:** Kaggle marks the kernel `COMPLETE`, but the output JSON says `status: incomplete` and `stop_reason: external_watchdog_deadline`. That handler ran about 2.5 seconds after the sixth epoch's saved summary, despite the measured time being far below its eight-hour deadline. The handler assigns that reason to any SIGTERM, so the signal source is not established. All six epoch prediction files and the best checkpoint exist and pass independent integrity checks. Treat E01 as the current **artifact-verified development leader**, while recording that its final status flag did not pass the original run gate. Avoid another costly rerun solely to repair the flag until the issue is isolated.

D01's 80/40-study GPU pilot passed with zero failures, verified input hashes, 982 MiB peak reserved VRAM, and a 10,350 s three-epoch projection including 20% headroom, below its 27,000 s stop limit. Its [full three-epoch run](https://www.kaggle.com/code/aymenslimani/rsna-knee-v2-d01-full-train) was launched and was running at the initial status check. It tests denser training-window sampling against three-epoch P00; compare D01 against both P00 and E01 after completion, without using the protected 58 expert labels.
