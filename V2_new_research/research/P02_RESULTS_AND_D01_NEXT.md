# P02 full result and next gate — October 12, 2026

P02's private Kaggle three-epoch full run completed. It changed P00's shared feature max pooling to a per-finding max of window logits; all other matched experimental settings, split, and report-proxy labels stayed fixed. This is development weak-label validation, not the Kaggle leaderboard.

| Run | 12-target macro-AUC | Masked validation BCE | GPU script time | Peak reserved VRAM | Failed studies |
| --- | ---: | ---: | ---: | ---: | ---: |
| P00 | 0.801735 | 0.483132 | 8,602 s (2.39 h) | 760 MiB | 0 |
| P02 | 0.712793 | 0.561238 | 10,265 s (2.85 h) | 758 MiB | 0 |

The same 863 validation studies, 839 groups, split checksum and label checksum were verified. Both checkpoint hashes match their saved summaries. P02 lost on all 12 targets. The paired P02-minus-P00 macro-AUC difference is **−0.088943**; a 1,000-draw paired group bootstrap 95% percentile interval is **[−0.109760, −0.070498]**. P00 remains the development leader.

| Finding | P00 AUC | P02 AUC |
| --- | ---: | ---: |
| ACL | 0.729496 | 0.635746 |
| MCL | 0.685112 | 0.673439 |
| Medial Meniscus | 0.825929 | 0.680557 |
| Lateral Meniscus | 0.749731 | 0.648126 |
| Medial OA | 0.874987 | 0.768280 |
| Lateral OA | 0.825246 | 0.746650 |
| PF OA | 0.796872 | 0.704470 |
| Effusion | 0.847167 | 0.707140 |
| Synovitis | 0.955435 | 0.844926 |
| Baker's | 0.768036 | 0.694795 |
| Contusion | 0.765573 | 0.722188 |
| Fracture | 0.797242 | 0.727194 |

P02 checkpoint SHA-256: `5c10c3e71b42aafc1dc78dd53d64cba3f7dae14902de9f709d3af24cdaf9ed58`. P02 completed all three epochs with zero training and validation study failures. The six-epoch E01 kernel reports `CANCEL_ACKNOWLEDGED` and has no completed output; it must not be interpreted as a performance result.

**Next:** D01's offline CPU check passed. Its bounded, one-hour GPU pilot was pushed as private Kaggle kernel version 2 on October 12 and was queued at the first status check. D01 doubles training windows per plane from four to eight while retaining P00's encoder, pooling, validation windows and recipe. Run its full three-epoch comparison only if the pilot reports zero failures, verified hashes and projected runtime below the 7.5-hour stop limit. Keep the 58 expert labels sealed and make no competition submission.
