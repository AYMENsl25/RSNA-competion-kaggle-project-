# Final RSNA competition experiments — status at Oct 22, 2026

This records architectures, model results and submissions from the final competition push. Kaggle scores are public leaderboard scores unless explicitly labeled otherwise. Pending means no score had appeared in the account listing when last checked.

| ID | Architecture or change | Result | Evidence/status |
|---|---|---|---|
| F0 | Unmodified public stack: nartaa Raptor CoAtNet 384 and 224; heliosli R6 ConvNeXt-S four-fold knee-anchored model with flip TTA; goodpjw reader at low weight | **0.958 public LB** | Submission 57309738, complete; incumbent |
| F1 | F0 plus 0.2 rank weight of S1B, ConvNeXt-S ImageNet initialized, global view, teacher-denoised label v3 recipe | **0.954 public LB** | Submission 57366237; 0.004 below F0 at displayed precision |
| K1A | Raptor-384 fine-tune with consensus labels and parent predictions, low learning rate | Gold-58 AUC 0.920 vs parent 0.923; prediction rank correlation 0.986 | Development comparison; not submitted |
| K1B | K1A plus dual global and 1.5x joint zoom view, zero-initialized view embeddings | Gold-58 AUC 0.917; prediction rank correlation 0.988 | Development comparison; not submitted |
| K2A | Fresh ImageNet EfficientNetV2-S, 1.5x zoom, consensus labels, attention MIL | fp16 overflow/NaN; diagnostic Gold-58 score 0.586 | Failed run |
| K2B | Fresh ImageNet EfficientNetV2-S global view, consensus labels, attention MIL | NaN during evaluation | Failed run |
| K3A | Fresh SE-ResNeXt50 compartment tiles at 1.8x, EMA, 256→320 resolution | Gold-58 AUC 0.842, correlation 0.81; weak blend | Rejected as too weak |
| K3B | Fresh EfficientNetV2-S compartment tiles at 1.8x, EMA, 256→320 resolution | NaN during evaluation | Failed run |
| K4A | Raptor-384 warm start, compartment tiles, new head, labels only, encoder LR 4e-5 | Gold-58 AUC 0.911; blend flat at 0.9230 | Development run; no submission |
| K4B | K4A with encoder LR 2e-4 | Collapsed; Gold-58 AUC 0.49 | Failed run |
| T1 / S1A | Teacher and label-v3 pipeline; first CoAtNet warm-start label-v3 arm | Teacher folds 0.886–0.901; teacher Gold-58 0.904; v3 Gold-58 0.921 vs consensus 0.896. S1A fp16 NaN | Label pipeline and failed student arm |
| S1B | ConvNeXt-S ImageNet initialized, global view, full data, teacher recipe | Gold-58 0.906; blend 0.9247 vs 0.9229 on the development comparison | Passed offline gate; F1 leaderboard submission still fell below F0 |
| F3 | F0 inference ensemble with 50% shrinkage of per-finding Raptor weights toward 0.5: lateral OA 0.20→0.35, PF OA 0.60→0.55, synovitis 0.80→0.65, Baker's 0.35→0.425; other weights and model inference unchanged | Visible preview: 3/3 studies, zero main-family failures, 547 s. Public score pending at last record | Submission 57451681, kernel version 1 |

F3 preview checks execution and output shape only; three visible examples cannot validate leaderboard quality or full hidden-test runtime. Kaggle's log recorded failed optional `pyjpegls` installation attempts, while the visible studies still completed. Refresh F3's status and score before treating this table as current.

## Decision

Keep F0 as the proven scored fallback. F1 is lower than F0. F3 is the only last-step variant submitted for direct public comparison; do not promote it until its score is returned. The offline Gold-58 metrics did not reliably predict F0 blend performance in this experiment sequence. No experiment here establishes private leaderboard performance.
