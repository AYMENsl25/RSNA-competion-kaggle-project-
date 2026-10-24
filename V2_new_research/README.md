# Independent RSNA Knee MRI study

This study starts from the official competition CSVs in ignored `data/metadata/`, the official task definition and independently reviewed public references. Its results and experiments are generated here.

## Final status (competition closed 2026-10-22)

- Best own model: **Ens4**, 0.857 public / 0.837 private ([deep dive](../docs/OUR_BEST_MODEL.md)).
- Public-stack fork F0: 0.958 / 0.942. The F1-F7 variants did not beat it ([final experiments](research/FINAL_COMPETITION_EXPERIMENTS_2026-10-22.md), [all scores](../docs/RESULTS.md)).
- Later own-model runs: N3b (transformer head) was worse; N4 (native 384) collapsed early ([N4 gate](research/N4_EARLY_GATE_AND_N5.md)); the N5 differential LR lost; and [N6](research/N6_RESULTS.md) showed that the N1 attention gain did not replicate on a second seed.
- Postmortem: [21st/22nd place comparison](research/POSTMORTEM_21ST_22ND_2026-10-24.md) and [lessons from the top solutions](../docs/LESSONS_FROM_TOP_SOLUTIONS.md).

The sections below are the historical status notes, kept as written.

## Earlier position

The data and leakage audits, weak-label development split, and initial matched model experiments are complete. On the same 863-study report-proxy validation set, the best three-epoch model so far is P00 (pretrained ResNet18, macro-AUC 0.801735). M00 scored 0.739719, M01 scored 0.690701, and P01A scored 0.687420. These are development scores, not Kaggle leaderboard scores. See [measured results](research/P00_P01A_FULL_RESULTS_AND_NEXT.md) and the [next experiment plan](research/POST_BASELINE_EXPERIMENT_PLAN.md).

The completed baseline follow-ups are recorded in [E01](research/E01_SIX_EPOCH_RESULTS.md), [P02](research/P02_RESULTS_AND_D01_NEXT.md), and the [D01/S01 comparison](research/D01_S01_RESULTS_AND_TWO_PLANS.md). E01 reached 0.821538 on the earlier 863-study development set; its first competition submission scored 0.769. Later V3 experiments use a separately audited 885-study / 843-group development set, so these development scores are not direct matched comparisons.

October 16 snapshot: [N1](research/N1_RESULTS_AND_N3A_DECISION.md) completed 12 epochs, with final public-label development AUC 0.8867645. [N3a](research/N3A_RESEARCH_REFRESH_2026-10-16.md) reached 0.8884626; its predeclared blend reached 0.8918068. These are weak-label development scores. The saved October 15 evidence records Ens4's public leaderboard score as 0.857. [N3b](research/N3B_RUN_AND_LEADERBOARD_GAP_2026-10-16.md) is published as launched code and a protocol; its outcome is recorded in the final status above.

The [V3 progress record](research/V3_NINE_DAY_PLAN.md#progress-log) documents one authorized diagnostic evaluation of E01 on the 58 expert studies. Those studies stay excluded from training and model selection. Historical plans and status notes retain the dates and assumptions under which they were written.

## Layout

- `research/`: verified facts, source log, plan and decision records; `external_evidence/` holds pulled public notebooks and research reports
- `scripts/`: F3-F7 submission builders and preview checks
- `tests/`: unit tests for data, models and submission builders
- [`src/`](src/README.md): audits, shared preprocessing/models, local training, paired evaluation, and monitoring
- [`notebooks/`](notebooks/README.md): Kaggle cache, training, preflight, offline inference, and ensemble bundles
- `experiments/`: proposed and measured experiment registry, plus [completed-run histories and paired evidence](experiments/completed_runs/README.md)
- `data/metadata/`: ignored copies of official CSVs
- `outputs/`: ignored generated caches, checkpoints and predictions

See the [publication record](research/PUBLICATION_2026-10-16.md) for verification and local asset requirements. Source files are research entry points: read their arguments and output paths before running them.
