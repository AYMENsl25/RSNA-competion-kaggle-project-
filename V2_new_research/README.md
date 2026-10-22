# Independent RSNA Knee MRI study

This study starts from the official competition CSVs in ignored `data/metadata/`, the official task definition and independently reviewed public references. Its results and experiments are generated here.

## Current position

The data and leakage audits, weak-label development split, and initial matched model experiments are complete. On the same 863-study report-proxy validation set, the best three-epoch model so far is P00 (pretrained ResNet18, macro-AUC 0.801735). M00 scored 0.739719, M01 scored 0.690701, and P01A scored 0.687420. These are development scores, not Kaggle leaderboard scores. See [measured results](research/P00_P01A_FULL_RESULTS_AND_NEXT.md) and the [next experiment plan](research/POST_BASELINE_EXPERIMENT_PLAN.md).

October 13 update: [E01](research/E01_SIX_EPOCH_RESULTS.md) reached 0.821538 macro-AUC after six epochs and is the artifact-verified development leader, with a documented final summary-status caveat. [P02](research/P02_RESULTS_AND_D01_NEXT.md) scored 0.712793 and did not improve on P00. The completed three-epoch [D01/S01 comparison and two next plans](research/D01_S01_RESULTS_AND_TWO_PLANS.md) shows 0.793934 and 0.792733 respectively, both below the matched three-epoch P00 score of 0.801735. No final competition submission has been made. The 58 expert-labeled studies remain protected from training and model selection.

The separate public competition stack experiments and final-week architectures are summarized in [final competition experiments](research/FINAL_COMPETITION_EXPERIMENTS_2026-10-22.md). F0 scored 0.958 public LB; F1 scored 0.954; F3's single shrinkage submission was pending at the latest recorded Kaggle status. These leaderboard results are distinct from this independent weak-report-label study.

## Layout

- `research/`: verified facts, source log, plan and decision records
- `src/`: small reproducible audits and, later, V2 modeling code
- `notebooks/`: future Kaggle audit, training and inference notebooks
- `experiments/`: proposed and measured experiment registry
- `data/metadata/`: ignored copies of official CSVs
- `outputs/`: ignored generated caches, checkpoints and predictions
