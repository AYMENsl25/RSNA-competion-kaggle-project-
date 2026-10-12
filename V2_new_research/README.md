# Independent RSNA Knee MRI study

This study starts from the official competition CSVs in ignored `data/metadata/`, the official task definition and independently reviewed public references. Its results and experiments are generated here.

## Current position

The data and leakage audits, weak-label development split, and initial matched model experiments are complete. On the same 863-study report-proxy validation set, the best three-epoch model so far is P00 (pretrained ResNet18, macro-AUC 0.801735). M00 scored 0.739719, M01 scored 0.690701, and P01A scored 0.687420. These are development scores, not Kaggle leaderboard scores. See [measured results](research/P00_P01A_FULL_RESULTS_AND_NEXT.md) and the [next experiment plan](research/POST_BASELINE_EXPERIMENT_PLAN.md).

E01 (six-epoch P00) and P02 (finding-specific per-window logits) have been launched on private Kaggle kernels. Their final results are pending in this snapshot. No final competition submission has been made. The 58 expert-labeled studies remain protected from training and model selection.

## Layout

- `research/`: verified facts, source log, plan and decision records
- `src/`: small reproducible audits and, later, V2 modeling code
- `notebooks/`: future Kaggle audit, training and inference notebooks
- `experiments/`: proposed and measured experiment registry
- `data/metadata/`: ignored copies of official CSVs
- `outputs/`: ignored generated caches, checkpoints and predictions
