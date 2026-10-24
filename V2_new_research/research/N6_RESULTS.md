# N6 completed results — Kaggle seed1 paired replication

Completed status verified from Kaggle and downloaded small artifacts. Both arms finished all12 epochs on two TeslaT4 GPUs in3983.257seconds (66.39minutes). No budget truncation. Each used3428 training studies and885 validation studies /843 patient groups. All24 epoch prediction files are present. Both genuine-cache CUDA preflights passed. Fixed public-label comparison recomputed locally and agrees with Kaggle within1e-12. Checkpoints remain in Kaggle outputs; this retrieval downloaded metrics, predictions and logs only.

## Fixed epoch12 results

| Arm | Public weak-label macro AUC | v3 macro AUC |
| --- | --- | --- |
| Max pooling, seed1 |0.8845740940|0.8906332161|
| Finding attention, seed1 |0.8825701863|0.8889567436|

Attention minus max: public -0.0020039077; paired group-bootstrap95%CI[-0.0068233979,+0.0029055212]. v3 difference -0.0016764724; CI[-0.0066544784,+0.0029664278]. Neither supports attention superiority in this repeat. The negative point estimate is not statistically established harm. Lateral-meniscus point difference -0.0157474; fracture -0.0156332; do not infer target-specific certainty without intervals/support.

Seed0 historical N1 versus E8: +0.0123156044, CI[+0.0070499941,+0.0182671165]. Thus the attention benefit did not replicate in the matched seed1 pair. Seed0 was on the laptop and seed1 on Kaggle: both random seed and runtime/hardware environment changed. The head comparison within seed1 is hardware-matched; cross-run variability cannot be attributed solely to seed. No universal claim that max pooling is better follows.

## Predeclared ensemble addition

Existing accepted N3a base:0.8918068136. Fixed80% base +20% seed1 attention ranks:0.8926285551. Gain+0.0008217415; CI[-0.0000947692,+0.0017522156]. No-Synovitis gain+0.0009511933; CI[-0.0000425550,+0.0018726793]. Gain is below the >=0.002 promotion requirement and CI crosses zero. Do not promote this blend or tune its weights against the same holdout.

These are fixed weak-label validation scores, not Kaggle public leaderboard submissions. N6 generated no competition submission, and the expert holdout was not evaluated.

Best-v3 checkpoint epochs were max9 and attention10, recorded but not substituted into the predeclared final12 comparison. Across20568 training steps per arm, GradScaler skipped10(max)/9(attention) updates within trainer tolerance. PyTorch emitted the initial scheduler-before-optimizer warning when AMP skipped the initial optimizer step; the existing matched per-batch schedule was preserved. No nonfinite epoch failure.

Decision: replication/promotion gate not passed. Keep the accepted incumbent ensemble and retain the N6 predictions and Kaggle checkpoints as research artifacts. Next useful completion is the paused N5 differential-rate arm, compared at epoch16 against its completed uniform16 control (public0.8896348502). N5 is paused for laptop shutdown, with epoch1 checkpoint verified and src/resume_n5.py ready; do not automatically resume until requested. Scheduled15-minute checks remain paused.
