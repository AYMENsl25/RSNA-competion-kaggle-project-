# Experiment report template

Use one copy for each completed GPU pilot or full experiment. A preparation or running job is not a result.

- ID, hypothesis, status, timestamp, code revision, data/label/split hashes:
- Changed factor; matched control and all fixed settings:
- GPU model, number of GPUs, measured GPU-hours, wall time, peak VRAM:
- Studies used for fitting/validation; expert/linked exclusions and failed studies:
- Validation reference provenance, observed positive/negative count per target:
- Validation macro-AUC with uncertainty; targets excluded for insufficient support:
- Per-target AUC (ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA, PF OA, Effusion, Synovitis, Baker's, Contusion, Fracture):
- Paired difference against previous baseline, uncertainty and per-target changes:
- Inference seconds/study p50/p95, projected full-test runtime and notebook result:
- Checkpoint and prediction artifact paths/hashes:
- Decision: advance, repeat, reject or investigate; reason and remaining risk:

The protected 58-study expert check is performed once after final candidate freeze, reported separately, and never used to choose between M00 and M01.
