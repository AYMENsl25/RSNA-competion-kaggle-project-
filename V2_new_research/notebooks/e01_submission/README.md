# E01 six-epoch offline submission (task I3)

`python build.py` writes `infer.py` (GPU kernel, this folder) and `preflight/infer.py` (identical script, CPU kernel) from `infer_template.py` plus the frozen `m00_data.py` / `m00_model.py` copies, which are hash-checked (same SHA-256 as E01 training and as recorded in the checkpoint).

- Checkpoint: private dataset `aymenslimani/rsna-knee-v2-e01-six-epoch-checkpoint`, `e01_best.pt`, SHA-256 `6b357912031969c267d670df1db304a0f554bb83ea31a6247ea1aef1dd062a60` (matches `research/E01_SIX_EPOCH_RESULTS.md`), epoch 6.
- Preprocessing = E01 validation exactly: `select_series`, IPP-ordered `load_series`, `build_study(train=False)` (16 windows/plane, per-series 1-99 percentile scaling to [0,1], no ImageNet norm). `model(x, mask)` batch 1 under fp16 autocast on GPU (never bf16), fp32 on CPU. 4 DataLoader workers.
- Failure policy, all counted in `e01_inference_summary.json`: series failure -> retry without that plane (`studies_degraded`); no usable plane -> column mean of successful test predictions (`studies_mean_fallback`, never 0.5); past 8 h -> same fallback (`studies_timed_out`).
- Train and visible-test DICOMs are all uncompressed Explicit VR Little Endian (`outputs/kaggle_dicom_audit`), so no extra codec wheels are attached; transfer syntaxes are logged.
- Kernel: T4 (`NvidiaTeslaT4`), internet off.

Runs (October 13, 2026): CPU preflight v1 passed (3/3 ok, 48 windows each, header/UID order equal to sample). GPU kernel `aymenslimani/rsna-knee-v2-e01-submission` v1 had a syntax typo and was superseded by v2 while queued (would fail at parse, no GPU work); v2 complete on cuda in 18.6 s, 3/3 ok, max |GPU fp16 - CPU fp32| about 3e-4. v2 submitted as "E01 six-epoch ResNet18 baseline". Outputs in `outputs/e01_submission_cpu_preflight` and `outputs/e01_submission_gpu_v2`.

Submission 57144276 (kernel v2) scored **public LB 0.769** (complete about 18 min after submit). Hidden-rerun logs and fallback counts are not visible to us; a failed hidden run would have errored instead of scoring.
