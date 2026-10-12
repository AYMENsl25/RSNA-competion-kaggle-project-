# M00 integrated GPU pilot (engineering only)

Private, internet-off, GPU script kernel on the official competition data. Not a model experiment.

- `pilot_template.py` is the source; `pilot.py` is generated (`python src/research/build_standalone_notebooks.py`) with `m00_data.py` and `m00_model.py` embedded. Kaggle uploads only `pilot.py`.
- 100 deterministic studies (SHA256 order) with no complete expert labels and no exact report match to expert studies: 70 optimizer-timing cases (bs 2, AMP), 20 no-grad validation-timing cases, 10 inference-timing cases (48 windows). These are timing partitions only; they are not the frozen development train/validation split. Four DataLoader workers (spawn, safe after CUDA initialization), seed 0.
- Targets/observed masks are seeded synthetic noise from a UID hash; real label values are never used. Weights and predictions are discarded; no AUC, no `submission.csv`. A temp checkpoint is timed then deleted.
- Budget: per-phase caps 900/420/420 s, hard 2400 s total (<= 45 min, < 1 GPU-hour).
- Output: `integrated_pilot_summary.json`, aggregate only (timings, studies/hour, step median/p95, peak VRAM, failures by exception type, valid-window counts, versions, provisional estimates for 3,457 training and 885 validation studies per epoch over eight epochs plus approximately 1,300 test studies, with 1.5x headroom and listed assumptions).
- Any failure is counted by exception type, never replaced. `completed` is true only if >=30 steps, 20/20 val, >=10 infer studies and zero failures.
