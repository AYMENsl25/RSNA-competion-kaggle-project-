# M00 implementation brief for Claude CLI

Work only inside this `V2_new_research` directory. This is an independent RSNA 2026 Knee Abnormality Detection study. Read `research/PLAN.md` and `research/RUN_GATES.md` first. Do not inspect or use `V1_previous_research` or other previous-study outputs. Do not read or send raw radiology reports, expert labels, secrets or credentials. No Kaggle submission, network call, GPU run or model training.

Implement the **M00** model and a small synthetic CPU correctness smoke, with concise documentation, under `src/` and `tests/` as needed. This is code preparation; full training remains gated.

Required architecture and data contract:

- PyTorch ResNet18 initialized randomly, 224x224 input. A window is three *physically adjacent* grayscale slices of one DICOM series as channels. Do not treat RGB or filename order as anatomy.
- A study has sagittal, coronal and axial role slots; deterministic series selection must use the official `train_series.csv` fields and handle multiple or missing series. No test-time report input.
- M00 takes a fixed tensor shaped `[batch, 3 roles, windows per role, 3 slices, 224, 224]` and a matching boolean valid-window mask. Training uses four windows per role, inference sixteen. Pool **shared max** over valid window features, then one linear 12-target head. A study with all roles absent must fail clearly; absent roles/windows must never affect max. Report any design ambiguity instead of silently inventing clinical semantics.
- DICOM slice order uses ImageOrientationPatient row/column cross product and ImagePositionPatient projection, with validated fallback only when tags are missing; SOP filename order is unsafe. Apply rescale slope/intercept if present, then robust per-series scaling; no flips. Handle MONOCHROME1, invalid/constant slices and nonfinite values explicitly.
- A simple masked binary cross-entropy utility may be included, but do not invent weak labels or a split. Never use the 58 expert studies for training, tuning or repeated selection.
- Offline Kaggle inference readiness matters: no runtime internet or downloading weights, clear deterministic failures, outputs finite 12 probabilities per study. Do not create a submission or push a Kaggle notebook yet.

For this request, write M00 implementation code and meaningful synthetic tests only. Run the CPU tests if dependencies are present; otherwise give exact commands and record missing packages. At the end, summarize files changed, risks, and what blocks a GPU pilot. Do not implement M01 yet.
