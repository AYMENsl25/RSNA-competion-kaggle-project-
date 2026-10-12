# M00 GPU pilot code brief for Claude CLI

Work only inside this V2 directory. Read `src/m00_model.py` and `research/RUN_GATES.md`; do not read V1, reports, expert labels or secrets. Write a private **synthetic-only** Kaggle GPU kernel in `notebooks/m00_gpu_pilot/` with `pilot.py`, a Kaggle `kernel-metadata.json`, and any helper module copied from V2 if needed. Do not push or run it yourself.

Use the M00 code exactly, no M01. The kernel has GPU enabled, internet disabled and competition source empty (no real data is needed). It must:

- Confirm CUDA GPU name, VRAM, torch/torchvision versions; fail clearly if CUDA is absent.
- Create deterministic random `[1,3,4,3,224,224]` tensors and a boolean mask with at least one missing role/padded window; use artificial binary targets/observed mask only for the backward pass.
- Run a few warmups then enough forward/backward optimizer steps to estimate seconds per study and peak GPU memory, using mixed precision only if supported. Cap its own script runtime well under 60 minutes, preferably <=10 minutes; no full training and no checkpoint.
- Verify finite outputs, loss and gradients; measure CPU wall time and `torch.cuda.max_memory_allocated()` / reserved. Write machine-readable `pilot_summary.json` to `/kaggle/working/` including measured iterations, device, model parameter count, batch/window shape, timings, peak memory, any failures, and explicit `engineering_only=true`.
- Avoid claiming total training/inference time from synthetic tensors. Do not import raw reports or any real labels. No competition submission.

Before ending, summarize what you wrote and any uncertainty about Kaggle GPU runtime. Do not execute the notebook or alter M00 model code.
