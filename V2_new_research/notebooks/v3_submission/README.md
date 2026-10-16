# V3 offline submission

`python notebooks/v3_submission/build.py <run>/best.pt --dataset <slug> [--upload]` bundles src/{m00_data,m00_model,canon,v3_model}.py (SHA-checked) into `gpu/` (T4, internet off, fp16 autocast, never bf16), `preflight/` (CPU, visible test) and `parity/` (CPU, 20 pilot train studies). Config (head, eval windows, res, mirror) comes from `<run>/log.json`. `--upload` creates/versions the private checkpoint dataset. Fallbacks as e01_submission (degraded plane drop, column-mean fallback, 8 h stop), summary in `v3_inference_summary.json`.

Known gap: if the selected series fails, inference drops the plane; training would have picked the next cached series.

E1 (Oct 13, 2026): dataset `aymenslimani/rsna-knee-v3-e1-checkpoint` (best.pt sha 46d66bf9...). Parity kernel v1: 20/20 ok, max |kernel - train_v3 cache eval| = 3.3e-7 (`parity_compare.py`). CPU preflight v1: 3/3 ok, 48 windows each. GPU kernel not pushed.

Ensemble (Oct 15, 2026): `build.py e45=outputs/e45_v3/best.pt e6=outputs/e6_v3/ep16.pt e2=outputs/e2_v3/best.pt --dataset rsna-knee-v3-ens3 --kernel rsna-knee-v3-ens3` (then gpu kernel id renamed to `rsna-knee-v3-ens3-gpu`: a kernel slug equal to the dataset slug gives 409 on push). Each study canonicalised once, windows filled per (mirror, windows), all models run; per-model `probs_<name>.csv`; submission = per-finding rank pct over test studies, mean over models that succeeded (failures counted in `model_failures`). Parity: max diff e45 2.4e-7, e6 1.8e-7, e2 2.4e-7 (20/20). `parity_compare.py <ckpt> probs_<name>.csv`.
Ens4 (Oct 15, 2026): `build.py e45=... e6=... e2=... e8=outputs/e8_cnxt/best.pt --dataset rsna-knee-v3-ens4 --kernel rsna-knee-v3-ens4` then set gpu id to `-ens4-gpu` (keep dataset_sources = dataset). timm is in the Kaggle image offline. Parity max diff all <3e-7 incl. convnext e8. LB 0.857.
