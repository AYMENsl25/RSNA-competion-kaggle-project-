# Source guide

Run commands from `V2_new_research/` with the project's Python environment. Official metadata, frozen splits, local caches, and verified pretrained weights are separate inputs; cloning the source does not download them.

| Files | Purpose and relationship |
| --- | --- |
| `m00_data.py`, `m00_model.py`, `p00_model.py` | Baseline series selection, ordered slice/window loading, masked pooling/loss, and verified ImageNet initialization. Six-slot selection groups each plane by fat suppression. |
| `s01_data.py`, `research/audit_s01_series.py`, `research/build_s01.py` | S01 coverage audit, multi-series loader, and standalone Kaggle bundle builder. Read the S01 result before reusing the experiment. |
| `canon.py` | Shared V3 physical 160-mm crop, canonical orientation, intensity scaling, and 256-pixel cache output. Laterality is reported here; mirroring happens in the shared model/input helpers. This publication uses the frozen N3a version, before N4 changes. |
| `relabel_v3.py` | Multilingual severity rules for selected report labels. Tests cover mild-only downgrades and preserve negative or unspecified mentions. |
| `v3_model.py` | Window preparation, normalization/mirroring, max/attention pooling, and the N3b sequence/position contextual head. Shared by training and inference. |
| `train_v3.py` | Memmap dataset, frozen split checks, hard/soft target masking, fp16 training, resume, validation, and checkpoint writing. Its module docstring gives example commands. |
| `run_n1.py`, `run_n3a.py`, `run_n3b.py` | Controlled experiment launchers: fixed parent settings, source identities, preflight gates, training, and paired comparisons. They need existing parent logs and local caches. |
| `check_n3b.py` | CPU masking, gradient, position sensitivity, and frozen N3a compatibility checks; requires the saved N3a source snapshot. |
| `compare_v3.py`, `research/audit_next_six.py` | Paired group-bootstrap comparisons and aggregate evidence audit. These use development predictions and fixed labels; they do not establish leaderboard performance. |
| `dashboard.py`, `progress.py` | Local monitoring of training logs. The dashboard also queries Kaggle submissions using locally configured credentials. |

Keep source snapshots embedded in already-run Kaggle bundles unchanged: they document the code actually used. Builders generate new bundles from their stated sources. Review generated files before uploading, because builder options can create private datasets or kernels.

CPU checks include `python -m pytest tests/test_m00.py tests/test_m01.py tests/test_p01a.py tests/test_p02.py tests/test_s01.py tests/test_n3a.py -q`, `python tests/test_relabel_v3.py`, and `python tests/test_canon.py`. The canonicalization check needs local pilot DICOMs. `tests/test_train_v3.py` additionally needs a pilot cache and CUDA; avoid it while another GPU experiment is running.
