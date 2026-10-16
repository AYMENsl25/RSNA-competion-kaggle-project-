# Kaggle bundle guide

Each kernel directory contains a script and `kernel-metadata.json`; builders embed the required source files so Kaggle can run offline. Templates are readable starting points, while generated scripts preserve the actual submitted bundle. Run builders from the working directory described in their docstrings or directory README.

| Directories | Role |
| --- | --- |
| `m00_*`, `m01_*`, `p00_*`, `p01a_*`, `p02_*`, `d01_*`, `e01_p00_six_epoch/` | Earlier audit, cache, CPU/GPU preflight, and controlled baseline training bundles. |
| `s01_cpu_check/`, `s01_gpu_pilot/`, `s01_full_train/` | Multi-series experiment gates and full three-epoch training. |
| `cache_v3/` | Canonical 256-pixel uint8 cache builder, pilot parity checks, five shard kernels, and resumable download helpers. Cache arrays stay outside Git. |
| `e01_submission/` | First baseline offline inference, source/checkpoint hashes, CPU preflight, and recorded public score. |
| `v3_submission/` | Shared V3 inference builder, CPU raw/cache parity comparison, preflight, and GPU bundles. Checkpoints are supplied through private datasets. |
| `e12_blend/` | Ens4/Raptor/goodpjw blend, reader provenance, failure handling, runtime budget, and CPU/GPU checks. The README records licenses and weight hashes. |

The local N1/N3a/N3b training entry points are in `../src/`. Their checkpoints and per-study predictions are local assets. Historical metadata can name a previously run kernel; cloning this repository does not launch or resubmit it.

N4 bundles are intentionally absent from this publication while that work is in progress.
