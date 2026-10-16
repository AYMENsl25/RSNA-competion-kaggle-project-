# E12 blend (Ens4 + Raptor maxspan + goodpjw reader)

`python notebooks/e12_blend/build.py` embeds `../v3_submission/gpu/infer.py` (must be the Ens4 build; OUT/STOP_AT_S redirected by env), `raptor_reader.py`, `gpjw_reader.py` into `gpu/infer.py` (T4, internet off) and `preflight/infer.py` (CPU). Weights at top of `blend_template.py`: raptor 0.35, gpjw 0.35, ens4 0.30. Each reader = own subprocess, writes per-study probs (NaN = failed); per-finding rank pct over the studies a reader predicted; weights renormalised per study over readers that succeeded; summary `e12_summary.json` (runtime, failures, studies_by_n_readers). Deadline: 8.5 h total, readers stop 10 min before; Raptor (last) cuts windows 62 -> >=16 after 20 studies if the projection overruns (logged in notes). fp16 autocast on CUDA only.

Submit (after the quota reset): `kaggle competitions submit rsna-knee-abnormality-detection -k aymenslimani/rsna-knee-v3-e12-blend-gpu -v 2 -f submission.csv -m "E12 blend ens4 .30 raptor .35 gpjw .35"`

## Provenance (checked Oct 15, 2026)
| dataset | licence | version (lastUpdated) | sha256 |
| --- | --- | --- | --- |
| dreaddevelopment/raptor-knee-maxspan | CC0-1.0 | 2026-08-22 20:14 | raptor_ft_coatnet_v5_full_swa.pt 20d45e0732a0e9120025d0172a4d2dd0170900bc250b06742dfb7ae08a63786a |
| goodpjw2008/rsna-knee-2-5d-convnext-reader | Apache-2.0 | 2026-10-05 00:09 | fold0 0b64baa9d1c29c8d5c4dffe06d9ab2c941119b37738e5a2346e882ab694e7b80, fold1 61696b6e67474695242fc8049e96e2ea13987db6e20d04410dc2961a689585cb, fold2 f5cf19cf591600c1c1a04f1aa5ff467751ef01513426a4ea3baa789443656cb6 |
| aymenslimani/rsna-knee-v3-ens4 | CC0-1.0 (ours) | 2026-10-15 11:33 | e45 d518bd33..., e6 965bbf0a..., e2 31b9a710..., e8 ad0ce4c9... (full in v3_submission/gpu/infer.py) |

Raptor inference adapted from public notebook dreaddevelopment/knee-mri-twelve-findings-from-a-single-model (widedense settings replaced by maxspan: IMG 336, span 2-98 %, 64 slices, 62 windows, res 384, as in rjings/rsna-versia-8). goodpjw: dataset's own knee.py/preprocess.py/infer.DicomStudyDataset, k=12, 3 folds mean, our batch-1 loop. Weights never downloaded locally.

## Runs
- preflight v1 (CPU, 3 visible): 3/3 all readers, 0 failures; ens4 57 s, gpjw 72 s, raptor 239 s (80 s/study CPU).
- gpu v1+v2 (T4, 3 visible): 3/3 all readers, 0 failures, 108 s / 128 s total. v2 loop time incl. warm-up: gpjw 18.8 s, raptor 21.9 s (62 windows) for 3 studies; ens4 inner 32 s incl. load.
- Hidden-test extrapolation (1,322 studies), upper bound from warm-up-inflated 3-study loops: gpjw 6.3 + raptor 7.3 + ens4 <=3.7 s/study -> <=6.4 h; expected 3-5 h. ~5-6.5 GPU-h incl. scoring rerun overhead. Guard keeps it under 9 h.
