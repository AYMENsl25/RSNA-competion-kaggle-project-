# M00 cache alternatives audit (CPU-only)

Private, internet-off Kaggle script on the official competition data. Follows `m00_cache_pilot`,
whose two-fixed-draw float32 cache projected to ~36 GiB (does not fit 19.5 GiB free).

Same 100 deterministic nonexpert studies as the loader/cache pilots. Two designs, each built from
the same DICOM input (decoded separately per design; conservative for build time):

1. **`float32_volume` (lossless, dynamic sampling preserved):** after `load_series`, each usable slice
   is resized once to 224x224 (torch bilinear, `align_corners=False`); full per-role resized volumes
   + `slice_ok` go to `np.savez_compressed`. Windows are drawn at read time with `window_centers`, so
   epoch-wise random sampling is unchanged. Parity: 10 studies, two seeded 12-window train draws +
   48-window inference vs direct `build_study`, bitwise; if not bitwise, max abs error is reported
   (slice-wise vs 3-channel interpolation ordering is the possible source).
2. **`uint16_two_draws` (LOSSY, policy change):** two fixed seeded training draws via `build_study`,
   quantized `round(x*65535)` to uint16, stored with masks. Reports actual compressed size and, for 10
   studies, max/mean/p99 abs error of `q/65535` vs original float32 (valid windows only; p99 from a
   histogram, resolution 5e-9) and exact mask equality. Gate: max error <= 1/(2*65535)+1e-6.
   Never called exact parity; fixes two windows per study, i.e. NOT dynamic sampling.

Per design: 4-worker build wall time, bytes/study (median/mean/p95/max), projection to 3,457 train
studies vs 19.5 GiB free minus 4 GiB reserve, aggregate worker RAM (sum of per-process peaks, an upper
bound), 100-study read + window-construction throughput, failures by exception type, cleanup.
Validation stays on-demand (overlapping 48 windows not cached). Estimates: full train cache build and
8-epoch CPU read time (1.5x headroom, provisional). No GPU estimate.

## Bounds and caveats
- Scratch `/kaggle/working/cache_alternatives_scratch`; design 1 files are deleted before design 2
  builds; each design guarded at 5 GiB (a few in-flight files can overshoot slightly; <6 GiB intended);
  everything deleted in `finally`.
- Time limits are **cooperative**: phase budgets (~1100 s total) cancel queued work only. Running
  workers are not killed, so the 20-minute target is not a hard wall cap.
- Reads: page cache is dropped best-effort (`posix_fadvise`) but may remain warm, so read numbers
  are optimistic. Forked worker RSS may include inherited parent RSS.
- Output: one aggregate JSON (`cache_alternatives_summary.json`); no UIDs, reports, images, arrays.

## Build
`python src/research/build_standalone_notebooks.py` embeds `src/m00_data.py` into
`cache_alternatives.py` (the `code_file`). Not executed on Kaggle or pushed; root reviews and runs.
