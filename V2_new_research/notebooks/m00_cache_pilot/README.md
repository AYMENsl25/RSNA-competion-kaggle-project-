# M00 cache feasibility pilot (CPU-only)

Private, internet-off Kaggle script on the official competition data. Follows the integrated GPU
pilot whose real DICOM loading failed the eight-hour full-run projection.

- Same 100 deterministic nonexpert studies as `m00_loader_throughput` (expert rows and exact
  expert-report matches excluded).
- Per study: decode the selected series once, build two fixed seeded training draws
  (`[3,4,3,224,224]` float32 + mask `[3,4]`), save with `np.savez_compressed` (lossless).
  Files are named by `sha256(uid)`, never raw UIDs.
- 48-window inference arrays for 10 studies, sized separately.
- Measures: four-worker build wall time, compressed bytes/study (median/p95/max, plus projection to
  3,457 train and 885 validation studies), four-worker compressed-read vs DICOM on-demand
  throughput, per-study CPU and worker peak RSS, bitwise equality + aggregate SHA-256 of cached vs
  freshly recomputed arrays (10 studies, same seeds).
- Failures are counted by exception type (type names only); no silent replacement.
- Bounds: 20-minute script (phase budgets cancel unstarted work), cache <5 GiB guard, cache in
  `/kaggle/working/cache_pilot_scratch` removed in `finally`. Output: one aggregate JSON
  (`cache_pilot_summary.json`) with no UIDs, reports, images, arrays or cache files.

## Caveats

- **Policy change:** two fixed draws per study alter epoch-wise random-window sampling. This is a
  pretraining input-policy choice requiring review; it is NOT exactly equivalent to dynamic M00.
- Estimates (full build time, eight-epoch read time, both with 1.5x headroom) are provisional.
  Cached reads are likely page-cache-warm (optimistic). Worker RSS may include inherited parent RSS
  from `fork`. No GPU runtime can be estimated from CPU-only numbers.
- The inference build re-decodes DICOM separately from the train build (conservative for build time).

## Build

`python src/research/build_standalone_notebooks.py` embeds the current `src/m00_data.py` into
`cache_pilot.py` (the `code_file`). Not executed or pushed yet; root reviews and runs.
