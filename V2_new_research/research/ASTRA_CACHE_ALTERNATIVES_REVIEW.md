# Astra review of the CPU cache alternatives — 2026-10-10

Decision: **no-go for either full cache design or further GPU work; go for targeted CPU work after the supervision contract is resolved.** Review of the private Kaggle aggregate in `../outputs/m00_cache_alternatives/cache_alternatives_summary.json` and its notebook code.

- Resized float32 volumes retain dynamic window selection, but their projected training storage is 50.00 GiB against 15.5 GiB usable. Serialization itself is lossless; end-to-end preprocessing was not bitwise equal to direct DICOM processing (maximum difference 1.79e-7), plausibly because interpolation sees a different tensor layout. Treat it as near-equivalent, not exact parity.
- Two fixed uint16 draws preserve masks and have measured maximum absolute quantization error 7.63e-6, within the declared bound. Their projected training storage is 19.36 GiB and they remove dynamic draw diversity. Neither disk nor input-policy parity passes.
- Reported cache reads return metadata only. They omit tensor delivery, collation, pinned memory and GPU transfer, so cannot establish DataLoader throughput. Summed worker peak RSS near 14 GiB is an upper bound across non-simultaneous process peaks.
- The 100-study selection was previously checked against split v3, with no excluded or expert study, although this audit reconstructs an older exclusion rule. Preserve that check in any new probe.

Next priority: blinded multilingual clinical report review and a frozen independent binary reference. Determine how many training studies have actual observed supervision before sizing another cache. If needed afterward, test a bounded partial cache with dynamic windows and on-demand fallback; measure end-to-end CPU tensor delivery and retain the disk ceiling. One fixed uint16 draw is a separate input-policy proposal, not M00 as approved. Neither current audit supplies validation AUC or clears the eight-hour training gate.
