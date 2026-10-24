# F6: Gaussian-rank ensemble, fixed before execution

## Submission confirmed

Submitted **version 1**, reference **57494546**, October 21 at 22:09:14 Istanbul
time (19:09:14 UTC). Live CLI confirms **PENDING**, no score. F5 is also pending.
Three daily submissions remain. Quota is **3.56 GPU hours remaining**, versus
3.72 before this preview (rounded displayed difference 0.16 hours). No final
selection changes were made.

Three numerical tests passed. Every Python cell, including generated worker
scripts, parsed; unchanged inference cells were compared with F0 and hashed.
Preview completed in 566.55 pipeline seconds, with all main families and reader.
Independent main-blend reconstruction differed by at most 5.09e-8, and final
reader-blend reconstruction matched exactly. Schema, UID uniqueness/alignment,
finite/range values and checksum passed. Submission SHA-256:
`5e9ba665f3e30642e18dbb555eb7cbd16ab8a659ffb5f7226c72aaac44c9df1b`.
The three-row final preview matches F3's earlier preview checksum; this does not
establish equality on the hidden population or predict any leaderboard score.

New files: `scripts/build_f6_gaussian_rank.py`, `scripts/verify_f6_preview.py`,
`tests/test_f6_gaussian_rank.py`, `notebooks/f6_gaussian_rank/` containing notebook,
metadata and source-hash manifest, this report, and execution/prediction receipts
under `outputs/f6_preview_v1/`. Baseline sources/artifacts were not edited.

## Original fixed design

The user requested another speculative submission using existing weights. F5 is
still PENDING; F0 remains the best known public score at 0.958. Quota was checked:
3.72 GPU hours remaining. One preview capped at 1,800 seconds is authorized,
followed by one version-pinned scoring submission after execution verification.

Transform each main model's per-finding values into normal scores:
`z = ndtri((average_rank - 0.5) / n)`. Combine Raptor384/Raptor224 with their
original 0.65/0.35 coefficients, then Raptor/R6 with F0's original per-finding
coefficients. Apply `ndtr` to the weighted sum for bounded output. Retain F0's
small-reader stage, coefficients and final ranks unchanged. This is a controlled
change to the main aggregation rule relative to F0, not a new trained model.

The nonlinear transform spreads extreme ranks more than central ranks. Hypothesis:
extreme specialist predictions may improve some combined orderings. Failure mode:
extreme errors can receive too much influence. Rank extremity is not calibrated
confidence. There is no independent validation evidence of a gain, no coefficient
sweep, and no transfer-learning claim. Repeated public leaderboard experiments
introduce selection bias.

Math reference: [SciPy ndtri](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.ndtri.html).
This documents the transform, not its effectiveness in this competition.
Midranks avoid infinities and handle ties/singleton inputs. Tests extract and run
the exact generated notebook function. All main predictions and the reader must
complete; failure aborts the experiment. Model source cells, offline inputs,
checkpoints, TTA and container are preserved. No expert labels or final test
labels are accessed. No new training is started.

Preview verification checks schema, UID alignment, finite/range values, checksum,
and independent reconstruction of both Gaussian aggregation and final reader
blend. Three visible studies cannot establish AUC or a full hidden-runtime bound.
For three untied ranks, normal scores are equally spaced, so the visible final
ordering may match F0 even though larger test sets differ. Hidden-scoring quota
and timeout semantics remain unverified; the successful F3 capped-run precedent
is not a formal bound. Preserve F0 and final selections.
