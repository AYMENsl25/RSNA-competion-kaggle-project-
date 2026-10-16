# Completed-run evidence

These immutable copies preserve aggregate training/validation histories for E1, E2, E3, E4+E5, E6, E7, E8, N1, and N3a. Each run had all configured epochs present before copying. N1 and N3a additionally include completed status records and their saved paired comparison summaries. `sha256.json` records the SHA-256 of each copied file.

`log.json` contains the historical argument set, cohort counts, per-epoch losses/timings, validation metrics, and selected best epoch. Paths in those arguments refer to the original local research workspace. These paths document the run; the corresponding private assets are not bundled here.

Comparison files report paired development-set estimates and confidence intervals. They are not leaderboard results. The histories retain each experiment's own settings and should not be treated as isolated single-factor ablations without checking the relevant research note.

No checkpoints, raw images, report text, per-study prediction arrays, expert-label tables, or N4 outputs are included. N3b's launch protocol is published separately; this evidence directory does not infer its completion.
