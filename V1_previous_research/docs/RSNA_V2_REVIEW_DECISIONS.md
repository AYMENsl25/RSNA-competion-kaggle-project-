# Independent review and implementation decisions

The user requested the LLM council skill. No council SKILL.md was installed in the searched skill/plugin directories. We disclosed an independent-agent fallback. One research/evaluation reviewer completed its audit. Preprocessing and training workers were interrupted by account usage limits; preprocessing source survived, training was completed by the main agent. This is not a completed formal council vote or a claim that the unavailable skill ran.

Accepted review findings:

- Keep original Kaggle mounts, attachment filenames, teacher version and grouped folds.
- Treat repeated historical gold evaluations as descriptive, and disable final audit until a recipe is frozen.
- Use a matched new v2 control; do not call changed preprocessing an exact v1.2 reproduction.
- Compare full-field and physical crops, normalization, slice coverage, channel representation and frozen encoder with one changed factor at a time.
- Measure teacher agreement with global per-target weighted BCE. Do not compare validation under changed weights.
- Preflight disk and pilot decoding, keep missing inputs masked, and use identical pixel transforms in submission.
- Preserve the v1.2 score discrepancy rather than silently choosing a historical summary number.

Deferred recommendations: fluid/T1 semantic slots, precomputed frozen features, focal patch pooling, uncertainty-weighted supervision, OOF stacking, ranking auxiliary loss, laterality mirroring and a diverse second family. Each requires a specific implementation and validation before being promoted from proposal to runnable experiment.

Source pointers: historical handoff/ledger and archived B2A/B2B1/inference notebooks under `docs/handoff/`; original v1.2 JSON and predictions under `artifacts/results/results/phase4b1b_first_mri_baseline_v1_1/`.
