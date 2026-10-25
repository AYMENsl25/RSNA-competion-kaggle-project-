# RSNA Knee MRI technical report

**Budgeted Model Selection for Weakly Supervised Knee MRI**  
Version 1.0 - 25 October 2026 - Project: AYMENsl25/RSNA-competion-kaggle-project-

An empirical account of our preprocessing, models, controlled experiments,
negative results and final competition outcomes. The report distinguishes the
models we trained from ImageNet initialization from the public-checkpoint stack
we reproduced. It is a technical report, not a peer-reviewed clinical study.

- [Read the report on GitHub](REPORT.md)
- [Download the PDF](rsna-knee-technical-report.pdf)
- [Aggregate results](data/aggregate_results.json)
- [Evidence and source hashes](data/evidence_manifest.json)
- [Build and check the PDF](BUILD.md)

## Main findings

Our strongest independently trained scored ensemble, Ens4, achieved **0.857
public / 0.837 private AUC**. Reproduction of the upstream public F0 stack
achieved **0.958 / 0.942**, with no demonstrated private improvement from our
late aggregation variants. Development experiments found conditional benefits
of pretraining, context and ranking, but the initial attention-head improvement
did not replicate in a second matched pair. Proxy-label scores and competition
scores are presented as different evaluation endpoints.

Only aggregate measurements, documentation, original schematic figures and
report-building code are included here. Readers need separately obtained data
and model artifacts to reproduce training; building this report does not run
training or access those artifacts.
