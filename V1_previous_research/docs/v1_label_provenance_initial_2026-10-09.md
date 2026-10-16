# Initial V1 label-provenance check — 2026-10-09

This is a source review, not an audit of every saved checkpoint's exact training membership.

The V1 `notebooks/fresh/01_Preparation_Labels_and_Audit.ipynb` preparation path builds 12 `y_*` values and 12 `w_*` masks by regular expressions over the `Report` column. It recognizes local negation and uncertainty, gives positive/negative mentions weight 1, and masks unmentioned/conflicting mentions. It writes `fresh_manifest.csv`, `label_audit.csv` and `fresh_preparation.json`. The shown `label_report` path does not read the competition's expert-label target columns as its label source. This does **not** establish that no downstream training or model selection accessed those columns.

The notebook groups folds by a patient identifier if available; otherwise it hashes normalized report text. The official train CSV has no patient identifier, so this fallback groups identical reports but cannot prove patient independence or detect different reports from the same patient. The source itself warns of this limitation.

The V1 `04_Full_Cohort_Readiness.zip` manifest includes all 58 expert-study UIDs across its five prepared folds. Prepared-fold membership is not equivalent to training exposure. Before evaluating any checkpoint against expert labels, extract its exact fold, training UID set, data source, label source and selection history. The prior [V1 research decision](../../V1_previous_research/docs/RSNA_RESEARCH_DECISION_2026-10-09.md) reports only 503/696 numerical `y_*` agreements with the 58 expert labels and many zero weights, consistent with imperfect report-derived supervision; this count must be reproduced if used as a V2 quantitative claim.

**Next provenance step:** For each candidate V1 artifact, tie checkpoint hash to its manifest, fold and training script. Then quantify report-label agreement and supervised coverage by target on the 58 cases, documenting that this set was already visible in V1 project development.
