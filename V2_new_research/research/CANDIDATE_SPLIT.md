# Candidate grouped split audit

This is a reproducible candidate artifact, not a frozen split or authorization to train. It uses only the official V2 `train.csv`, report normalization, and the current ignored `patient_groups.csv` census. Expert label cells are inspected only to identify fully labeled UIDs; their values are never exported, used to balance partitions, or used to select models.

The builder connects studies sharing an exact casefolded, whitespace-normalized report, census patient hash, or detected five-word-shingle Jaccard >=0.90 report edge. Entire connected groups touching an expert or census exclusion flag are excluded. A fixed SHA-256 group assignment with a 20% validation threshold gives 3,456 candidate training studies, 889 candidate validation studies, and 62 exclusions, preserving 58 experts and all four report-linked nonexpert UIDs. These counts are observed for the current source hashes, not a balancing target.

Run from the repository root:

```powershell
python V2_new_research/src/audit_near_reports.py --input V2_new_research/data/metadata/train.csv --output V2_new_research/research/near_report_audit.json --edges-output V2_new_research/outputs/candidate_split/near_report_edges.csv
python V2_new_research/src/research/build_candidate_split.py --train V2_new_research/data/metadata/train.csv --patients V2_new_research/outputs/kaggle_dicom_audit/patient_groups.csv --near-edges V2_new_research/outputs/candidate_split/near_report_edges.csv --output V2_new_research/outputs/candidate_split --summary V2_new_research/research/candidate_grouped_split_audit.json
```

The ignored `outputs/candidate_split/manifest.csv` contains UIDs, hashes, partitions, exclusion reasons, and `expert_labels_access_allowed=0`; it contains no report text or target values. Its companion audit and the aggregate tracked `candidate_grouped_split_audit.json` record source/manifest hashes and check that experts and all census exclusions are absent from development, and group/report/patient hash keys are disjoint across partitions. The audit explicitly prohibits training, tuning, checkpoint selection, and label-value inspection for each of the twelve expert targets. This is an auditable policy, not an operating-system access restriction; downstream loaders must enforce it.

The ignored near-report edge CSV contains only UID pairs and Jaccard similarities, with a source/edge hash sidecar. The builder requires this artifact, verifies both hashes and the 0.90 threshold, unions all 1,101 detected edges before assigning partitions, and checks that every edge stays in one partition. Against the reproducible exact-report/patient-hash candidate, 18 detected edges crossed training and validation; after reconciliation, zero edges cross any partition. There are 4,235 connected components, with largest size 37.

Two consecutive edge and split builds produced identical bytes. Edge SHA-256: `024b4e20fb67e3455d7e3d51d518b4b1b77b7e50a0531389686cb323e508e819`. Manifest SHA-256: `8769e66f39c5f67fd589bcaca99c4d9fab6dcb0613cd74fee276c963a9d7db83`. All integrity checks passed.

The detected near-report edges preserve the same four expert-linked nonexpert exclusions after transitive component propagation. The audit uses rare-shingle candidates plus forced exact pairs; it is not an exhaustive all-pairs near-report census. Templated reports may link unrelated patients, and transitive components can overgroup studies and distort validation composition. Sampled image duplicate auditing remains pending. De-identified patient IDs may be study-specific, so disjoint hashes do not prove patient independence. Fixed weak-label references and per-target observed positive/negative support checks remain pending. No split freeze, training, tuning, or expert performance evaluation was performed.
