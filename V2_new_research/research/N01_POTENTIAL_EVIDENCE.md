# Potential evidence inventory — triage only

`src/research/n01_evidence_inventory.py` inventories eligible official reports using the current candidate manifest. It never consults expert target values and skips excluded studies before report access. No reports leave the local CPU process. Strict N01 assertion rules and candidate labels are unchanged.

Each count is the number of reports with a lexical target mention or at least one named cue within 120 characters before/after a mention. Cues include severity, low severity, negation, uncertainty, temporal/surgical language and numeric dimensions. They can concern a different finding, overlap, or occur in history. For example, “cannot exclude” contributes both negation and uncertainty cues; it is not an observed negative. No polarity, binary labels, masks, or verified target support are produced.

The multilingual stems are exploratory and require native-language review. Missed inflections, aliases, spelling, accents and abbreviations mean this is not a complete evidence census. Counts provide a potential upper bound only among detected lexical matches. A target mention, particularly an OA compartment, may contain no qualifying pathology. Ligament abbreviations may be ambiguous. Language assignments retain the existing deterministic confidence >=0.90 heuristic and can be wrong.

The tracked `n01_potential_evidence_inventory.json` contains aggregate counts by partition, heuristic language and target, report denominators, limitations and provenance hashes. The ignored `outputs/n01/potential_evidence_inventory.csv` contains UID/partition/language/target and cue flags, without report text. It is a reviewer routing aid, not training supervision. Reviewers should use the official report locally and apply the host rubric independently.

```powershell
$env:PYTHONPATH = (Resolve-Path V2_new_research/.tools).Path
python -m unittest discover -s V2_new_research/src/research -p test_n01_evidence_inventory.py
python V2_new_research/src/research/n01_evidence_inventory.py --train V2_new_research/data/metadata/train.csv --manifest V2_new_research/outputs/candidate_split/manifest.csv --private V2_new_research/outputs/n01/potential_evidence_inventory.csv --summary V2_new_research/research/n01_potential_evidence_inventory.json
```

Four synthetic tests cover overlapping negation/uncertainty, explicit negative cue without labels, Spanish negation, nearby severity/uncertainty, distant-cue exclusion, silence and decimal dimensions.
