# N01 initial candidate assertion reader

Status: **candidate, unvalidated; no evaluation reference frozen; no training authorized**.

`src/research/n01_assertions.py` implements four report states for the 12 targets. Only positive and negative cells have observation mask 1. Silence and unsupported languages are `not-addressed` with mask 0; uncertainty, conflicting assertions, historical/surgical mentions and ambiguous joint anatomy are masked. The private CSV includes a language-supported flag so unsupported language is distinguishable from actual English silence. States describe text assertions, not image truth.

Only English is supported. Language identity uses the existing deterministic langdetect heuristic with confidence >=0.90, which needs blinded review. Spanish, Turkish, Croatian, Greek, German, Bulgarian, Dutch, French and undetermined reports are entirely masked. No translation is attempted.

The official [host rubric](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733343) was independently retrieved through indexed web search on October 9 after direct retrieval returned no content. The revised reader requires explicit high severity for ACL; acute high severity for MCL; moderate/large effusion or cyst; acute fracture line/cortical break; and impact-related contusion with explicit absence of fracture. OA positives require compartment-specific high-grade/full-thickness cartilage loss plus explicit moderate/large area or a loss/defect dimension >=10 mm. Explicit normal/preserved cartilage or absence of cartilage loss can give OA negatives. Unqualified OA, grade numbers without a named system, and unspecified lesion dimensions remain uncertain. These are lexical proxies for image criteria, not verified MRI truth.

Rules require explicit anatomy and severity evidence for ligament positives; definite meniscal tear language remains a provisional proxy for the host surface/morphology criteria. Explicit normal/intact anatomy can give a negative. Degeneration alone stays uncertain. Bone contusion requires an explicit bone/osseous/marrow contusion or bruise phrase plus impact context and no fracture; marrow edema alone is unaddressed. Clause-local negation and uncertainty apply. Commas, sentence boundaries, newlines and contrast conjunctions split clauses. Decimal dimensions are preserved. Current Findings/Impression headings can terminate an explicit preceding historical section. Mild/small and historical findings are conservatively masked rather than assigned a soft or inferred negative.

Known limitations: this is a shallow rules baseline, not a clinical NLP parser. Clause boundaries can lose coordinated negation or shared anatomy; mentions across clauses are missed; narrative historical sections without recognised headings may remain; English misclassification can cause errors; isolated synovitis nouns may be headings; limited uncertainty synonyms and target aliases miss valid text; contradiction aggregation masks even a clearly resolved impression; normal anatomy can fail to establish the exact host threshold. Definite meniscal tear wording does not independently confirm contact on two images. Numeric OA dimensions may describe length rather than an actual lesion area. These are conservative candidate rules without a measured precision estimate. Blinded manual review is mandatory before freezing any label file. Validation target support below 20 positive/20 negative fails the planned selection gate.

The runner filters on the candidate manifest before reading report text, checks report hashes, rejects exclusion reasons and access-policy violations, and never inspects expert label column values. CSV parsing necessarily loads a source row, but label values are never used, printed or retained as outputs. Row labels are restricted to ignored `V2_new_research/outputs/`. The tracked `n01_candidate_coverage.json` contains only aggregate state/mask counts by candidate partition, detected language and target plus hashes. No raw report or UID is stored in the aggregate.

Run from the repository root with the existing isolated audit dependencies on PYTHONPATH:

```powershell
$env:PYTHONPATH = (Resolve-Path V2_new_research/.tools).Path
python -m unittest discover -s V2_new_research/src/research -p test_n01_assertions.py
python V2_new_research/src/research/n01_assertions.py --train V2_new_research/data/metadata/train.csv --manifest V2_new_research/outputs/candidate_split/manifest.csv --output V2_new_research/outputs/n01/candidate_assertions.csv --summary V2_new_research/research/n01_candidate_coverage.json
```

Six synthetic tests cover language/silence masks, contrast and comma negation scope, current versus historical assertions, uncertainty/questions, anatomy and contradictions, degeneration, explicit threshold positives and absent-threshold abstention. No expert report or public candidate labels are used by these tests or rules.
