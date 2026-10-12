# AI report review v1 audit — 2026-10-10

The user supplied two AI annotation passes and adjudication for each prepared report packet under ignored `outputs/ai_review_v1/`. The returned CSVs are **AI-generated report assertions, not clinical judgments or image ground truth**. This audit did not inspect the 58 protected expert labels or run a model.

## Integrity and support

- Packet A: 60/60 complete, original report/ID/order fields unchanged, literal evidence spans and allowed states valid; returned SHA-256 `8e14e15ddcf56b92edf639c5eda191e9b58f4a384441c490e7d8277dcc2a2e02`.
- Packet B: 120/120 complete under the same checks; returned SHA-256 `652ad4416dee08caf06eb2aee1558f430786e068cb625ebb4ee048f27a022745`.
- Both pass JSONs cover all assigned reports and 12 targets with valid states/spans. Recomputed disagreements equal saved dispute counts: A **33/720** cells, B **91/1440** cells. Every disputed cell has an adjudication. Re-merging from the source passes/adjudications gives zero state/span mismatches against returned CSVs.
- The reviewer field names Claude Opus 5.5 for pass 1, Claude Sonnet 5.5 for pass 2, and Claude Opus 5.5 for adjudication. The isolation prompt is present; execution/session logs that would prove independent access and model provenance are absent. Agreement is not accuracy evidence. The adjudicator selected pass 1's state in 105/124 state disputes, pass 2's in 15, and a third state in four.

| Target | A positive / negative | B positive / negative | Both meet 20/20? |
| --- | ---: | ---: | --- |
| ACL | 22 / 31 | 15 / 83 | No |
| MCL | 3 / 43 | 4 / 92 | No |
| Medial Meniscus | 24 / 28 | 44 / 58 | **Yes** |
| Lateral Meniscus | 16 / 34 | 25 / 84 | No |
| Medial OA | 8 / 36 | 14 / 64 | No |
| Lateral OA | 4 / 34 | 3 / 75 | No |
| PF OA | 7 / 38 | 13 / 58 | No |
| Effusion | 23 / 27 | 24 / 65 | **Yes** |
| Synovitis | 13 / 0 | 24 / 1 | No |
| Baker's | 6 / 25 | 14 / 40 | No |
| Contusion | 15 / 7 | 19 / 51 | No |
| Fracture | 7 / 10 | 9 / 49 | No |

## Scientific limits and decision

The prompt classified explicitly below-threshold findings as `negative`, while `REVIEWER_INSTRUCTIONS.md` described insufficient severity as `uncertain`. This changes the observation mask and must be resolved before any reference freeze. A literal phrase check confirms the evidence occurs in the report; it does **not** establish correct negation, acuity, grade, compartment or image-label equivalence. The merge code adjudicates state disagreements and invalid spans, but same-state shared errors can survive. Fracture had 25 pass disagreements in B, making threshold-sensitive findings particularly uncertain. Some positive fracture phrases describe subchondral insufficiency fractures, whose fit to the host's acute-fracture definition requires special scrutiny.

**No-go for full 12-target M00/M01 training under the approved gate.** Only two targets satisfy the predeclared 20-positive/20-negative requirement in both packets. The available AI packets can inform training-side labeler development (A) and a limited report-assertion reference (B), provided B remains isolated from rule revisions. They cannot support a 12-target macro-AUC claim or a claim of clinician-verified labels. The runtime/storage gate also remains failed independently.

The user cannot arrange clinical review for this competition. Next decision is a documented AI-only evaluation redesign, including one consistent below-threshold rule, stronger provenance, and a prespecified way to handle rare targets; this would revise the earlier clinical-review gate and must not be presented as equivalent evidence. Until then, preserve the split and packets as frozen sampling artifacts and do not launch full M00 or a third GPU pilot.
