# Blinded report review handoff — 2026-10-10

Two reviewer archives are prepared locally under `../outputs/report_review_packets_v1/`:

| Archive | Report count | Purpose | SHA-256 |
| --- | ---: | --- | --- |
| `reviewer_packet_A.zip` | 60 | Training-side rule-development review | `2f5054302eda9fc58a595f4c16ba71aad78f013590fd57b0fc93a945e543cfd3` |
| `reviewer_packet_B.zip` | 120 | Blinded held-out reference review | `da2634e6be82d603d151ba751e630992ea56d0631ac5ae416dfcc00c08cc3f8d` |

Each archive contains only its report CSV, `REVIEWER_INSTRUCTIONS.md` and `LABEL_RUBRIC.md`. Archive listings were checked: **no private UID maps** are included. Original-language reports are still sensitive study data; the project owner should distribute them only to qualified clinical/language reviewers under applicable competition and data-sharing terms. This workflow has not transmitted them to anyone.

Give the training-side packet to reviewers working on the assertion rules. Keep the held-out packet's judgments blinded from rule revisions and model choices. Reviewers should fill only blank annotation, reviewer and notes fields, preserve `review_id` and row order, and return the completed CSVs locally. Clinical adjudication is needed for threshold-sensitive cells. The protected 58 expert-labeled studies are absent.

After return, validate row IDs, allowed states, evidence spans, reviewer identity/date, adjudication and per-target positive/negative support. Freeze the reviewed binary reference and its hash before M00/M01 selection. The current packets are **unannotated**; their 120 held-out reports may not supply 20 positive and 20 negative examples for every finding, so the predeclared support gate must be checked rather than presumed.

The local checker is `../src/research/check_review_packet.py`. Run it separately for each returned CSV with the original unannotated packet as the first argument and the returned CSV as the second. It reports `incomplete`, `invalid` or `complete`, plus per-target support counts, without printing report text or identifiers. A `complete` structural check does not replace clinical adjudication or freeze the reference.
