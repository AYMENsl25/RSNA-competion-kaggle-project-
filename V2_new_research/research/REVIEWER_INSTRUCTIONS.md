# Blinded MRI report annotation instructions

Review only the deidentified **report text** in the provided packet. This task checks whether a report explicitly supports the competition's 12 findings; it is not a review of MRI images or a patient diagnosis. Apply the organizer's [finding definitions](LABEL_RUBRIC.md) for severity and anatomy. Do not consult candidate model outputs, existing weak labels, public scores, or the 58 protected expert-labeled studies.

For each finding, enter exactly one state in its `state` column:

| State | Use when |
| --- | --- |
| `positive` | The report explicitly supports the organizer-defined finding, including any required severity, acuity and compartment. |
| `negative` | The report explicitly rules out that organizer-defined finding at the relevant anatomy/threshold. |
| `uncertain` | The report mentions a possible, borderline, historical, postoperative, insufficiently severe, or otherwise ambiguous finding. |
| `not_addressed` | The report does not discuss the finding enough to decide. Silence is not negative. |

Copy a short exact supporting phrase into each finding's `evidence span` column when the state is positive, negative or uncertain; leave it blank for `not_addressed`. If wording conflicts between sections, record the conflict in `notes` and use `uncertain` pending adjudication. Preserve the report's original language. Do not infer a compartment, severity, acute fracture, contusion, synovitis, or ligament grade from a generic term. The host image labels can differ from report assertions, so report judgments must stay labeled as **report-derived**.

Fill in `reviewer_id` and `review_date` (YYYY-MM-DD). A second qualified reviewer should adjudicate threshold-sensitive or conflicting cells using the same rubric while blinded to candidate assertions. Unresolved disagreements stay uncertain/masked. Do not alter the report text, review ID, row order or packet columns. Return the completed packet to the project owner for local aggregation; it must not be uploaded to a hosted labeling service by this workflow.
