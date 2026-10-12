# Image-label rubric for report assertion work

Source: the competition host's [pinned clinical overview and label definitions](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733343), checked 2026-10-09. The [host clarification](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733826) says the labels were assigned independently from images, and image labels take precedence over report text when they disagree. This document paraphrases the rubric for a **report-derived weak-label candidate**; a report phrase cannot prove the image finding.

| Target | Positive image threshold to respect in conservative report assertions |
| --- | --- |
| ACL | High-grade partial tear involving more than half the fibers, or complete tear; mild signal or thickening alone is insufficient. |
| MCL | Acute high-grade partial or complete tear; low-grade sprain and chronic stress change are insufficient. |
| Medial / Lateral Meniscus | Definite tear reaching the surface on more than one image or a clear morphologic tear sign in the named meniscus; intrasubstance degeneration alone is insufficient. |
| Medial / Lateral / PF OA | Moderate or large area, around 1 cm or greater, with high-grade cartilage loss exceeding half the thickness in the named compartment. Generic chondropathy or mild cartilage change is insufficient. |
| Effusion | Moderate or large joint fluid collection. |
| Synovitis | Thickened/inflamed synovium. |
| Baker's | Moderate or large popliteal/Baker cyst. |
| Contusion | Impact-related bone marrow edema pattern without a discrete fracture line. |
| Fracture | Acute cortical break or fracture line. |

The host treated borderline cases as negative in expert image annotation. A text extractor must still mark ambiguous or unaddressed report cells as **unknown/masked**; it must not infer a confirmed image-negative label from report silence. Reviewers should separately note current versus remote/history findings, compartment and severity, negation scope, and bilateral-study ambiguity.
