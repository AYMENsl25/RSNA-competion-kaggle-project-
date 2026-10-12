# Private image confirmation without embedded identifiers

This CPU Kaggle script reads the official mounted competition files and the output of our private `rsna-knee-image-duplicate-screen` notebook. Its uploaded source has no study UID, series UID, report, label, or image payload. The first candidate screen's 272 pairs are compared using geometry-sorted, higher-resolution slices; 100 seeded same-sequence random pairs are diagnostic controls. All 58 expert studies are protected from pixel decoding. Only expert-row completeness is checked to identify their UIDs, and their label values are not used.

The script writes aggregate similarity categories plus a private CSV keyed by SHA-256 of the candidate UID pair. Those hashes can be joined to the local candidate list after download; no UID or image is exported from Kaggle by this notebook. Similarity thresholds are audit triage, not proof of identical patients. Unresolved pairs keep the development split unfrozen.

`kernel-metadata.json` attaches the previous private screen as a kernel source and disables internet and GPU. The script caps execution at 1,500 seconds or 3,800 pixel decodes. It contains no model training or competition submission.
