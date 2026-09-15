# RSNA v2 local validation — 2026-09-14

- Generated nine standalone Kaggle notebooks from shared source.
- All notebook JSON schemas and Python code cells validated.
- Embedded source bootstraps and preprocessing/training/evaluation/submission imports executed offline for all nine notebooks.
- Synthetic test suite: **14 passed**. Covers supervision/folds/group isolation, conflict resolution, target-weight normalization, DICOM spatial ordering, full cache writing/resume/hash contract, masked pooling, a real synthetic CPU training epoch, checkpoint reload, final refit, and submission schema/UID order.
- `git diff --check` passed.

No real competition images were preprocessed locally, no patient-data training was launched, no Kaggle GPU run was performed, and no score improvement is measured by these tests. Real decoder availability, image retention, session memory/runtime and final hidden-test execution require Kaggle pilot evidence.

Validation runtime used local Python 3.14 / PyTorch 2.9 CPU / torchvision 0.24, with pydicom 3.0.2, timm 1.0.29 and project-local test dependencies. Kaggle package versions may differ; each notebook records environment information and checks required imports.
