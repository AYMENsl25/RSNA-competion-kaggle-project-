# M00 full training (three-epoch, report-proxy labels)

Draft only. **Not run, not uploaded.** Implements `research/M00_THREE_EPOCH_CONTRACT.md`.

## Build

`train_template.py` is the reviewed source. `python src/research/build_standalone_notebooks.py`
embeds the audited `src/m00_data.py` and `src/m00_model.py` (plus SHA256 of both and of the
template) into `train.py`, the single file uploaded to the Kaggle script kernel. That builder
regenerates every notebook, not only this one. CPU-only synthetic check (no data, needs torch,
torchvision and pydicom installed): `python notebooks/m00_full_train/train.py --self-check`.

## Kaggle inputs (private, offline, GPU)

- Competition `rsna-knee-abnormality-detection`: `train_series.csv` + DICOMs only. `train.csv` is
  never opened, so no reports or expert target columns are read.
- `stevenleehans/rsna-knee-llm-report-labels` v1 → `llm_labels_full.csv`, SHA256 `96444cd3…0774537`.
- `aymenslimani/rsna-knee-v2-frozen-dev-split` → `split.csv`, SHA256 `dbe879ab…3b3819`.

Each attached file must be found exactly once under its fixed mount candidates and match its hash.

## Guards before any DICOM decode

Split header `StudyInstanceUID,group_id,partition`; partitions `candidate_train`/`candidate_validation`/`excluded`
with counts 3457/885/65; no duplicate UID; no `group_id` across partitions; split UIDs equal the
`train_series.csv` universe. Label header exactly `StudyInstanceUID` + the 12 `m00_model.TARGETS`;
no duplicate/unknown UIDs; every train/validation study labelled; values finite in [0,1];
p>0.5→1, p<0.5→0, p==0.5→masked. Exactly 29 all-masked training studies dropped and 31,088
observed training cells; ≥20 positives and negatives per target in validation. Excluded studies are
never decoded (their label values are not even parsed).

## Run behaviour

Random-init ResNet18 2.5D, 12 train / 48 validation windows, shared max, no flips, no cache,
batch 2 / 1, 4 workers, FP16 autocast + GradScaler, masked BCE, AdamW 3e-4, per-step cosine, seed 0.
Each epoch: per-target tie-aware ROC-AUC and pos/neg support, macro AUC, masked BCE, failed studies
by exception type, zero-observed skips, AMP skipped steps, valid windows, epoch and cumulative wall
time, peak reserved VRAM. Best checkpoint by macro AUC → lower BCE → earlier epoch.

Time: graceful stop at 7.5 h after script start (30 min reserved for saving); a separate watchdog
sends SIGTERM at 7 h 59 min and force-stops the parent at 8 h if it does not exit. Before epochs 2–3 the
slowest measured epoch × 1.2 must fit before that deadline. Stopped or crashed runs write
`status: incomplete`; only three finished epochs give `complete`. Nonfinite logits/loss, shape or
mask errors, duplicate or unaccounted studies abort the run (still recorded as incomplete).

Outputs in `/kaggle/working`: `m00_best.pt` (model state, config, source hashes, epoch, metrics),
`m00_val_predictions_epoch{N}.csv` (UID + 12 probabilities, no labels), and
`m00_full_train_summary.json` (aggregate only).

## Limitations

- Validation AUC measures mimicry of thresholded public report labels, not expert image labels.
- Label column names are checked against `m00_model.TARGETS`; partition values are checked against
  the frozen `candidate_train`/`candidate_validation`/`excluded` file before decoding.
- The 8 h cap is measured from script start; Kaggle container startup before that is not counted.
  The watchdog is a child process independent of the training loop, with a final force-stop at 8 h.
- No resume: an interrupted run keeps only the best checkpoint from completed epochs.
- Seeded but not bitwise deterministic (cuDNN / `scatter_reduce` backward are nondeterministic).
- Any series failure fails the whole study (as in the integrated pilot); failures are counted and
  make the run incomplete. The 22 validation studies with zero observed targets are skipped.
- No expert check, M01, test inference or submission.
