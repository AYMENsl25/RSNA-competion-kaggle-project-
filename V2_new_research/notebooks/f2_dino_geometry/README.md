# F2 local frozen DINOv2 pilot

This pilot uses the local RTX 4050 for frozen feature extraction, followed by CPU
float32 fitting of two small heads. **Kaggle GPU usage is zero.** It does not call
Kaggle, submit predictions, read reports, or evaluate expert labels.

## Controlled comparison

- Encoder: generic DINOv2 ViT-S/14, frozen, 224 px, 3 adjacent grayscale slices as
  the three channels; normal ImageNet channel normalization.
- Inputs: existing N4 384 px canonical uint8 cache, full 160 mm FOV resized to 224.
  One distinct series per plane/fluid slot, fat-suppression then UID preference;
  8 deterministic windows per selected series. Missing slots are masked. Failed
  selected series fail the study; they are not substituted. The measured cache has
  one failed training study (flagged bad slices). Both arms exclude that same study;
  the explicit gate permits at most one training failure and zero validation failures.
- Same features, split, public-label v1 hashes, masked hard targets, seed, batches,
  optimizer, schedule and macro-AUC/BCE/earliest-epoch selection rule for both arms.
- A: projected frozen features plus finding-specific attention pooling.
- B: same shared initialization, plus slot/physical-position embeddings and one
  within-series Transformer block. Series are not assumed to be registered.
- B adds capacity as well as context/geometry. A gain would justify a subsequent
  same-capacity/no-geometry ablation; it would not isolate geometry alone.
- Twelve head epochs; AdamW 3e-4, weight decay 0.01, cosine, seed 0. This is a new
  head-learning experiment, not a matched P00 image-training reproduction.

## Run

From the repository root, using `V2_new_research/.venv/Scripts/python.exe`:

1. `V2_new_research/tests/test_f2_dino.py` — synthetic CPU contracts.
2. `V2_new_research/notebooks/f2_dino_geometry/run.py --mode preflight`.
3. `V2_new_research/notebooks/f2_dino_geometry/fetch_weights.py` — explicit one-time
   generic encoder download from Meta's official URL. No training data is sent.
4. `V2_new_research/notebooks/f2_dino_geometry/run.py --mode run`.

After the completed run, a CPU-only matched-capacity diagnostic can use:

`V2_new_research/notebooks/f2_dino_geometry/ablate_geometry.py <run-directory>`

This fits C with B's exact architecture, initialization, minibatch ordering and
selection rule, but zeros the five physical-geometry inputs. Slot embeddings and
within-series context remain. It writes a new `geometry_ablation/` directory and
refuses to overwrite it. It is a post-pilot diagnostic on the reused fold.

`V2_new_research/notebooks/f2_dino_geometry/verify_artifacts.py <run-directory>`
reloads A, B and C strictly on CPU, reproduces every stored validation prediction,
and records checkpoint/prediction digests in `artifact_verification.json`.

All outputs go to a new UTC-stamped `outputs/f2_dino_geometry/` folder, including
source/input hashes, benchmark, status, feature-completion marker, per-epoch metrics,
distinct `f2_A_best.pt` / `f2_B_best.pt` checkpoints, held-out probabilities and the
paired comparison. UID order is private and never printed. Keep outputs out of Git.

## Budget

The independent supervisor kills the worker after two local wall hours, including
setup, benchmark, extraction and both heads. The encoder phase has a 90 minute
limit. A 16-study benchmark must project within the extraction budget with 50%
headroom while preserving 20 minutes for the heads and five minutes for saving.
The worker has no DataLoader subprocesses, so a supervisor kill cannot leave a
separate encoder worker running. Partial features have no completion marker and
must not be used. No Kaggle run is prepared or launched automatically.

## Interpretation

The frozen P00 cohort is 3,428 train / 863 validation studies, excluding all 65
expert-connected studies. One recorded cache failure leaves 3,427 train / 863
validation for both heads. Only eligible proxy values are parsed. This experiment
measures report-proxy agreement. The same fold selects epochs, so the group
bootstrap interval is exploratory. No independent full-F0 comparison exists in
this pilot, and no expert-image or leaderboard gain can be claimed.

Cache index digests and array shapes/bounds are verified. The cache's recorded
slice-array SHA256 values are retained, not recomputed over about 120 GB. The
generic encoder file is hashed and loaded strictly through timm's local-file
pretrained filter; implicit hub/network access is disabled during the run.

## Completed local result (October 20, 2026)

Main run: `outputs/f2_dino_geometry/20261020T113047Z`, 512.39 s, COMPLETE.
Feature extraction: 443.25 s. Extra CPU C diagnostic: 50.78 s, COMPLETE.

| Head | Weak-label validation macro AUC | Parameters |
|---|---:|---:|
| A: attention pooling | 0.780321 | 69,656 |
| B: series context + physical geometry | 0.798874 | 220,184 |
| C: same architecture as B, geometry zeroed | 0.797473 | 220,184 |

B minus A: +0.018553, exploratory group-bootstrap 95% interval
[+0.012153, +0.025107]. B minus C: +0.001401, exploratory interval
[+0.000079, +0.002633]. The same fold selected checkpoints; C was added after
seeing the first comparison. These intervals are not independent confirmation.
All selected checkpoints were epoch 12. Reload parity is exact for all 863 x 12
stored predictions for each head. Ten synthetic CPU contract/integration tests
passed. This experiment consumed zero Kaggle GPU hours and made no submission.

The larger context/series head explains most of the observed gain. Physical
geometry has a smaller incremental signal on this endpoint. There is no measured
improvement over F0 and no basis for adding this head to F0 yet.
