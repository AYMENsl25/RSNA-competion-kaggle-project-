# F2 local DINO pilot — October 20, 2026

## Outcome

The approved frozen-DINO comparison completed locally, with **zero Kaggle GPU
hours consumed**. The laptop RTX 4050 encoded the images; CPU float32 trained
the heads. No notebook was pushed, no submission was made, and no competition
model weights, raw reports, expert labels or final test labels were used.

| Head | Validation macro AUC | Masked BCE | Selected epoch | Parameters |
|---|---:|---:|---:|---:|
| A: attention pooling | 0.780321 | 0.511151 | 12 | 69,656 |
| B: within-series context + physical geometry | 0.798874 | 0.490006 | 12 | 220,184 |
| C: B architecture with geometry inputs zeroed | 0.797473 | 0.490895 | 12 | 220,184 |

The endpoint is agreement with the frozen public report-proxy labels. These
numbers are **not comparable to F0's 0.958 public leaderboard AUC**.

## Interpretation

- B minus A: **+0.018553** macro AUC; paired 839-group, 1,000-draw bootstrap
  percentile interval **[+0.012153, +0.025107]**. Eleven findings improve and MCL
  declines. Largest increases are lateral OA (+0.05464), lateral meniscus
  (+0.03315), medial OA (+0.03250), fracture (+0.02510) and PF OA (+0.02299).
- This first comparison changes capacity, series embeddings, context and geometry
  together. It cannot attribute the gain to physical coordinates.
- After inspecting that result, a CPU-only C diagnostic reused the exact saved
  features. It matches B's model construction/initialization, dropout seeds,
  minibatch ordering, optimizer and checkpoint selector, but zeros all five
  geometry inputs. Series embeddings and within-series context remain.
- B minus C: **+0.001401**, exploratory interval **[+0.000079, +0.002633]**.
  The geometry contribution is much smaller than the combined B-versus-A effect.
- All heads selected epoch 12. The validation fold selected the checkpoints, and
  C was added after the first comparison. Bootstrap intervals are exploratory;
  this is one seed and not independent confirmation.
- No honest full-F0 held-out predictions were available. Nothing here establishes
  incremental value over F0 or supports submitting an F0 blend.

## Data and experiment contract

- Existing N4 canonical 384 px, 160 mm field-of-view caches, resized to 224 px.
  Generic Apache-2.0 DINOv2 ViT-S/14, frozen. Three adjacent cached slices are
  input as channels; ImageNet channel normalization; no encoder backward pass.
- At most one distinct series per plane/fluid slot, ranked by fat suppression
  then UID. Eight deterministic centers per series; no cross-series windows.
  Missing slots are masked. This same fixed feature cache is used by every head.
- Geometry: median-centered physical position, preceding/following gaps, physical
  extent and relative position. No assumption that separate series are registered.
- Existing split SHA256:
  `dbe879ab5cb719e09c0f9aec612d5b6b53648d6f32dd8d405a075482339b3819`.
- Existing public weak-label v1 SHA256:
  `96444cd3c7d396bae4a71454317d9cbe0689759b5278b30939781e04d0774537`.
  Probabilities above/below 0.5 become hard targets; exactly 0.5 is masked.
  Only eligible study proxy values are parsed; excluded values are not parsed.
- Frozen eligible cohort: 3,428 train / 863 validation. Preflight detected one
  training study with flagged bad slices in selected cached series. It is recorded
  as a study failure and excluded from every head; no replacement series is used.
  Actual fitted/evaluated cohort: **3,427 train / 863 validation**. Zero validation
  failures; the 65 expert-connected studies remain excluded.
- AdamW 3e-4, weight decay 0.01, cosine over 12 head epochs, batch 64, seed 0.
  Shared A/B projection and pooling parameters start identically. Heads fit on CPU
  in float32, with finite loss/gradient checks and norm clipping.
- Selector: highest validation macro AUC, then lower masked BCE, then earlier
  epoch. This is a new head experiment, not a P00 image-training reproduction.

## Resource evidence and checks

- Sixteen-study local benchmark: 592 windows in 1.5 s, reserved GPU peak 210 MiB.
  Projected extraction including 50% headroom: 630.94 s.
- Full feature extraction: **443.25 s**, 4,290 studies, 166,008 windows.
- Main run including setup, extraction, both heads and bootstrap: **512.39 s**.
- Additional C training/analysis: **50.78 s**, CPU only.
- Main independent supervisor hard limit: two local wall hours. Extraction limit:
  90 minutes, with projection gating and save/head reserves. No Kaggle GPU session
  or API call is part of the experiment.
- Ten synthetic CPU contract/integration tests passed. They cover ranking and
  duplicate UIDs, failed-series handling, physical gaps and translation invariance,
  malformed geometry, exclusion of protected values, group isolation, padding,
  gradients, masked targets, paired fitting, checkpoint selection and output shape.
- A, B and C checkpoints reload strictly. Recomputed probabilities for all
  **863 x 12** validation cells match saved predictions **exactly (max difference
  0.0)**. Source/input/feature/checkpoint/probability hashes are recorded.
- Cache index SHA256 and array headers/bounds are verified. The existing recorded
  hashes of approximately 120 GB of slice arrays are retained but not recomputed.

## Artifacts and exact changes

New implementation files:

- `src/f2_data.py`: frozen-cohort/QC/series selection and physical window geometry.
- `src/f2_model.py`: the two head architectures and masked BCE.
- `notebooks/f2_dino_geometry/config.json`: explicit local budget and experiment.
- `notebooks/f2_dino_geometry/run.py`: preflight, benchmark, supervised local run,
  feature extraction, paired training and exploratory comparison.
- `notebooks/f2_dino_geometry/fetch_weights.py`: explicit generic encoder download.
- `notebooks/f2_dino_geometry/ablate_geometry.py`: CPU-only C diagnostic.
- `notebooks/f2_dino_geometry/verify_artifacts.py`: CPU checkpoint/prediction parity.
- `notebooks/f2_dino_geometry/README.md`: commands, controls, interpretation/results.
- `tests/test_f2_dino.py`: ten synthetic CPU tests.
- This result note.

New local generic encoder and provenance:
`assets/f2_dinov2/dinov2_vits14_pretrain.pth` and `manifest.json`.
Downloaded 88,283,115 bytes from Meta's official DINOv2 URL; SHA256
`b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9`.

New outputs: `outputs/f2_dino_geometry/20261020T113047Z/`, including the
`geometry_ablation/` subdirectory, source/input manifests, complete feature cache,
checkpoints, probabilities, histories, comparisons and `artifact_verification.json`.
The earlier `20261020T112908Z/` folder preserves the initial strict preflight failure;
it contains no fitted model. Python bytecode caches were also generated for new files.
No existing P00/D01/S01/F0 training sources or historical results were edited.

## Decision

Keep the four-hour Kaggle GPU allowance unspent by this experiment. Preserve F0.
The immediate useful evidence is that a modest series/context head adds substantial
weak-label signal to the frozen encoder at negligible head-fitting cost. Geometry's
increment is smaller and needs confirmation. A next experiment should address local
spatial evidence or assess real complementarity with F0 under an honest validation
protocol; repeating a generic all-target student blend is not supported here.
