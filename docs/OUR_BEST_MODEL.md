# Our best own model: Ens4

**Public LB 0.857 · Private LB 0.837 · weak-label val 0.880** (Kaggle submission 57228847, Oct 15 2026)

This is the best result from models **we trained ourselves**, using only competition data and generic ImageNet initialization. The 0.958 / 0.942 submissions (F0-F7) ran public checkpoints from other participants and are covered in [RESULTS.md](RESULTS.md), not here.

---

## 1. What Ens4 is, in one paragraph

Four 2.5D CNNs, each reading the knee MRI as a bag of 3-slice "windows" from one sagittal, one coronal and one axial series, all trained on soft labels extracted from the radiology reports. At test time each model predicts 12 probabilities per study. These are converted to per-finding **rank percentiles** across the test set and averaged with equal weights.

| Member | Backbone | Head | Recipe difference | Val (best epoch) |
|---|---|---|---|---:|
| E4+E5 | ResNet18 (ImageNet) | max pooling | base recipe: labels v3, soft targets, clipped pos_weight | 0.8625 |
| E6 | ResNet18 (ImageNet) | max pooling | + anatomy-safe augmentation, 16 epochs, average of last 3 checkpoints | 0.8593 |
| E2 | ResNet18 (ImageNet) | per-finding attention MIL | fixed 12-slice slabs instead of random windows | 0.8461 |
| E8 | ConvNeXt-Tiny (ImageNet-22k → 1k) | max pooling | lr 1e-4, weight decay 0.05 | 0.8744 |
| **Ens4** | | | equal-weight rank average | **0.8800** |

Two members (E6 and E2) were *worse* than the base alone, but they made different errors, so they helped the ensemble (see §5).

---

## 2. Pipeline, step by step

### 2.1 Preprocessing: canonical cache (`V2_new_research/src/canon.py`)

Built once on Kaggle CPU kernels as 5 uint8 memmap shards (~62 GiB). Decoding DICOM on the fly had left the GPU idle 95% of the time; the cache made training about 10× faster, so 12-16 epochs fit on a laptop RTX 4050 (6 GB).

For every series:
1. **Sort slices by physical position** (ImagePositionPatient projected on the slice normal), not by filename.
2. **Canonical orientation:** in-plane flips/transposes only (no interpolation), so that sagittal columns point posterior, coronal and axial columns point to patient-left, and rows point inferior.
3. **160 mm physical square crop/pad** around the image centre.
4. **Antialiased area resize to 256 px** (0.625 mm/px).
5. **Intensity:** Otsu foreground, then the p0.5-p99.8 percentile window, then uint8.
6. **Laterality** (L/R from DICOM tags, else from patient x-position) is recorded but not applied in the cache.

The same `canon_series()` function is bundled into the offline inference notebook. A 20-study parity check showed train and inference inputs matching within ≤3e-7 for every member.

### 2.2 Series and window sampling (`m00_data.py`)

- **3 slots:** one series per plane (sagittal, coronal, axial). Preference: fluid-sensitive, then fat-suppressed, then smallest UID.
- **Window:** 3 physically adjacent slices stacked as RGB channels (2.5D).
- **Training:** 4 random windows per slot per epoch (E2: 12 fixed slab windows).
- **Inference:** 16 evenly spaced windows per slot.
- **Mirroring:** left knees are mirrored into right-knee orientation at load time (reverse slice order for sagittal, flip columns for coronal/axial), so medial/lateral stay consistent.
- Input resized 256 → 224 on the GPU, ImageNet mean/std normalization.

### 2.3 Model (`v3_model.py`, `m00_model.py`)

```
study ──► up to 3 series × N windows ──► shared 2D encoder (ResNet18 / ConvNeXt-T)
                                           │  one 512/768-d feature per window
                                           ▼
          max head:  element-wise MAX over all valid windows of all planes ─► Linear(→12)
          attn head: per-finding gated attention (Ilse 2018) + learned plane bias
                     ─► 12 pooled vectors ─► 12 logits
```

- Padded or missing windows are masked, so they never touch BatchNorm statistics or the max.
- The attention head (E2) learns a separate weighting over windows for each finding, with a learned bias per plane (e.g. it can learn that the sagittal plane matters more for ACL).

### 2.4 Labels: "labels v3" (`relabel_v3.py`)

We started from public LLM report labels (probabilities per finding), then:
- **Severity-aware relabel (E4):** the host rubric counts Effusion and Baker's cyst only when moderate or large. If every non-negated mention of the finding in the report carries a "mild" word (small/trace/minimal/leve/gering/klein…, in 9 languages: EN ES TR HR EL DE BG NL FR) and none carries a moderate+ word, the positive is downgraded to a **soft 0.3**. Rules were written on training reports only and checked on 120 held-out reports.
- **Soft targets (E5):** BCE against the probability itself rather than a 0/1 threshold. Exactly 0.5 ("unknown") is **masked** out of the loss.
- **Clipped pos_weight:** neg/pos per finding, clipped to [1, 10] (ranges 1.0 for Synovitis and Effusion to 5.8 for Fracture).

### 2.5 Training (`train_v3.py`)

| Setting | Value |
|---|---|
| Split | 3,428 train / 885 val studies, patient-grouped, frozen |
| Optimizer | AdamW, cosine schedule; ResNet18 lr 3e-4 / wd 0.01; ConvNeXt-T lr 1e-4 / wd 0.05 |
| Batch | 2 studies |
| Epochs | 12 (E6: 16) |
| Precision | fp16 AMP (never bf16 on T4) |
| Augmentation (E6 only) | ±10° rotation, 8% shift, ±10% zoom, ±15% intensity; **no flips** (they would swap medial and lateral) |
| Hardware | Laptop RTX 4050 6 GB; ~0.5-1.7 h per run |
| Selection | best epoch on weak-label val |

### 2.6 Inference and ensembling (`notebooks/v3_submission/`)

- An offline Kaggle notebook on T4 GPUs loads all 4 checkpoints and decodes each study once.
- For each model and finding: `rank(prob) / n_studies` over the test studies.
- Final score = mean of the 4 rank percentiles (equal weights, never tuned).
- Preflight: 3/3 visible studies, 0 failures; parity max diff 3.0e-7.

---

## 3. How we got there (each step was paired against the previous one)

| Step | Change | Val | Public | Private | Gain (public) |
|---|---|---:|---:|---:|---:|
| E01 | ResNet18, on-the-fly decode, hard labels, 6 epochs | 0.822* | 0.769 | 0.735 | |
| E1 | Canonical cache + ImageNet norm + mirroring + 12 epochs | 0.852 | 0.807 | 0.788 | **+0.038** |
| E4+E5 | Labels v3 + soft targets + pos_weight clip | 0.863 | 0.841 | 0.821 | **+0.034** |
| Ens3 | + E6 + E2, rank average | 0.870 | 0.848 | 0.827 | +0.007 |
| Ens4 | + E8 ConvNeXt-T | 0.880 | 0.857 | 0.837 | +0.009 |

\* different validation split.

**The two largest steps were preprocessing (+0.038) and labels (+0.034).** Each ensemble member added under 0.01.

---

## 4. Per-finding validation AUC (weak labels)

| Finding | E4+E5 | E6 | E2 | E8 |
|---|---:|---:|---:|---:|
| ACL | 0.821 | 0.796 | 0.811 | 0.828 |
| MCL | 0.828 | 0.817 | 0.783 | 0.821 |
| Medial Meniscus | 0.846 | 0.837 | 0.818 | 0.843 |
| Lateral Meniscus | 0.800 | 0.792 | 0.777 | 0.813 |
| Medial OA | 0.894 | 0.891 | 0.878 | 0.904 |
| Lateral OA | 0.866 | 0.875 | 0.853 | 0.879 |
| PF OA | 0.839 | 0.844 | 0.824 | 0.858 |
| Effusion | 0.867 | 0.864 | 0.848 | 0.875 |
| Synovitis | 0.954 | 0.958 | 0.972 | 0.956 |
| Baker's | 0.920 | 0.916 | 0.887 | 0.921 |
| Contusion | 0.817 | 0.815 | 0.808 | 0.856 |
| Fracture | 0.899 | 0.907 | 0.894 | 0.938 |
| **Macro** | **0.8625** | **0.8593** | **0.8461** | **0.8744** |

Weakest everywhere: ACL, MCL and both menisci. These are small structures that 4 random windows at 224 px often miss.

---

## 5. Why the ensemble worked

- **E6** lost 0.003 alone, but **E4+E5 + E6** gained +0.005 [0.002, 0.008].
- **E2** lost 0.016 alone, but **+ E2** brought the 3-model rank average to +0.008 [0.004, 0.012] over E4+E5.
- **E8** (different backbone family) gained +0.0097 [+0.0076, +0.0118] on top of Ens3, **better on 12/12 findings**, and it showed up on the LB as +0.009.

Members with a different backbone, head or sampling helped. Members that were near-copies would not have.

---

## 6. Why it still scored only 0.837 private

1. **The labels capped us.** Report-derived labels reach only ~0.88-0.90 AUC against the experts. Our model learned "what the report mentions", while the test was graded "from the images". On the 58 expert studies, E01 scored 0.735; the labels themselves scored 0.878.
2. **Validation was optimistic in the wrong direction.** Our val measured agreement with report labels. Our LB was always 0.02-0.045 **below** val; other teams saw LB above val. Making the model fit the reports better did not make it fit the experts better.
3. **Too little image per study.** One series per plane, 4 training windows, 224 px from a 256 px cache. Top teams used 15-48 slices per series, all series, 384 px, plus anatomy-centred crops.
4. **No image-based labels and no OOF self-distillation in the own-model line.** We built a teacher (T1) late, and its student hurt the public stack (F1: −0.004).
5. **Compute.** One laptop GPU plus ~30 Kaggle GPU-hours per week, against top teams running 5-fold families on 8-24 GPUs.

---

## 7. What we would do differently next time

1. Spend the first days on **label quality, measured on the gold set**: a vision LLM reading the images, plus per-finding fixes against the host rubric.
2. Build a **bigger input** early: all series, 15+ windows, 384 px, a knee-centred crop plus the full view.
3. Use **grouped K-fold OOF** for every model family, so self-distillation and blend weights are honest.
4. Treat the **gold set as the validation that counts**, and weak-label val only as a training-loss sanity check.

See [LESSONS_FROM_TOP_SOLUTIONS.md](LESSONS_FROM_TOP_SOLUTIONS.md) for the playbook.

---

## 8. Reproduce

| Piece | Path |
|---|---|
| Cache build | `V2_new_research/notebooks/cache_v3/`, `src/canon.py` |
| Labels v3 | `V2_new_research/src/relabel_v3.py` (+ `tests/test_relabel_v3.py`) |
| Training | `V2_new_research/src/train_v3.py` |
| Model | `V2_new_research/src/v3_model.py`, `src/m00_model.py` |
| Run histories | `V2_new_research/experiments/completed_runs/{e45_v3,e6_v3,e2_v3,e8_cnxt}/log.json` |
| Submission notebook | `V2_new_research/notebooks/v3_submission/` |

Example (E8 member):

```bash
python src/train_v3.py --cache outputs/cache_v3/s0 ... outputs/cache_v3/s4 \
  --labels v3 --target soft --pos-weight-clip 10 --epochs 12 --res 224 \
  --backbone convnext_tiny --lr 1e-4 --wd 0.05 --mirror --out outputs/e8_cnxt
```

The cache, checkpoints, reports and label tables are not in Git (competition data rules). They live in private Kaggle datasets (`aymenslimani/rsna-knee-v3-ens4`).
