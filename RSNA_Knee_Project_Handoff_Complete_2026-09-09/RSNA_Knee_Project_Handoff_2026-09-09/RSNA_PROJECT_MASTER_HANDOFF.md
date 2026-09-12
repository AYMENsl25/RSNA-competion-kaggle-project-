# RSNA Knee Abnormality Detection — Master Project Handoff

**State date:** 2026-09-09  
**Purpose:** Give a new research/ML agent enough context to restart the project without repeating already-resolved work, understand every major experiment and failure, audit the public ~0.94 solutions, and propose the next high-value experiments.

---

## 1. Competition objective

The task is multilabel prediction from knee MRI studies for 12 findings:

1. ACL
2. MCL
3. Medial Meniscus
4. Lateral Meniscus
5. Medial OA
6. Lateral OA
7. PF OA
8. Effusion
9. Synovitis
10. Baker's
11. Contusion
12. Fracture

Primary metric: **macro ROC-AUC**, i.e. each target contributes equally regardless of prevalence.

The competition training data contains **4,407 studies**. There are **58 official gold-labeled studies** and 4,349 initially unlabeled reports. Two special studies were excluded from the eligible pseudo-label pool, leaving 4,405 eligible studies.

The project deliberately preserves a clean local benchmark:
- 58 studies: `GOLD_EVAL_LOCKED`
- 6 non-gold studies in the same near-duplicate report groups: `GOLD_GROUP_LOCKED_NON_GOLD`
- 4,341 studies: `PSEUDO_TRAIN`

The grouping key is `cv_group_id`. No gold `cv_group_id` may appear in pseudo training.

---

## 2. Non-negotiable scientific contracts

### 2.1 Gold isolation

The 58 official gold studies are not used for:
- architecture selection,
- loss selection,
- epoch selection,
- pseudo-fold selection,
- ensemble weight tuning.

They are used only as a final locked evaluation after choices are frozen.

This rule can be relaxed **only for a final submission-only refit after all choices are frozen**, while preserving previous locked-gold results as the unbiased research audit.

### 2.2 Macro-target loss

The frozen weak-supervision objective is:

1. calculate BCE for each target;
2. multiply by the target-specific supervision weight;
3. normalize each target independently by its active total weight;
4. average the 12 target losses equally.

This is intentional because the Kaggle metric is macro AUC.

### 2.3 Public leaderboard score is not a clean architecture score

A public ~0.94 notebook may combine:
- many trained members,
- multiple folds,
- multiple backbones,
- multiple preprocessing layouts,
- rank transforms,
- per-target blend logic,
- calibration,
- public leaderboard iteration.

It must not be compared directly to a single clean local model.

---

## 3. Dataset / leakage freeze

A dataset freeze was created with `cv_group_id` to protect against near-duplicate report leakage.

Important role counts:
- `PSEUDO_TRAIN`: 4,341
- `GOLD_EVAL_LOCKED`: 58
- `GOLD_GROUP_LOCKED_NON_GOLD`: 6

A crucial rule for every future agent:

> Never randomly split by study UID without respecting `cv_group_id`.

---

## 4. Report supervision pipeline

### 4.1 Rule teacher v0.2

Freeze:
`PHASE4A1_RULE_TEACHER_v0_2_MECHANICAL_FIX`

SHA-256:
`10bb470b6473db508b660b22fe0cdbb3b4489487ef1f36bbcf55bf310ff6294f`

The update fixed a negation substring bug using token/phrase boundaries and bounded negation scope.

Gold audit:
- strict macro ROC-AUC: **0.6871**
- broad macro ROC-AUC: **0.7080**
- strict coverage: **28.59%**
- covered balanced accuracy: **0.8496**
- hard errors: **22**

The rule expert was frozen as:
`RULE_EXPERT_FREEZE_v1`

It is intentionally high precision / low coverage. Do not silently rewrite the rules.

### 4.2 Multilingual E5

Multilingual E5 embeddings were used to provide continuous semantic scores when the rule system abstains.

Hard E5 rescue increased errors. Therefore E5 is used as a **continuous weak target**, not a hard override.

### 4.3 NLI

Model audited:
`MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`

Verified SHA:
`65af59b1ff4450b09ecbf13ca35c840dbf038b26ff8e10e5ea89ca724828ed1e`

NLI was useful diagnostically but globally weaker than the E5 fusion. It often returned neutral on positive radiology findings.

**Final decision:** NLI is not in the frozen pseudo-label fusion.

### 4.4 Best report teacher

Best trustworthy report teacher:

**Frozen Rule v0.2 + continuous multilingual E5**

Locked-gold macro AUC:
**~0.7606700701**

This remains materially stronger than our current MRI-only models on the 58-study audit.

---

## 5. Frozen pseudo-label supervision

Freeze:
`PSEUDO_LABELS_v1_1`

Fusion:
- rule POSITIVE → soft target `0.95`, raw weight `1.0`
- rule NEGATIVE → soft target `0.05`, raw weight `1.0`
- rule abstain → continuous E5 probability `p`
- E5 raw weight → `0.35 * (2 * abs(p - 0.5))`
- NLI → not used

The pseudo-label table contains a target and confidence weight for each of the 12 findings.

Important interpretation:

> Pseudo labels approximate the report-derived semantics. They are not guaranteed to reproduce the exact official competition annotation procedure.

Phase 4B-2B1 provided direct evidence of this mismatch.

---

# 6. MRI Phase 4B-1A — indexing, ordering, loader and QC

The MRI preprocessing phase established:
- study / series indexing;
- anatomical plane labels;
- fat-suppression / fluid-sensitive descriptors;
- DICOM geometry;
- slice ordering;
- decode QC;
- selected baseline slices.

The pipeline stored physical geometry including PixelSpacing and dimensions, which later enabled the 130-mm physical field-of-view audit.

---

# 7. MRI baseline Phase 4B-1B v1.1

Architecture:
- ImageNet-pretrained ResNet18
- 1-channel adaptation
- 2D slice encoder
- mean + max MIL across selected slices
- sequence average within plane
- average across available planes
- 12 logits

Config:
- image size: 224
- train slices: 8
- pseudo-val slices: 16
- gold slices: 32
- batch: 6
- optimizer: AdamW
- LR: 1e-4
- weight decay: 1e-4
- dropout: 0.2
- grad clip: 1.0

Gold result:
- macro ROC-AUC: **0.7097700187**
- macro AP: **~0.5334**

Selection history:
- epoch 1: 0.452864
- epoch 2: 0.433996
- epoch 3: 0.431142
- epoch 4: **0.420995**

Best epoch was 4/4 → right-censored.

### Critical v1.1 loss bug

The intended contract was per-target normalization followed by equal target averaging.

Actual v1.1 code globally normalized all target weights:

`loss = (raw * w).sum() / w.sum()`

This implicitly gave targets with more supervision weight more gradient.

Therefore:
- v1.1 remains a valid historical result;
- v1.1 is **not canonical** for architecture comparisons.

---

# 8. MRI baseline Phase 4B-1B v1.2

v1.2 changed only:
1. correct macro-target loss;
2. matching pseudo-val metric;
3. max epochs 4 → 8;
4. patience 2 → 3;
5. explicit right-censor flag.

Gold result:
- macro ROC-AUC: **0.69935786697**
- macro AP: **0.52892601229**

Per-target AUC:

| Target | v1.2 AUC |
|---|---:|
| ACL | 0.843137 |
| MCL | 0.614512 |
| Medial Meniscus | 0.621394 |
| Lateral Meniscus | 0.631056 |
| Medial OA | 0.717829 |
| Lateral OA | 0.775629 |
| PF OA | 0.746461 |
| Effusion | 0.814907 |
| Synovitis | 0.709677 |
| Baker's | 0.655797 |
| Contusion | 0.622132 |
| Fracture | 0.640278 |

Selection pseudo-val loss:
- e1 0.450108
- e2 0.436161
- e3 0.428761
- e4 0.425599
- e5 0.420231
- e6 0.421889
- e7 0.419367
- e8 **0.416340**

Best epoch was 8/8 → still right-censored.

Interpretation:
- freeze v1.2 as the canonical corrected baseline;
- the ~-0.010 macro difference from v1.1 is too noisy on n=58 to justify reverting the loss contract.

---

# 9. Public high-score audit before 4B-2A

An earlier ~0.922 notebook showed that the public score gap was not produced by one architecture.

The reusable principles were:
- strong self-supervised ViT encoders;
- plane × contrast/FS study slots;
- physical crop normalization;
- neighboring-slice context;
- target-specific study attention;
- multiple folds;
- rank averaging;
- RadImageNet / CoAtNet diversity.

The non-reusable components without independent validation were:
- public-LB-tuned blend constants;
- target-specific blend constants;
- compressed calibration payloads;
- competition-fine-tuned checkpoint evaluation on our locked gold without provenance.

---

# 10. Phase 4B-2A — six-slot / physical-crop audit

The audited slot scheme was:

- SAG_FS
- SAG_NONFS
- COR_FS
- COR_NONFS
- AX_FS
- AX_NONFS

## 10.1 Slot coverage

| Slot | Studies | Coverage |
|---|---:|---:|
| SAG_FS | 4,150 | 94.17% |
| SAG_NONFS | 4,266 | 96.80% |
| COR_FS | 4,248 | 96.39% |
| COR_NONFS | 3,406 | 77.29% |
| AX_FS | 4,407 | 100.00% |
| AX_NONFS | 857 | 19.45% |

Studies by number of available slots:
- 3 slots: 4
- 4 slots: 1,259
- 5 slots: 2,578
- 6 slots: 566

Conclusion:
**missing-slot masking is required**. A rigid six-present-slot assumption is wrong.

## 10.2 16-slice central-band feasibility

Fraction with at least 16 slices in the 12%-88% central band:

| Slot | Fraction |
|---|---:|
| AX_FS | 95.21% |
| AX_NONFS | 99.53% |
| COR_FS | 89.55% |
| COR_NONFS | 86.91% |
| SAG_FS | 93.01% |
| SAG_NONFS | 97.73% |

Conclusion:
16-slice inference is feasible for most series. Fixed-size tensors should use deterministic repeated/nearest sampling when fewer are available rather than dropping the study.

## 10.3 130-mm crop

Selected slot-series: **21,334**

- physical geometry metadata valid: **100%**
- 130-mm crop fits: **99.7797%**
- crop-fit failures: **47**

Conclusion:
130-mm physical-FOV normalization is validated and should be retained.

---

# 11. Phase 4B-2B1 — six-slot 2.5D ResNet18 + target attention

The controlled question was:

> Does the richer study representation improve the clean baseline before changing the backbone?

Architecture:
- 6 plane × FS slots
- missing-slot mask
- 130-mm centered physical crop
- 224×224
- 8 ordered anchors per slot
- each anchor = `[previous, center, next]` 2.5D RGB-like input
- ImageNet ResNet18
- learned within-slot anchor attention
- learned slot embeddings
- 12 target-specific cross-slot attention queries

Selection:
- pseudo-selection train: 3,437
- pseudo-selection val: 904
- gold locked: 58
- best epoch: **5**
- best pseudo-val macro weighted BCE: **0.4533692995707194**
- epoch 6 worsened to ~0.46362

This was a true selection optimum, not right-censored.

Final refit on all 4,341 pseudo-training studies:

| Epoch | Train macro weighted BCE |
|---:|---:|
| 1 | 0.501460 |
| 2 | 0.456052 |
| 3 | 0.430777 |
| 4 | 0.409514 |
| 5 | 0.393232 |

Locked-gold result:
- macro ROC-AUC: **0.6447290368442214**
- macro AP: **0.48095712206109026**

Per-target:

| Target | AUC | AP |
|---|---:|---:|
| ACL | 0.688725 | 0.564435 |
| MCL | 0.503401 | 0.196957 |
| Medial Meniscus | 0.663462 | 0.620834 |
| Lateral Meniscus | 0.637267 | 0.477975 |
| Medial OA | 0.683721 | 0.407483 |
| Lateral OA | 0.605416 | 0.247349 |
| PF OA | 0.701416 | 0.548282 |
| Effusion | 0.816149 | 0.872409 |
| Synovitis | 0.648746 | 0.597421 |
| Baker's | 0.637681 | 0.429893 |
| Contusion | 0.545209 | 0.353771 |
| Fracture | 0.605556 | 0.454677 |

Compared with canonical v1.2, B2B1 improved only:
- Medial Meniscus: +0.0421
- Lateral Meniscus: +0.0062
- Effusion: +0.0012

Large losses included:
- ACL: about -0.154
- MCL: about -0.111
- Lateral OA: about -0.170
- Contusion: about -0.077

### Interpretation

The model successfully optimized the pseudo-label objective but generalized worse to official gold.

This is strong evidence that:
1. increased capacity can fit report-teacher idiosyncrasies/noise better;
2. pseudo-label loss is not a sufficient architecture-selection proxy;
3. stronger representation alone is not enough if the supervision mismatch is not addressed.

Do not spend another large run simply tuning this ResNet18 attention architecture.

---

# 12. MRI / report disagreement case from 4B-2B1

A deterministic locked-gold study with all six slots was reviewed.

Selected UID:
`1.2.826.0.1.3680043.8.498.10170898615867673028696505248839028269`

Key comparisons:

| Target | Gold | MRI probability | Report teacher |
|---|---:|---:|---:|
| ACL | 0 | 0.909 | 0.050 |
| MCL | 0 | 0.064 | 0.050 |
| Medial Meniscus | 0 | 0.923 | 0.050 |
| Lateral Meniscus | 1 | 0.400 | 0.950 |
| PF OA | 1 | 0.136 | 0.702 |
| Effusion | 0 | 0.107 | 0.950 |
| Synovitis | 1 | 0.876 | 0.950 |
| Contusion | 0 | 0.869 | 0.313 |

The clinical report explicitly described:
- intact ACL;
- no medial meniscus tear;
- horizontal tear at the anterior horn of the lateral meniscus;
- patellofemoral osteochondral/cartilage pathology;
- moderate effusion;
- reactive synovitis.

This single case demonstrates two different mismatches:
- some MRI predictions are confidently wrong relative to both report and gold;
- some report-derived labels differ from official gold semantics (e.g. effusion).

Therefore the weak-supervision problem is not simply “extract the report more accurately.” The official label definition and report language can disagree.

---

# 13. Current model ranking

Observed locked-gold MRI-only macro AUC:

1. Historical ResNet18 SliceMIL v1.1: **0.70977**
   - caveat: global-weight-normalized loss
2. Canonical ResNet18 SliceMIL v1.2: **0.69936**
3. Six-slot target-attention B2B1: **0.64473**

Best report teacher:
- Rule v0.2 + continuous E5: **~0.76067**

The report teacher is not available as a test-time modality in the same way as MRI; its main role is training supervision and error analysis.

---

# 14. The new ~0.94 notebooks — critical finding

Two new notebooks were supplied:

- `cohezion-rsna-knee-sota-ensemble.ipynb`
- `the-bee-s-knees-final-rsna-push.ipynb`

They should **not** be treated as two independent successful ideas.

The Bee notebook explicitly says it faithfully reproduces Mattia Angeli public V27 and that the source version displayed **0.940**.

A source comparison of the principal inference stages found the corresponding DINO, A5, RadImageNet, and CoAtNet/final-fusion cells to be **exact matches**.

Therefore these notebooks are two presentations of essentially the same inference graph, with the Bee version adding audit/reproduction instrumentation and explanations.

This is valuable because it gives a highly detailed specification of the public ~0.94 recipe, but it does not provide two independent confirmations.

See `PUBLIC_094_NOTEBOOK_AUDIT.md`.

---

# 15. What the ~0.94 recipe teaches us

The strongest lesson is that the public score comes from **breadth, diversity, dense coverage, and ranking**, not merely a single target-attention layer.

The recipe contains:

## DINOv2 public member bank
- 20 trained MRI members
- DINOv2 small
- 12 transformer blocks
- last 6 blocks trainable in the saved training configuration
- feature dim derived from CLS + patch mean, sometimes focal top-k patch pooling
- physical crop: 130 mm
- cache/input resolution: 336
- geometry-aware slice ordering
- laterality normalization
- multiple overlapping windows
- member-level and target-level rank fusion
- six sequence-aware slots

The default recovered slots are notably more semantic than our Phase 4B-2A plane×FS scheme:

1. SAG_FLUID_FS
2. COR_FLUID_FS
3. AX_FLUID_FS
4. SAG_FLUID_NOFS
5. COR_T1
6. SAG_T1

This is likely an important correction for our next model.

The public head also contains a fixed **soft anatomical slot prior** table for the 12 findings.

## A5 DINOv3
- 5 folds
- `vit_small_patch16_dinov3.lvd1689m`
- `pool=xcodex`
- 6 plane×FS slots
- 130-mm crop
- 336×336
- 16 slices
- central band 12%-88%
- rank aggregation across folds
- contribution: **45%** of the DINO/A5 rank blend

## RadImageNet branch
- RadImageNet ResNet50 encoder
- 224×224
- multiple head families
- E13 member weight inside Rad block: **0.50**
- E10 parent/Rad vote: **0.50**
- second E13 pass on E11 slot layout: **0.15**
- fixed calibration stage with calibration blend **0.40**

These coefficients must be treated as source recipe constants, not automatically transferable hyperparameters.

## CoAtNet / Raptor branch
Four CoAtNet views:
- maxspan view
- dense native-384 view
- reversed maxspan view
- native-384 view with smaller coverage

Source view weights:
- 0.60
- 0.10
- 0.10
- 0.20

The embedded CoAt reader uses:
- CoAtNet RMLP-2
- 384-pixel model resolution
- dense slice coverage (64 or 44 slices depending on view)
- 140-mm crop in the Raptor source
- broad span such as 0.02-0.98
- optional reversed order
- finding-specific gated spatial evidence

A residual CoAtNet family combines checkpoints from epochs:
- 4
- 6
- 8

Those residual ranks contribute **40%** inside the Raptor/CoAt branch.

Final fusion:
- CoAt branch weight: **60%** for most findings
- Lateral Meniscus: **100%** CoAt branch
- final percentile rank transformation

Again: do not copy these weights as “optimal” without independent evidence.

---

# 16. Major differences between our B2B1 and the public ~0.94 recipe

Our B2B1:
- ResNet18
- 224px
- 8 anchors
- 2.5D previous/center/next
- plane×FS slots
- no strong explicit laterality-normalized multi-view ensemble
- one model
- one final refit
- pseudo-label weak supervision

Public recipe:
- DINOv2 + DINOv3 + RadImageNet ResNet50 + CoAtNet
- 336/384/224 multi-resolution
- much denser slice/window coverage
- explicit geometry and laterality normalization
- semantic fluid/T1 slots in the DINO member bank
- soft anatomy slot priors
- 20 DINOv2 members + 5 DINOv3 folds + multiple Rad heads + multiple CoAt views/checkpoints
- rank aggregation at several stages

Therefore B2B1 did **not** really test “the 0.94 architecture.” It tested only one small subset of its ideas with a much weaker encoder and much lower ensemble diversity.

---

# 17. Highest-value next ideas

## Priority 0 — Reproduce the public 0.940 inference benchmark

If the goal includes leaderboard progress, reproduce the audited Bee/V27 pipeline with its pinned inputs.

Purpose:
- establish whether our Kaggle environment can reproduce the source score;
- establish a strong leaderboard reference;
- separate infrastructure problems from modeling problems.

Do not treat the public score as our independent research validation.

## Priority 1 — Sequence-aware slot scheme v2

Replace our plane×FS slots with the public recovered semantic slots:

- SAG_FLUID_FS
- COR_FLUID_FS
- AX_FLUID_FS
- SAG_FLUID_NOFS
- COR_T1
- SAG_T1

Why this is high-value:
- our AX_NONFS slot exists in only 19.45% of studies;
- “non-FS” mixes different structural sequence semantics;
- the public recipe distinguishes fluid-sensitive and T1-like structural sequences.

This change should be audited first for coverage using our Phase 4B-1A metadata.

## Priority 2 — Laterality normalization

The public code explicitly normalizes right knees by flipping appropriate planes using tag + geometry fallback.

Our B2B1 pipeline did not make this a central frozen contract.

Hypothesis:
laterality normalization reduces unnecessary left/right variability and makes target-specific spatial features easier to learn.

This is cheap relative to a new backbone.

## Priority 3 — Strong SSL backbone with conservative fine-tuning

Use:
- DINOv3 ViT-S/16 or
- DINOv2 small

But avoid immediately fine-tuning the whole encoder against noisy pseudo labels.

Recommended staged experiment:
1. freeze encoder;
2. train small study head;
3. unfreeze only final blocks if pseudo-val and stability support it.

This directly addresses the B2B1 lesson that more trainable capacity can overfit weak supervision.

## Priority 4 — Patch-aware pooling, not only global ResNet features

The public DINO branch combines:
- CLS token
- patch mean
- optional focal top-k patch evidence

This may be particularly important for:
- meniscal tears,
- focal cartilage/OA lesions,
- contusions,
- fractures.

## Priority 5 — Soft anatomical slot priors

The public DINO target head includes a soft slot-prior table.

Use a weak additive prior, not a hard routing rule.

Advantages:
- preserves learning flexibility;
- prevents early noisy pseudo supervision from routing every target arbitrarily;
- encodes broad anatomy without public-LB target blend tuning.

## Priority 6 — Dense multi-window inference

Our 8 anchors are sparse.

Test:
- training with a moderate anchor budget;
- inference with more deterministic windows / slice coverage;
- rank-average the windows.

This is relatively cheap if features can be cached.

## Priority 7 — Weak-supervision robustness

B2B1's key failure is supervision alignment.

Promising approaches:
1. teacher-confidence weighting remains frozen as the base;
2. reduce influence of high-confidence report labels that systematically disagree with official label semantics;
3. train multiple folds and use student consensus to detect unstable pseudo targets;
4. do not overwrite pseudo labels based on one student;
5. consider a robust bootstrap objective:
   `target = alpha * frozen_teacher + (1-alpha) * stopgrad(ensemble_student)`
   only for low-confidence E5 targets;
6. preserve rule high-precision labels as anchors.

Any teacher-revision experiment must be independently versioned; do not silently mutate `PSEUDO_LABELS_v1_1`.

## Priority 8 — OOF rank ensemble

The Kaggle metric is rank-based AUC.

For each architecture family:
- produce group-safe out-of-fold predictions;
- convert each target to ranks;
- average ranks across folds;
- compare family diversity by rank correlation.

Do not choose ensemble weights by the 58 gold studies.

Use:
- equal weights first;
- pseudo-OOF / cross-validation evidence for any deviations.

## Priority 9 — Model-family diversity

After a strong DINO family exists, add one genuinely different family:
- RadImageNet ResNet50, or
- CoAtNet dense 2.5D reader.

The public ~0.94 notebook strongly suggests family diversity is worth more than micro-tuning one model.

## Priority 10 — Final submission-only gold refit

After:
- architecture,
- pseudo-label method,
- epoch count,
- ensemble members,
- ensemble weights

are frozen without gold tuning, consider a final training refit that includes the 58 official gold labels with stronger supervised weight.

This uses all provided training information for the actual competition submission while preserving the earlier locked-gold results as the research audit.

---

# 18. Suggested next experiment sequence

### Experiment A — Sequence-slot audit v2
No training.
Measure coverage of:
- SAG_FLUID_FS
- COR_FLUID_FS
- AX_FLUID_FS
- SAG_FLUID_NOFS
- COR_T1
- SAG_T1

STOP GATE:
the new slot design should cover nearly all studies with enough useful slots.

### Experiment B — DINO frozen-feature baseline
- DINOv3 ViT-S/16
- 130-mm crop
- sequence-aware slots v2
- laterality normalization
- moderate slices/windows
- frozen encoder
- small target-specific head
- same `PSEUDO_LABELS_v1_1`

Purpose:
test representation quality without allowing the backbone to memorize weak labels.

### Experiment C — selective DINO unfreeze
Unfreeze only final transformer blocks.

Compare:
- pseudo validation
- training stability
- one-time locked gold result

### Experiment D — five group-safe folds
Create OOF predictions and rank-mean.

### Experiment E — dense window inference
Increase deterministic test-time slice coverage; rank-average windows.

### Experiment F — second family
RadImageNet or CoAtNet.

### Experiment G — equal-rank family ensemble
DINO + second family.

### Experiment H — robust pseudo supervision v2
Only after architecture is strong enough to expose supervision bottlenecks.

### Experiment I — submission-only refit
Train frozen final recipe using all allowed labeled training data.

---

# 19. What not to do

Do not:
- tune blend weights repeatedly to the public leaderboard;
- treat 0.940 as a single-model architecture score;
- use the 58 gold cases to choose every experiment;
- silently change the frozen report rules;
- silently change `PSEUDO_LABELS_v1_1`;
- train another expensive ResNet18 attention variant without a specific new hypothesis;
- assume more attention layers solve weak-supervision mismatch;
- discard missing-slot studies;
- make AX_NONFS mandatory;
- copy the public target-specific final blend weights without OOF justification.

---

# 20. Immediate recommendation

Run two tracks in parallel conceptually:

## Leaderboard track
Reproduce the public audited 0.940 inference pipeline.

This establishes a strong practical benchmark.

## Research / innovation track
Build a cleaner, more efficient variant around:
1. semantic fluid/T1 slots;
2. laterality normalization;
3. DINOv3 frozen/partially frozen encoder;
4. dense multi-window features;
5. robust weak supervision;
6. OOF rank ensembling;
7. one diverse second family.

The most promising innovation is **not another attention head**.

It is the combination of:

> better sequence semantics + strong SSL representation + weak-supervision robustness + rank-aware diversity.

---

# 21. Current stopping point

Completed:
- report teacher
- pseudo-label freeze
- MRI baseline v1.1
- corrected baseline v1.2
- six-slot/crop feasibility audit
- six-slot target-attention B2B1 final training and gold evaluation
- visual MRI/report disagreement audit
- public 0.94 notebook structural audit

Next:
**sequence-aware slot v2 audit and public 0.940 reproduction / DINO strong-backbone track.**
