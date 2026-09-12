# Next Experiments and Innovation Plan

## North-star objective

Beat the public ~0.94 reference by finding improvements that are:
- reproducible;
- not dependent on leaderboard micro-tuning;
- computationally feasible on Kaggle;
- robust to report-derived pseudo-label noise.

## Two-track strategy

### Track A — practical leaderboard reference

Reproduce the audited public V27/Bee 0.940 pipeline exactly.

Purpose:
- verify environment and inputs;
- obtain a strong benchmark;
- understand runtime and failure modes;
- create intermediate predictions for diversity analysis.

This is a benchmark, not a clean independent discovery.

### Track B — research innovation

Develop our own model under clean group-safe validation.

---

## Innovation 1 — semantic sequence slots

Current weak point:
our `AX_NONFS` slot exists in only ~19% of studies and “non-FS” mixes sequence semantics.

Proposed slots:
- SAG_FLUID_FS
- COR_FLUID_FS
- AX_FLUID_FS
- SAG_FLUID_NOFS
- COR_T1
- SAG_T1

Expected benefit:
- less heterogeneity inside a slot;
- more consistent anatomy/contrast relationship;
- better compatibility with public DINO recipe;
- fewer sparse meaningless slots.

First step:
audit coverage before training.

---

## Innovation 2 — laterality canonicalization

Canonicalize right knees using DICOM geometry/tag logic.

For coronal/axial planes:
flip the appropriate spatial axis.

For sagittal:
use the source-compatible rule only after verifying our orientation convention.

Expected benefit:
reduce mirror-image nuisance variation.

Cost:
low.

---

## Innovation 3 — frozen DINO features before fine-tuning

B2B1 shows pseudo-label overfit risk.

Experiment:
1. DINOv3 ViT-S/16, frozen;
2. extract per-window features;
3. train only a compact study head;
4. compare to v1.2;
5. unfreeze final blocks only if justified.

Why:
strong pretrained representation with low weak-label capacity may generalize better than training many ResNet/head degrees of freedom.

---

## Innovation 4 — patch/focal evidence pooling

Use:
- CLS;
- patch mean;
- top-k/focal patch summary.

Targets with small focal abnormalities should especially benefit:
- meniscus;
- focal OA/cartilage;
- contusion;
- fracture.

---

## Innovation 5 — anatomy-informed soft priors

Add a small fixed bias to target→slot attention logits.

Do not hard route.

Potential prior examples:
- ACL → sagittal + coronal structural/fluid
- MCL → coronal
- PF OA → axial + sagittal
- Baker's → sagittal
- effusion/synovitis → fluid-sensitive sequences

Freeze prior strength before gold evaluation.

---

## Innovation 6 — dense window inference

Train with manageable window count.

At inference:
- generate many deterministic windows across the useful span;
- predict each;
- rank-average or mean then rank.

This directly attacks our sparse 8-anchor limitation without making training infeasible.

---

## Innovation 7 — teacher-disagreement-aware weak supervision

Create a **new version**, never mutate v1.1 silently.

Potential v2 logic:
- preserve rule-positive/negative high-confidence anchors;
- for E5-abstain targets, downweight examples with unstable cross-fold student predictions;
- use ensemble consensus only as an uncertainty estimator;
- optionally bootstrap low-confidence targets.

Avoid self-confirmation:
student output should not simply replace teacher labels.

---

## Innovation 8 — pairwise/ranking auxiliary loss

Because macro ROC-AUC depends on ordering, consider a small pairwise ranking loss on high-confidence pseudo-label pairs.

Do not replace BCE initially.

Candidate:
`L = BCE_macro + lambda * pairwise_rank_loss`

Use only pairs with sufficiently separated frozen teacher targets.

Risk:
ranking losses can amplify pseudo-label mistakes.

---

## Innovation 9 — OOF rank stack

For each family:
- five group-safe folds;
- OOF predictions;
- percentile ranks per target;
- equal rank mean across folds.

Then compare family rank correlations.

A low standalone AUC family can still help if it is genuinely complementary.

The B2B1 checkpoint should be retained for this reason, although it should not be the primary model.

---

## Innovation 10 — diverse second family

Best candidates:
1. RadImageNet ResNet50 target-attention reader;
2. CoAtNet dense 2.5D reader.

Only add after the strong DINO family exists.

---

## Innovation 11 — target-group specialization without leaderboard tuning

Instead of per-target public-LB weights, define anatomically justified model groups before evaluation:

- ligament group;
- meniscus group;
- OA/cartilage group;
- inflammatory/fluid group;
- trauma group.

Train shared encoder with group-specific heads or query sets.

Any group-specific ensemble rule must be selected through OOF, not gold/public LB.

---

## Innovation 12 — submission-only use of official gold

After the entire recipe is frozen:
- architecture;
- supervision version;
- epochs;
- folds;
- ensemble rules;

run a final submission-only training job that includes the 58 official gold labels with stronger supervised weight.

Do not reevaluate this model on the same 58 and call that clean validation.

---

# Prioritized run order

1. Public 0.940 reproduction.
2. Semantic-slot v2 audit.
3. Laterality-normalized frozen DINO baseline.
4. Partial-unfreeze DINO.
5. Five-fold DINO OOF/rank mean.
6. Dense window inference.
7. RadImageNet or CoAtNet second family.
8. Equal-rank family ensemble.
9. Weak-supervision v2.
10. Submission-only gold refit.

---

# Stop gates

## Semantic slot audit
Stop if new slots cause unacceptable missingness.

## Frozen DINO
Stop if it cannot at least approach canonical v1.2 on clean gold after frozen selection.

## Partial unfreeze
Stop if pseudo loss improves while clean gold collapses again.

## Second family
Keep only if it adds rank diversity or target-specific complementarity.

## Ensemble
Do not add complexity without OOF evidence.
