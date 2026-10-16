# Audit of the two supplied ~0.94 notebooks

## Files

1. `cohezion-rsna-knee-sota-ensemble.ipynb`
   - SHA-256: `5daf4ef876b94692b98ec5f54395e125d4c8456cfea29635b2c03f3f8bd743fb`
2. `the-bee-s-knees-final-rsna-push.ipynb`
   - SHA-256: `48679822b130cd8882b608123814b417bc5690596341d23efede0c5cf86bb1c8`

## Main conclusion

These are **not two independent 0.94 solutions**.

The Bee notebook states that it faithfully reproduces Mattia Angeli's public V27 and says that source version displayed **0.940**.

I compared the corresponding principal source-code stages in the two uploaded notebook files. All 20 compared main stages were exact source matches.

The Bee notebook adds:
- explanatory markdown;
- stage snapshots;
- source hash checks;
- prediction validity guards;
- final reconstruction checks;
- licenses and pinned input documentation.

It does not provide a genuinely independent second architecture.

## Exact-match stage audit

| stage                            |   cohezion_cell |   bee_cell | exact_match   |   similarity_ratio |   chars |
|:---------------------------------|----------------:|-----------:|:--------------|-------------------:|--------:|
| DINO setup                       |               4 |          4 | True          |                  1 |    4013 |
| DINO input discovery/cache       |               5 |          5 | True          |                  1 |    2443 |
| DINO headers/laterality          |               6 |          6 | True          |                  1 |    5360 |
| DINO slot selection              |               7 |          7 | True          |                  1 |     739 |
| DINO slice ordering              |               8 |          8 | True          |                  1 |    4758 |
| DINO laterality normalization    |               9 |          9 | True          |                  1 |     197 |
| DINO cache build                 |              10 |         10 | True          |                  1 |    3669 |
| DINO target slot head            |              11 |         11 | True          |                  1 |    1280 |
| DINO model wrapper               |              12 |         12 | True          |                  1 |    1169 |
| DINO backbone builder            |              13 |         13 | True          |                  1 |     941 |
| DINO public member loader/fusion |              14 |         14 | True          |                  1 |   18537 |
| DINO submission writer           |              18 |         15 | True          |                  1 |    1378 |
| DINO main                        |              19 |         17 | True          |                  1 |      18 |
| A5 setup                         |              20 |         20 | True          |                  1 |    1623 |
| A5 DICOM reader                  |              21 |         21 | True          |                  1 |    4149 |
| A5 architecture                  |              22 |         22 | True          |                  1 |   17226 |
| A5 inference                     |              23 |         23 | True          |                  1 |    4594 |
| A5 blend                         |              24 |         24 | True          |                  1 |     738 |
| RadImageNet block                |              25 |         28 | True          |                  1 |   33673 |
| CoAtNet/Raptor/final fusion      |              26 |         32 | True          |                  1 |   23800 |

## Score provenance

The Bee notebook explicitly supports this statement:
- the source V27 displayed 0.940 on Kaggle.

The Cohezion notebook file itself does not contain an independently verifiable leaderboard-score statement in its notebook text/output. It is the same core source family and prediction graph.

Therefore the correct interpretation is:

> We have one highly detailed public ~0.94 ensemble recipe represented in two notebook forms, not two independent ~0.94 discoveries.

## Public recipe architecture

### Stage 1 — DINOv2 member bank

- 20 trained MRI members
- DINOv2 small backbone
- 12 transformer blocks
- saved configuration exposes last 6 blocks as trainable
- 130-mm crop
- 336-pixel cache/input
- geometry-aware ordering
- explicit laterality normalization
- overlapping windows
- target-specific slot attention
- percentile/rank combination

Recovered slot scheme:
- SAG_FLUID_FS
- COR_FLUID_FS
- AX_FLUID_FS
- SAG_FLUID_NOFS
- COR_T1
- SAG_T1

Alternative public slot scheme:
- SAG_FLUID
- COR_FLUID
- AX_FLUID
- SAG_STRUCT
- COR_STRUCT
- AX_STRUCT

This is a major clue: the public high-score recipe uses **sequence semantics**, not only plane × FS.

The DINO head also includes a soft target→slot prior table with strength 0.55.

### Stage 2 — A5 DINOv3

- 5 fold checkpoints: `m_f0.pt` … `m_f4.pt`
- `vit_small_patch16_dinov3.lvd1689m`
- pool `xcodex`
- input norm `none` in the saved configuration
- 6 plane×FS slots
- 130-mm crop
- 336×336
- 16 slices
- slice band 0.12–0.88
- rank averaging
- A5 blend weight: **0.45**

### Stage 3 — RadImageNet

- RadImageNet ResNet50
- 224-pixel layouts
- multiple released head families
- E13 member inside Rad block: **0.50**
- E10 baseline/Rad vote: **0.50**
- second E13 pass on E11 layout: **0.15**
- calibration blend: **0.40**

These are recipe constants, not automatically portable hyperparameters.

### Stage 4 — CoAtNet / Raptor

The notebook's own explanation gives:
- four Raptor views weighted **60% / 10% / 10% / 20%**
- CoAtNet RMLP-2
- adjacent-slice / dense-study reading
- 384 model resolution
- 64 or 44 slice layouts
- broad study span
- one reversed-order view

Residual CoAt checkpoints:
- epoch 4
- epoch 6
- epoch 8

Residual branch contribution inside CoAt/Raptor:
- **40%**

Final branch blend:
- **60% CoAt** for most findings
- **100% CoAt for Lateral Meniscus**
- final percentile rank

## Reproducibility / safety engineering worth copying

Very useful engineering ideas:
- checkpoint fingerprints;
- SHA-pinned weights;
- exact submission-schema checks;
- non-finite prediction guards before ranking;
- deterministic rank transformations;
- stage snapshots;
- independent reconstruction of final fusion;
- laterality tag + geometry fallback;
- DICOM geometry ordering;
- explicit pinned-input manifest.

These are strong practices even if we do not copy the modeling recipe.

## What should be copied into our own research

High-confidence transferable ideas:
1. semantic fluid/T1 slots;
2. laterality normalization;
3. strong SSL ViT features;
4. dense multi-window slice coverage;
5. target-specific evidence pooling;
6. fold/rank averaging;
7. genuinely diverse backbone families;
8. checkpoint fingerprints and stage receipts.

## What should not be copied blindly

Do not blindly copy:
- 0.45 A5 blend;
- 0.50/0.15 Rad coefficients;
- 0.40 calibration weight;
- Raptor 0.60/0.10/0.10/0.20 weights;
- 0.40 residual CoAt weight;
- 0.60 final CoAt weight;
- 1.00 Lateral Meniscus CoAt weight.

Those may reflect source-specific OOF work or public-leaderboard iteration. They need their own provenance.

## Why these notebooks help despite not being independent

They reveal that our B2B1 experiment omitted several high-impact ingredients:
- sequence-aware rather than plane×FS slots;
- laterality normalization;
- much denser study coverage;
- SSL transformer representations;
- patch-level/focal evidence;
- multiple folds;
- multiple families;
- repeated ranking.

The correct next move is therefore **not** "add more attention to ResNet18."

The correct next move is to upgrade the **representation + sequence semantics + supervision robustness + ensemble diversity**.
