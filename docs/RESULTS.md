# Results: every submission and experiment

RSNA Knee Abnormality Detection (Kaggle, closed 2026-10-22). Metric: mean AUC over 12 findings.
Final standing: no medal. Top private score 0.960; a 499th-place writeup reports bronze at 0.943 private, so our F0 fork (0.942) missed bronze by about 0.001.

Leaderboard scores were pulled with `kaggle competitions submissions` after the competition closed.
"Val" is our weak-label validation (885 studies, labels derived from radiology reports), **not** an expert score.

## 1. All Kaggle submissions

### Our own models (trained from ImageNet weights on competition data only)

| Date | Submission | What changed | Val | Public | Private |
|---|---|---|---:|---:|---:|
| Oct 13 | E01 | ResNet18, 6 epochs, on-the-fly DICOM decoding | 0.822* | 0.769 | 0.735 |
| Oct 13 | E1 | New cached preprocessing (canonical orientation, 160 mm crop, ImageNet norm), 12 epochs | 0.852 | 0.807 | 0.788 |
| Oct 14 | E4+E5 | Severity-aware relabel (labels v3) + soft targets + clipped pos_weight | 0.863 | 0.841 | 0.821 |
| Oct 14 | Ens3 | Rank average: E4+E5, E6, E2 | 0.870 | 0.848 | 0.827 |
| Oct 15 | **Ens4** | Ens3 + E8 (ConvNeXt-Tiny) | **0.880** | **0.857** | **0.837** |

\* E01 val used the older 863-study split, so it is not directly comparable.

**Ens4 is our best model built by us.** See [OUR_BEST_MODEL.md](OUR_BEST_MODEL.md).

### Public-weights track (inference over public Kaggle checkpoints)

| Date | Submission | What changed | Public | Private |
|---|---|---|---:|---:|
| Oct 17 | F0 | Unchanged fork of the public 0.958 stack (CoAtNet Raptor 384/224 + heliosli R6 ConvNeXt-S + goodpjw reader) | 0.958 | 0.942 |
| Oct 18 | F1 | F0 + 0.2 rank weight of our S1-B student (ConvNeXt-S, teacher-denoised labels v3) | 0.954 | 0.938 |
| Oct 20 | F3 | F0 with 50% shrinkage of the per-finding Raptor weights | 0.957 | 0.942 |
| Oct 21 | F5 | F0 without the small reader branch | 0.958 | 0.942 |
| Oct 21 | F6 | F0 with Gaussian-rank blending | 0.958 | 0.942 |
| Oct 22 | F7 | F0 with squared percentile ranks | 0.958 | 0.942 |

None of the F-variants beat the unchanged fork. Adding our own student (F1) **lowered** the score.

## 2. Development experiments (not submitted)

All scores are macro AUC on the 885-study weak-label validation set unless noted. Paired group-bootstrap 95% CIs in brackets.

### V3 own-model line (Oct 13-16)

| ID | Change vs control | Val | Verdict |
|---|---|---:|---|
| E2 | Fixed 12-slice slabs + per-finding attention MIL head | 0.846 (−0.016 vs E4+E5) | Worse alone; kept as a **diverse ensemble member** |
| E6 | Anatomy-safe augmentation + 16 epochs + average of last 3 checkpoints | 0.859 (−0.003) | No gain alone; kept for the ensemble |
| E7 | Input 320 px instead of 224 | 0.851 (−0.012) | Rejected: the cache is 256 px, so 320 only upsamples |
| E8 | ConvNeXt-Tiny (ImageNet-22k) instead of ResNet18 | **0.874 (+0.012 [+0.004, +0.020])** | Best single model at the time |
| E3 | E8 + 6 slots (plane × fat-suppression) | 0.878 (+0.003, CI crosses 0) | Ens5 = Ens4 + E3 → 0.885 |
| N1 | E8 with attention-MIL head, random windows | 0.887 (+0.012 [+0.007, +0.018] vs E8) | Best single; **did not replicate on seed 1 (N6)** |
| N3a | N1 with 6 slots | 0.888 | Blend 0.892 |
| N3b | Transformer context head | worse | Rejected |
| N4 | Native 384 px cache, 320 input | collapsed (~0.55 at epochs 1-2) | Stopped early; cause not found |
| N5 | Differential encoder/head LR | −0.009 [−0.015, −0.003] | Rejected; uniform-LR 16-epoch arm added +0.0026 to the blend |
| N6 | Seed-1 replication of N1 vs E8 on Kaggle | attention − max = −0.002 [−0.007, +0.003] | Attention advantage did not replicate |

The best weak-label ensemble (N1 + N3a + N5, val 0.894) was **never submitted**.

### Public-weights and label track (Oct 17-21)

| ID | What | Result |
|---|---|---|
| K1A/B | Fine-tune Raptor-384 on consensus labels (+ dual view) | Gold-58 0.920 / 0.917 vs parent 0.923; rank corr 0.99 = a clone |
| K2A/B | Fresh EfficientNetV2-S | fp16 overflow / NaN |
| K3A/B | Compartment tiles at 1.8× (SE-ResNeXt50 / EffNetV2-S) | 0.842 Gold-58 / NaN |
| K4A/B | Raptor warm start + compartment tiles | 0.911, blend flat / collapsed at LR 2e-4 |
| T1 | 4-fold ConvNeXt-S teacher; OOF predictions denoise the labels (v3 → teacher-blended) | Label Gold-58 **0.921 vs 0.896** for report consensus |
| S1B | ConvNeXt-S student on teacher-denoised labels | Gold-58 0.906; blend +0.002 offline, but **−0.004 on public LB (F1)** |
| F2 | Frozen DINOv2-S + context/geometry head | 0.799 proxy; geometry alone +0.001 |
| F4 | F2 + pairwise ranking loss / series dropout | +0.009 / −0.003 proxy |

## 3. The 58 expert-labelled studies (gold)

- Opened once, on Oct 13, for E01: model AUC **0.735** (matched its LB 0.769). The public LLM report labels themselves scored **0.878** as predictors, and **0.809** once thresholded at 0.5.
- Verdict at the time: the model was the main problem, and hard labels were a second one.
- Later label work: report consensus 0.896 → teacher-denoised v3 0.921.
- For comparison, the top teams reached 0.942-0.956 label quality by having a vision LLM read the images ([LESSONS_FROM_TOP_SOLUTIONS.md](LESSONS_FROM_TOP_SOLUTIONS.md)).

## 4. Key numbers to remember

| Gap | Size |
|---|---|
| Our best own model vs public stack fork (private) | 0.837 vs 0.942 (**−0.105**) |
| Public fork vs top private score | 0.942 vs 0.960 (−0.018) |
| Our val vs public LB | LB was always 0.02-0.045 **below** val (other teams saw it above) |
| Public vs private, our models | private ≈ public − 0.02 |
