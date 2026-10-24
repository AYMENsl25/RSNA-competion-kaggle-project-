# Public-weights track: F0 reproduction and K1 Raptor fine-tunes (2026-10-17)

The user approved building on public Kaggle weights and asked for two parallel Kaggle experiments that customise, fine-tune and change the architecture of those weights. Evidence base: `research/external_evidence/deep_research_20261017/`. Our best LB before this track was 0.857.

## F0: reproduce the public 0.958 stack (submission test)

- Kernel `aymenslimani/rsna-knee-f0-fork-0958`, private: an unchanged copy of `goodpjw2008/rsna-knee-coatnet-r6-2-5d-mil-lb-0-958` v27. It uses the same five datasets and the same pinned docker image.
- Members:
  - nartaa Raptor-384 and Raptor-224, both with mirror TTA.
  - heliosli R6, 4 folds, with flip TTA.
  - The goodpjw reader at 1/10 weight.
- Blending: per-finding rank weights.
- Purpose: prove we reproduce 0.958 under our account, and use it as the base for every later submission.
- Compliance: nartaa's weights were trained with OAI (controlled-access). The required acknowledgement is in the notebook text. Prize eligibility is uncertain; medals are not affected.

## K1: two fine-tunes of nartaa's Raptor-384 (LB 0.949 alone), one per T4, in one kernel

Kernel: `aymenslimani/rsna-knee-k1-raptor-ft`. Code: `notebooks/k1_raptor_ft/`.

**Shared inputs**
- **Training tensors:** lonnieqin/rsna-knee-raptor-tensors-s0..s3 (Apache-2.0), 4,407 studies.
  - Geometry: 96 slices in 5 slots, 384 px over 140 mm, stored as the centre 352 px.
  - They were built by the embedded DICOM reader whose decoded source is byte-identical (sha256 5a59b6f8…) to the reader the 0.958 notebook runs at test time. A 320 crop here is therefore the inference input.
  - In-kernel gate: rebuild 3 studies from DICOM and require max|diff| = 0.
- **Parent:** `raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt` (sha 7e5315da…).
  - Trained on all RSNA training studies plus 2,399 OAI knees.
  - Its SWA epochs were selected on gold-58.
- **Targets:**
  - Consensus = mean of the public LLM report-label tables. The kernel uses whichever of these it can read, and needs at least three:
    - stevenleehans v4 blend
    - pilkwang v2
    - lixin73 gpt56sol
    - laymond qwen
    - nartaa v1_blend
  - Final target = 0.5 × consensus + 0.5 × the frozen parent's prediction on the same augmented windows. The parent half keeps what the parent learned from OAI. A missing consensus cell falls back to the parent's prediction.
- **Optimisation:**
  - AdamW: encoder 1.5e-5, head 3e-4, new parameters 1e-3, weight decay 0.05.
  - 30 warm-up steps, then cosine.
  - fp16 with GradScaler (T4 has no bf16).
  - Effective batch of 16 studies, 12 random stratified windows per study, gradient checkpointing.
  - Budget 8 h per arm, at most 6 epochs. The epoch count is fixed from measured speed after 120 studies.

**The two arms**

| Arm | One change vs the other | Deploys as |
|---|---|---|
| A | Architecture unchanged. Augmentation: anatomical mirror (p 0.5), ±12 px crop shift, gain and gamma 0.9–1.1. | An extra arm in the notebook's `ARMS` list. Same class; strict load checked locally. |
| B | Same targets and augmentation, plus a dual view. Each window is seen as the global 320 crop and as a 0.6 centre zoom (211 → 320 px, 1.5× detail on the joint). (view, slot) embeddings start at zero; zoom attention starts at an offset of −2. | Needs a small inference class (`DualView`) in the notebook. |

**Rationale**
- Per-finding gold-58 AUCs for Raptor (claimed by goodpjw) are weakest on:
  - Lateral OA: 0.776
  - Synovitis: 0.812
  - PF OA: 0.861
  - Lateral meniscus: 0.894
- Small joint structures are where resolution should matter.
- Public evidence says only newly trained, different models move a ~0.958 blend.

**What decides**
- There is no clean validation set: the parent saw every training study. Gold-58 AUC per epoch is logged for information only and is biased toward the parent.
- The predeclared checkpoint is the last epoch.
- The public LB decides, against F0.
- A gain is real only above about ±0.003, the noise on roughly 390 public-test studies.

**Local verification before push**
- Both arms ran end to end on the laptop GPU with synthetic tensor shards, the real CoAtNet-2 architecture and a teacher checkpoint.
- An arm-A checkpoint loads strictly into the inference `RaptorClassifier`.
- The consensus-label section ran on the real public tables: 4,349 training studies, 58 gold excluded.

## Results (2026-10-17 evening)

- **F0:** public LB **0.958**. The public stack reproduces under our account.
- **K1 run health:** both arms finished. Parity gate 0/0/0. All five label sources were read. No errors.
- **Arm A:** 6 epochs, 56 min each, 5.9 h in total.
- **Arm B:** 3 epochs, 108 min each.

| | Parent | K1-A ep5 | K1-B ep2 | Parent + A (rank 50/50) |
|---|---|---|---|---|
| Gold-58 macro AUC | 0.9229 | 0.9198 | 0.9172 | 0.9215 |
| Rank correlation with the parent | — | 0.986 | 0.988 | — |

- **Per-finding changes:**
  - Lateral OA rose: 0.776 → 0.799 (A) and 0.795 (B).
  - Synovitis fell: 0.829 → about 0.798.
- **Parent + A versus parent on gold-58:** 95% CI [−0.0049, +0.0021].

**Verdict:** both fine-tunes are near-clones of the parent. The likely cause is warm-starting from a converged SWA model with a low learning rate while 50% of the target was the parent itself, which pinned the students to it. The expected blend gain is about zero, so neither arm is promoted.

**Lesson:** diversity has to come from a model that is not initialised from, or distilled toward, an existing stack member.

## K2: fresh, different members (launched 2026-10-17 night)

Decision source: `research/NEXT_DECISION_PROMPT_2026-10-17b.md` and `research/external_evidence/deep_research_20261017b/`. A new member only helps if its errors decorrelate from the blend: at an error correlation of about 0.75, a 0.94 model adds about +0.004 (derived); at 0.88 it adds nothing.

**Kernel:** `aymenslimani/rsna-knee-k2-fresh-effv2`. Code: `notebooks/k2_fresh/`.

**Shared setup**
- Backbone: EfficientNetV2-S, initialised from ImageNet-21k (`tf_efficientnetv2_s.in21k_ft_in1k`). This family is not in the stack.
- Head: the proven Raptor per-finding attention MIL.
- Targets: consensus labels only. No parent, no distillation. Unknown label cells are masked out of the loss.
- BatchNorm statistics frozen, because each batch is a bag of 4-window chunks from one study.
- Optimisation: encoder LR 2e-4, head LR 1e-3, warm-up then cosine, at most 16 epochs, 9 h budget.
- Data and parity gate: same as K1.

**The one difference between arms: the input view**
- **A:** a 0.6 centre zoom (211 → 320 px), aimed at lateral OA and lateral meniscus.
- **B:** the global 320 view, which is the same input F0's Raptor gets.

**Gates before any submission**
1. Gold-58 macro AUC of at least 0.90. This is a veto for broken models only.
2. Rank correlation with the parent on gold-58 below 0.90.
3. The surviving arm joins F0 as an extra rank member at a fixed weight of 0.2, decided in advance and not tuned.

**Final two selections:** plain F0 plus the best diverse blend, and only if that blend gains at least +0.003 on the LB.

## K3: compartment-tile members with an efficiency recipe (launched in parallel with K2)

**Kernel:** `aymenslimani/rsna-knee-k3-compart-seresnext`. Code: `notebooks/k3_compart/` (K2's trainer plus a patch).

**View:** two square tiles per window, side 0.5·S, centred at x = S/2 ∓ 0.22·S and resized to 320 px (1.8× detail). This is closest to the only own-model gain reported publicly: compartment crops at rank weight 0.2, 0.956 → 0.957 (author claim).

**Arms**
- **A:** SE-ResNeXt-50 (racm ImageNet weights), a new family.
- **B:** EfficientNetV2-S, so that comparing K3-B with K2 isolates the effect of the view.

**Efficiency changes relative to K2**
- No activation checkpointing.
- channels_last memory layout and cuDNN benchmark mode.
- Progressive resizing: the first 60% of epochs at 256 px, the rest at 320.
- EMA of the weights (decay 0.998) for evaluation and saving.
- Gradient accumulation of 8 instead of 16.
- 8 window centres, which gives 16 tiles per study.

**Budget:** 6 h. **Gates:** the same as K2.

## K2 / K3 results (2026-10-18)

**EfficientNetV2-S failed in all three arms (K2-A, K2-B, K3-B).**
- The fp16 forward pass overflowed: the loss became NaN at about optimiser step 37, and the GradScaler scale collapsed to 0.004, so almost every step was skipped.
- K2-A finished but barely learned: gold-58 0.586.
- K2-B and K3-B crashed at gold evaluation on NaN predictions.
- Two fixes for any future run:
  1. A NaN guard that stops an arm early and says so.
  2. A NaN-safe evaluation, so one bad arm cannot lose the others' results.

**K3-A (SE-ResNeXt-50, compartment tiles, EMA) trained cleanly.**
- Gold-58: 0.842 at epoch 13, still rising about +0.002 per epoch when the cosine schedule ended.
- Rank correlation with the parent: 0.81. It is diverse, but too weak to help a blend.

Adding K3-A to the parent at different weights (gold-58):

| Weight on K3-A | 0 | 0.1 | 0.2 | 0.33 |
|---|---|---|---|---|
| Gold-58 macro AUC | 0.9229 | 0.9219 | 0.9202 | 0.9154 |

- Per finding, K3-A beats the parent only on Lateral OA (0.820 vs 0.776) and Effusion (0.986 vs 0.973).

**Break-even (derived, Gaussian model):** at ρ = 0.81, a new member must reach about 0.876 on gold-58 to add anything, and about 0.90 or more to add something measurable. Backbone strength is the binding constraint: dreaddevelopment's CoAtNet-2 recipe, trained from ImageNet without OAI, reached gold 0.917 and LB 0.924.

## Raw material: label sources and the parent model vs gold-58 (2026-10-18)

| Finding | Consensus report labels | Parent model | Reading |
|---|---|---|---|
| ACL / MCL | 0.991 / 0.980 | 0.964 / 0.943 | vision gap: the labels know more than the model learned |
| PF OA | 0.899 | 0.860 | vision gap |
| Lateral OA / Synovitis / Lateral meniscus | 0.820 / 0.798 / 0.880 | 0.776 / 0.829 / 0.896 | both weak: report language vs expert definitions |
| Effusion / Contusion / Fracture | 0.880 / 0.861 / 0.814 | 0.973 / 0.937 / 0.958 | the model already beats its labels |
| Mean | 0.896 | 0.923 | |

Individual label sources (macro): steven 0.893, pilkwang 0.866, lixin 0.835, qwen 0.818.

## K4: warm start from the knee-pretrained encoder, different everything else (launched 2026-10-18)

**Kernel:** `aymenslimani/rsna-knee-k4-warm-compart`. Code: `notebooks/k4_warm_compart/`.

**Design**
- Encoder: nartaa's Raptor-384 encoder, which is domain-pretrained on RSNA plus OAI. This is used instead of self-supervised pretraining, which there is no time for.
- View: compartment tiles (K3 showed ρ 0.81 for this view).
- Head: re-initialised.
- Targets: consensus labels only. No teacher.
- Training: EMA 0.998, NaN guard, 6 window centres (12 tiles per study), accumulation 8, head LR 2e-3, at most 10 epochs, 8.5 h budget.

**Arms (one change):** encoder LR. A = 4e-5 (the original recipe's rate); B = 2e-4 (leave the parent's basin).

**Gates:** gold-58 at least 0.90, rank correlation with the parent below 0.90, then F0 + 0.2 on the LB.

## T1: 4-fold teacher for out-of-fold label denoising (Plan B, launched alongside K4, 2026-10-18)

**Evidence:**
- The method is from repo A-b29/RSNA-Knee-Competition. It has no licence, so only the method is reused, not the code; its README is saved in `external_evidence/deep_research_20261017b/`.
- In that repo, labels = 0.5 · report + 0.5 · mapped teacher OOF raised the labels' gold-58 AUC from 0.903 to 0.927 (effusion 0.88 → 0.964, fracture 0.90 → 0.94). Its students' LB result is still pending.
- Report-text severity rules do not help: strict host-definition cuts lower gold-58 AUC for 6 of the 7 findings checked (severity-label agent report).

**Kernel:** `aymenslimani/rsna-knee-t1-teacher-convnexts`. Code: `notebooks/t1_teacher/`.

**Design**
- Model: ConvNeXt-S from ImageNet-22k, global view, labels only. It uses LayerNorm, so it is fp16-safe.
- Folds: 4, grouped by normalised report text with GroupKFold; gold excluded.
- Training per fold: EMA 0.998, 256 → 320 progressive resizing, no checkpointing.
- Out-of-fold predictions: 48 windows, plain plus mirror.
- Scheduling: GPU 0 runs folds 0 and 2, GPU 1 runs folds 1 and 3; about 1.9 h per fold, 4 h in total.

**Next step on the CPU:** labels v3 = 0.5 · consensus + 0.5 · teacher OOF, quantile-mapped onto each finding's consensus distribution. Its gold-58 score uses the mean of the 4 fold models on the gold studies, which none of them trained on. The gate is labels v3 above 0.896 on gold; only then do students train (Raptor encoder warm start + compartment view).

**Quota:** the K4 and T1 sessions run concurrently and both count against it, leaving about 4 h for students.

## T1 results (2026-10-18): the label gate passed

- **Run:** `rsna-knee-t1-teacher-convnexts-2` (the user's re-run). 4 folds of 1,087–1,088 studies, 8 epochs each, 1.4 h per fold, 12.5 min per epoch. No errors. All 4,349 out-of-fold predictions were written.
- **Teacher gold-58 by fold** (each trained on 75% of the data): 0.901, 0.886, 0.890, 0.893.
- **Teacher mean of the 4 folds:** 0.904 on gold, a fully honest held-out score. Rank correlation with the Raptor parent on gold: 0.903.

| Gold-58 | Teacher (mean of 4) | Consensus report labels | Labels v3 |
|---|---|---|---|
| ACL / MCL | 0.942 / 0.864 | 0.991 / 0.980 | 0.975 / 0.959 |
| Lateral meniscus / Lateral OA | 0.893 / 0.868 | 0.880 / 0.820 | 0.920 / 0.859 |
| PF OA / Synovitis | 0.804 / 0.720 | 0.899 / 0.798 | 0.877 / 0.782 |
| Effusion / Contusion / Fracture | 0.955 / 0.945 / 0.940 | 0.880 / 0.861 / 0.814 | 0.965 / 0.908 / 0.884 |
| **Mean** | **0.904** | **0.896** | **0.921** |

- **Labels v3 minus consensus**, paired bootstrap over studies (2,000 draws): **+0.0249, 95% CI [+0.0087, +0.0429]**. Labels v3 is in `outputs/t1_teacher/labels_v3.json`.
- The image teacher and the reports are complementary:
  - the reports are better on ACL, MCL, PF OA and synovitis;
  - the teacher is better on effusion, contusion, fracture and the menisci.

## K4 results (2026-10-18)

- **K4-A** (encoder LR 4e-5): gold-58 peaked at 0.915 (epoch 5) and ended at 0.911. Rank correlation with the parent: 0.923. Parent + 0.2 × K4-A scores 0.9230 against the parent's 0.9229, which is flat. It is not diverse enough.
- **K4-B** (encoder LR 2e-4): **collapsed**. Gold-58 fell from 0.52 to 0.49, which is chance level. The CoAtNet encoder breaks at this rate.
- **Risk to S1-A:** it uses 1e-4, which lies between the rate that works (4e-5) and the one that collapsed (2e-4). S1-B (fresh ConvNeXt-S on labels v3) is the safety arm.

## F1 submission prepared (2026-10-18): F0 + one S1 student

Code: `notebooks/f1_fork_plus_student/`. `build.py <A|B> [weight]` produces the unchanged F0 notebook plus a Part C cell. The kernel source is S1's output.

**What Part C does**
- Runs `infer_c.py` in its own process, using the embedded Raptor reader. The volume is cut to the centre 352 px, the same input as training: 48 windows, plain plus mirror.
- Has a deadline 8.6 h after the notebook starts, and is skipped if more than 7.6 h have already been used.
- Backs up the F0 submission first. It restores F0 on any failure, NaN, constant column or incomplete coverage.
- Blends as a rank: (1 − w) · F0 + w · student, with w = 0.2 fixed.

**Checks passed locally**
- Parity between `infer_c.predict` and the trainer's own gold evaluation, for global, compartment and CoAtNet-compartment checkpoints: max |diff| from 6e-5 to 8e-4 (fp16 noise).
- Fallback: with no student checkpoint attached, the cell restores F0 byte-identical.

**Submission rule:** only an S1 arm that passes both gates is submitted (gold-58 at least 0.90, and ρ with the parent below 0.90).

## S1 results (2026-10-18)

- **S1-A** (Raptor encoder, encoder LR 1e-4, compartment view, labels v3): the fp16 loss went non-finite in epoch 0. The NaN guard stopped it after 50 bad steps (scale 3.7e-9). The warm-started CoAtNet is unstable at encoder LR of 1e-4 and above.
- **S1-B** (fresh ConvNeXt-S, labels v3, full data, 10 epochs, 1.9 h): gold-58 **0.906**, with a best of 0.908. Rank correlation with the parent **0.891**. **It passes both gates.** Parent + 0.2 × S1-B scores **0.9247** against 0.9229 for the parent (+0.0018). It is the first member to improve the blend on gold.
- **F1** = F0 + 0.2 × S1-B, pushed as `aymenslimani/rsna-knee-f1-f0-plus-student` for its commit run.

**F1 commit run:** passed. Parts A and B ran with status PASS and 0 fallbacks. Part C scored 3/3 studies with 0 NaN in 9 s and blended at 0.2; mean rank correlation with F0 was 0.750 (3 studies only). **Submitted 2026-10-18 19:16 UTC as ref 57366237; pending.**
