# Prompt: decide the next experiments after F0 and K1 (2026-10-17 evening)

Act as a senior medical-imaging ML researcher who is also an optimization engineer and a Kaggle grandmaster. Give me a **decision**, not a survey. The deadline is 2026-10-22 23:59 UTC, about 5 days away. Resources:
- About 23 Kaggle T4×2 GPU-hours this week; each session can run at most 12 h.
- 4 submissions per day.
- A laptop with an RTX 4050 (6 GB).
- One Kaggle account only.

## Verified facts

**Our leaderboard position.**
- F0 is an unchanged copy of the public stack: nartaa Raptor CoAtNet-384 and -224 (trained with OAI), heliosli R6 ConvNeXt-S (knee-anchored, 4 folds, flip TTA), and goodpjw's reader at 1/10 weight. It scored **public LB 0.958**. Our previous best was 0.857.
- 886 teams are tied at exactly 0.958 (about ranks 144–1029).
  - Silver (about rank 307) and bronze (about 615) both sit inside that tie, so medals are decided below the third decimal, on the private set (70% of about 1,284 studies).
  - 0.959 is about rank 143; gold needs about 0.963 or more; the top score is 0.969.

**K1: two fine-tunes of the Raptor-384 parent.**
- Setup: warm start from the parent, low learning rate, targets = 0.5 consensus of 5 public LLM label tables + 0.5 the parent's own prediction.
- Arm A kept the architecture; arm B added a dual view (global plus 1.5× joint zoom).
- Data: lonnieqin tensors, verified pixel-identical to the inference reader.
- Results:

| | Parent | Arm A | Arm B | Parent + A |
|---|---|---|---|---|
| Rank correlation with the parent | — | 0.986 | 0.988 | — |
| Gold-58 AUC | 0.923 | 0.920 | 0.917 | 0.9215 (gain over parent: 95% CI [−0.005, +0.002]) |

- Per finding: Lateral OA rose from 0.776 to 0.80; Synovitis fell from 0.83 to 0.80.
- Verdict: both arms are clones, and they were not submitted.

**F0 runtime.** We measured this from F0's own log (its run on the 3 visible test studies).
- R6 costs about 38.8 s of decoding and 18.3 s of GPU time per study, with about 529 windows.
- The Raptor models cost seconds per study.
- Free time on the hidden test is estimated at about 2–4 h; it has not been measured.

## Evidence from public notebooks and discussions

Unless marked, the leaderboard deltas below are confirmed against the author's scored versions.

**Measured on a 0.958 base, all flat or negative:**

| Change | Result |
|---|---|
| S6 added at weight 0.3 | 0.957 |
| Per-finding weights set from gold-58 | 0.957 and 0.955 |
| Meniscus specialist | −0.001 |
| One-fold ConvNeXt | −0.003 |

**Positive steps:**
- Better members of the other family: R7, then R6 with flip, each +0.001.
- heliosli's per-finding weights, probed on the LB: +0.001.
- The only positive result from a team's own model (author claim, not verified): a private ConvNeXt-S + EfficientNetV2-S on medial/lateral compartment crops, at rank weight 0.2, took 0.956 to 0.957.

**Inference-only changes:**
- More Raptor windows: 42 → 62 gave +0.003, and 24 → 94 gave +0.006 (claimed). The base already uses 94.
- Mirror TTA: +0.001.
- Narrower crop: −0.001 to −0.002.
- bf16 on a T4: catastrophic, because it hits the time guard.

**Weakest findings on gold-58, Raptor versus R6** (claimed; Raptor's numbers are optimistic because its epochs were selected on gold-58):

| Finding | Raptor | R6 |
|---|---|---|
| Lateral OA | 0.776 | 0.865 |
| Synovitis | 0.812 | 0.859 |
| PF OA | 0.861 | 0.918 |
| Lateral meniscus | 0.894 | 0.916 |
| Fracture | 0.962 | 0.887 |

A hint from a top competitor: gains came on the "classes with external labels" (Lateral OA, PF OA, Synovitis).

**Risks and resources:**
- Some teams report that their reruns on the hidden test "threw an exception". The advice is to commit-test on about 1,300 training studies, with NaN and DICOM guards.
- nartaa published Apache-licensed training code: labels, folds_v0 and the S6fall adapter. No out-of-fold predictions and no fold models were released.
- There has been no host ruling on OAI or external weights since Oct 14.

## Methods research (calculations marked "derived" are my agent's own, not from a paper)

**How much a new member can add (derived; theory: Gontijo-Lopes et al., ICLR 2022; Fort et al., 2019):**
- A new member's gain is roughly proportional to (d_new − ρ·d_ens)² / (1 − ρ²), where ρ is the correlation of its errors with the *blend*.
- For a member scoring 0.94 alone, added to the 0.958 blend:

| Error correlation ρ | Expected gain |
|---|---|
| 0.88 | +0.000 |
| 0.75 | about +0.004 |
| 0.60 | about +0.009 |

**What produces the most decorrelation per GPU-hour, in order:**
1. A different input view or crop, such as compartment or finding-specific ROIs at higher resolution, or a different set of sequences.
2. A different slice aggregator: bi-GRU or transformer instead of attention MIL.
3. A different backbone family trained fresh from ImageNet, with labels-only targets: EfficientNetV2-S, ResNeSt or NFNet rather than another ConvNeXt or CoAtNet.

**Avoid:**
- Seeds, SWA and self-distillation.
- Boosting toward stack errors (it fits label noise).
- Negative-correlation learning (it needs honest out-of-fold stack predictions).
- Distilling from the other family.

**Fast recipe** (estimated 3–5 h, 0.935–0.945 alone):
- DDP on both T4s, fp16, channels_last.
- 16–32 of the 96 slices per study per epoch.
- About 10 epochs at 224–256 px with OneCycle, then 1–2 epochs at 320.

**What the noise can resolve:**
- Gold-58: the SE of a paired difference is about 0.007–0.01. Use it only to veto broken models and to measure ρ.
- Public LB: paired SD about 0.003. The probability that a change is real is about 0.63 at +0.001, 0.75 at +0.002 and 0.84 at +0.003. The best of 10 submissions is inflated by about +0.004.

**MONAI / TorchIO:**
- MRI artifact augmentation is worth about 0 to +0.003 directly (we have no scanner domain shift). It is useful only as a cheap source of diversity.
- MONAI's ensemble tools add nothing over rank averaging.

**Evidence files:**
- `research/external_evidence/deep_research_20261017/` and `.../deep_research_20261017b/`
- `research/K1_F0_PUBLIC_WEIGHTS_TRACK.md`
- `outputs/k1_raptor_ft/`

## Answer through four lenses, briefly, then decide

1. What would a top researcher or optimization engineer do here that I am not asking for or doing?
2. What is the rawest material they would start from? For example: per-study disagreement between members, gold-58 per-finding errors, test-time runtime logs, label-source disagreements.
3. Which specific tools and methods would they use that we have not?
4. Other than you or me, what shows that a result is right? Name the arbiter for each decision and what it cannot tell us.

## Decisions I need

**A. Is a +0.001–0.004 gain worth chasing at all?** Consider that medals are decided inside a tie of 886 teams, on the private set. Estimate the expected effect on private rank of each option against simply keeping F0.

**B. The one new member to train** with about 10 of the 23 GPU-hours. Specify:
- the view or crop and which findings it targets;
- the backbone family and initialisation;
- the slice aggregator;
- the labels;
- the training recipe and time;
- the success metrics: its standalone gold-58 AUC and its ρ with F0 on gold-58, plus a kill criterion before it gets a submission.

Say whether to add the MONAI/TorchIO augmentations, and whether the K1 lessons (shrink-and-perturb, head re-initialisation) make a warm start viable.

**C. Inference-only changes to F0**, such as more TTA or shift crops, which cost no training. Which, if any, are worth a submission, given that the measured ones mostly cost runtime?

**D. Runtime safety.** How do we measure F0's real hidden-test runtime and headroom, and guard the added member (deadline, fallback, a commit test on about 1,300 training studies)?

**E. Blend weight and the final two selections**, with no weight tuning on the LB or on gold-58.

**F. Laptop work** that runs in parallel: preparation, integration dry-runs and label work. Say whether its 6 GB GPU should train anything.

**G. A day-by-day calendar to Oct 22**, with GPU-hours and submissions per day. Include what we stop doing.

## Rules for your answer

- Make one change per experiment.
- Label every number as verified, claimed or estimated.
- If you disagree with a premise above, say so first.
