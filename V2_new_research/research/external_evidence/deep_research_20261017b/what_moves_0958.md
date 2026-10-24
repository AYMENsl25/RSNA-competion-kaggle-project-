# What moves the public LB above a 0.954-0.958 base (snapshot 2026-10-17 ~19:00 UTC)

Read-only. Nothing was submitted, pushed, run or uploaded.

**Sources**
- Threads were re-fetched through the internal API: `kgfetch.py`, written to `scratchpad/kg/*.md`. Older copies are in `kg_prev/`.
- Notebooks were pulled at their latest versions into `scratchpad/nb2/`. Markdown extracts are `scratchpad/nb2_*.md`.
- Version histories came from `kver.py` (`kernels.KernelsService/ListKernelVersions`), output in `scratchpad/kver_out.txt`.
- The full public LB was downloaded to `scratchpad/lb/`.

**Tags**
- **[V]** verified from code, the leaderboard, or the author's own scored-version text with a submission id.
- **[C]** the author's claim only.
- **[E]** my estimate.

## 0. Leaderboard reality (full public LB csv, 18:51 UTC today) [V]

| Score | Teams | Cumulative rank |
|---|---|---|
| 0.969 | 1 (RB & Matheus & Tucker & Tony) | 1 |
| 0.968 | 3 (azamatt, Sergey Bryansky, CentralyDS) | 4 |
| 0.967 | 1 | 5 |
| 0.966 | 4 | 9 |
| 0.965 | 4 | 13 |
| 0.964 | 7 | 20 |
| 0.963 | 10 | 30 |
| 0.962 | 17 | 47 |
| 0.961 | 19 | 66 |
| 0.960 | 29 | 95 |
| 0.959 | 48 | 143 |
| **0.958** | **886** | **1029** |
| 0.957 | 225 | 1254 |
| 0.956 | 435 | 1689 |

- 6,150 teams in total. Every 0.958 fork, ours included, sits in an 886-team tie block.
- Approximate medal lines: silver is the top 5% (about 307) and bronze the top 10% (about 615). Both fall inside the 0.958 block, so rank there is decided below the third decimal.
- 0.959 is rank 143 or better. Gold needs about 0.963 or more.
- Since about Oct 15 the top has moved: 0.967 → 0.969, and the count at 0.963+ went from 18 to 30. Top teams are still improving.

## 1. Documented deltas on top of a 0.954–0.958 base

| # | Base → result | What changed | Source | Tag |
|---|---|---|---|---|
| 1 | 0.954 → 0.956 | Swap S6 for **R7** (4-fold ConvNeXt-S, fastMRI aux) next to Raptor-384, 50/50 rank | picond r7-pinned (submission 57143234 = 0.956; S6 version 0.954); hoshinoyumemi #11 | V |
| 2 | 0.956 → 0.956 (rank 233 → 156) | + **Raptor-224** checkpoint (same OAI recipe, 0.8 crop, single view) at 35% of the Raptor half, i.e. 17.5% overall | varshithakolupuri "one unused checkpoint"; hoshinoyumemi #12, ranked above #11 by auto-select | V (display), C (rank) |
| 3 | 0.956 → 0.956 (rank 357 → 289) | + **mirror TTA** on the 224 model | varshithakolupuri "one free flip" v1 | C |
| 4 | 0.956 → 0.957 | **R6 (bundle fe055406) instead of R7**, + flip TTA, 50/50 | goodpjw table; picond 9576 | V (0.957 display) |
| 5 | "0.9573" → "0.9576" | R6 without flip → R6 with flip (both in the Raptor 384+224 blend) | heliosli kernel titles v1/v2 vs 0.9586-v1 | C (4th decimal not visible publicly) |
| 6 | 0.957 → 0.958 | **Per-finding Raptor weight** (Raptor share): Lat OA 0.2, PF OA 0.6, Synovitis 0.8, Baker's 0.35, rest 0.5 | heliosli blend-mine; goodpjw v27 | V (goodpjw table) |
| 7 | 0.958 → "0.9586" | + goodpjw 2.5D reader at 0.003–0.035 per finding; 224 without mirror; heliosli weights (Lat OA 0.15, PF OA 0.65, Syn 0.85, Baker's 0.30) | heliosli 0.9586 v7 (Oct 17) | C (4th decimal) |
| 8 | 0.958 → **0.957** (−0.001) | + **S6** (heliosli compact-v1) as a second compact model at 0.3 next to R6 | goodpjw does-S6 v4/v5 | V |
| 9 | 0.958 → **0.957** (−0.001) | Per-finding weights moved on 3 findings by **gold-58** gaps (Fracture 0.8, Effusion 0.7, ACL 0.3) | goodpjw does-S6 **v6** | V |
| 10 | 0.958 → **0.955** (−0.003) | All 12 per-finding weights set to the gold-58 optimum, clipped to 0.1–0.9 | goodpjw does-S6 **v7** (Oct 17) | V |
| 11 | 0.956 → 0.957 | + **bliverrigpdoge private "student"**: knee-centred crops (medial and lateral compartments from sag+cor, plus axial), small ConvNeXt-S + EffNetV2-S ensemble trained on dreaddevelopment soft report labels; 0.8 base / 0.2 own per-finding rank. Base = R7 + Raptor-384 | bliverrigpdoge v1 (Oct 14); weights private, so a fork scores 0.956 | C |
| 12 | 0.956 → 0.956, lower unrounded | + goodpjw reader with per-finding weights on the R7+Raptor base | hoshinoyumemi #13 (auto-select ranked it below #11/#12) | C |
| 13 | 0.956 → 0.954 | goodpjw reader at full weight | goodpjw (previous report) | C |
| 14 | 0.957 (leoprovorov) | R384 + R224 (mirror) + R7 + **nartaa S6fall** (0.6 R7 / 0.4 S6 in the compact half), 50/50. Rank 206 on Oct 15, versus 0.956 for the base recipe | leoprovorov v12 | C |
| 15 | 0.950 → 0.954 | nartaa S6fall-style S6 added to Raptor-384 + reader | goodpjw v8 title history | V |

Older-generation steps (hoshinoyumemi ledger, 14 submissions, [V] with submission refs):
- **+0.002** from per-finding outer weights (0.939→0.941).
- **+0.005** when the new nartaa CoAtNet-384 replaced a 20-model stack (0.945→0.950).
- **+0.006** from R7 (0.950→0.956).
- Negatives: a meniscus specialist −0.001; a public one-fold ConvNeXt M448 at 20% −0.003.
- Author's conclusion: "every visible step up came from a newly published trained model, not re-weighting".

**Pattern** [E]:
- Every gain at 0.954+ came from a **different trained model family or crop policy**: R7/R6 knee-anchored ConvNeXt, or bliverrigpdoge's knee-centred student.
- Same-family additions were neutral or negative: S6 next to R6, the 224 sibling, the weak reader, and our self-distilled Raptor (ρ 0.986).
- Per-finding weights fitted on gold-58 **lose** 0.001–0.003. Weights probed on the LB gained +0.001 (heliosli).
- The only documented +0.001 from an own model is bliverrigpdoge: knee-centred compartment crops, CNN, 20% weight, on a 0.956 base. On a 0.958 base, expect +0.000 to +0.001 at best [E].

## 2. Inference-only levers and their measured LB effect

| Lever | Effect | Source / tag |
|---|---|---|
| Anatomical mirror TTA on Raptor-384 | +0.001 | nartaa 746792 [C] |
| Mirror on Raptor-224 | Display unchanged; rank +68 | varshitha [C] |
| R6 flip TTA (logit mean, medial/lateral swapped back, MCL from the plain pass) | About +0.0003 | heliosli 0.9573 → 0.9576 [C] |
| More windows: Raptor 42 → 62 | +0.003 | 737696 / 746792 [C] |
| More windows: 94 vs 24 | 0.940 vs 0.934 | 737696 / 746792 [C] |
| More windows: 48 → 72 (another team) | 0.944, cost +8 min | 737696 [C] |
| Narrower Raptor crop | −0.001 to −0.002 | nartaa [C] |
| Per-finding weights, LB-derived | +0.001 | heliosli [V] |
| Per-finding weights, gold-58 | −0.001 / −0.003 | goodpjw v6/v7 [V] |
| fp16 vs fp32 | No LB comparison found | — |
| bf16 on T4 | Catastrophic: emulated, slow, hits the time guard, 0.851 → 0.710 | 744230 [C] |
| More folds | No isolated measurement (R6/R7 = 4 folds, reader 3/5) | — |
| Cross-finding post-processing (e.g. Synovitis += Effusion) | −0.020 macro on split-half gold over 500 splits | hoshinoyumemi [C] |
| rabari9999 "temperature / cross-prior micro-boosts" (+0.0008 each, shown to 4 decimals) | Not credible: LB shows 3 decimals; ignore | — |

**Untested in public**
- Mirror/flip TTA for any member not yet using it. All four 0.958 members already use it except the 224 model in the heliosli variant.
- Extra crop shifts.
- More R6 windows.

## 3. Runtime of the 0.958 stack on the hidden test

No author states the hidden-test wall time of the full stack. [V that it is absent]

Indirect facts:
- nartaa (746792): submit-to-score turnaround **26.1 min** for Raptor-384 with mirror, and **7.2 min** for the 224 single view. Both include queueing. [C]
- R6 is the long pole. The earlier estimate is about 20 s of GPU per study per T4, which gives **3.5–4 h** for 1,284 studies on 2×T4. [E, from the 0.9586 commit log]
- Guards in the code [V]:
  - Raptor stages time out at 2–2.5 h.
  - The R6 deadline is 8.3 h from start.
  - goodpjw Part B is skipped after 8.2 h.
  - bliverrigpdoge caps its own stage at min(4 h, 8.5 h − elapsed).
  - nartaa S6fall solo uses an 8 h hang guard.
- Public-run uptimes (209–640 s for 3 studies) say nothing about the hidden test.
- **Best available measurement:** our own F0 submission 57309738, submitted 11:44:15 UTC today. Its submitted → scored interval on the Submissions page is the stack's real runtime plus queue. The API exposes no completion time, so read it in the UI.
- Free budget [E]: about 3–4 h on 2×T4. Budget an extra member at ≤2 h with its own deadline and a fallback to the base file.

## 4. Last 48 h (Oct 15–17)

**Host** [V]
- **No host post since Oct 14.** Seven new user threads (750236, 750257–750260, 749858) ask the host for yes/no rulings on:
  - nartaa OAI weights;
  - R7 fastMRI;
  - the SKM-TEA/KneeXNet `anatomy.pt`;
  - RadImageNet;
  - whether medals are compliance-checked.
- All are unanswered. Four were downvoted to between −22 and −25.
- Submission problems:
  - "Notebook threw exception" reports on the hidden rerun (749623, 749798). Chris Deotte's advice: commit-run on 1,300 train studies and guard against NaN and DICOM failures.
  - Queue delays of over an hour today (750318).

**New assets**
- nartaa / Danial Zakaria (750180, Oct 17) published `training_code/` under Apache-2.0 [C, author]:
  - S6fall: the exact training adapter, `labels_v1_blendjev` (parquet plus build script), `folds_v0.json` (sha `12bebc89…`), and `pool_ok.json` (4,321 studies).
  - CoAtNet 384/224 training (`train_knee.py` and `ext_trainer.py`): OAI rows masked to PF OA, Lat OA and Synovitis; SWA = top-3 epochs ranked on gold-58.
  - S6 is epoch 16 with nothing selected on gold.
  - **No OOF predictions and no fold models were published.** The OAI manifest is not public.
- omerzalman "97.13% accuracy" 11-label soft-label dataset (750295): 60% coverage, accuracy (not AUC) on gold. [C, unvetted, downvoted]
- Adel Zebiche (749426): DINOv2 ViT-S, 1 fold, 224 px, own labels, no external data. **0.952 LB, 0.934 gold-58**, best blend 0.957. Looking for teammates; team merging closed Oct 15. [C]

**Notebooks**
- No new 0.959+ public notebook. Publishing is closed (Oct 15 deadline); only updates to existing notebooks are possible.
- Latest updates:
  - goodpjw does-S6 v8: two negatives.
  - heliosli 0.9586 v7: retitled, same models.
  - leoprovorov v12.
  - lavinwins "top-1-gold": a from-scratch single-series 16-channel trainer, no score.

**0.960+ teams**
- Silent apart from earlier hints: azamat "labels, AI not by hand, don't rely on reports alone" (0.959 single architecture, no external data); a "basically a single model" 0.960 single submission (748111).

## 5. Per-finding weaknesses of the 0.958 stack

Gold-58 AUC, Raptor half vs R6 flip (goodpjw; neither trained on gold, but Raptor SWA was **selected** on gold, so its numbers are optimistic) [C]:

| Finding | Positives | Raptor | R6 | LB-tuned Raptor weight |
|---|---|---|---|---|
| ACL | 24 | .969 | **.996** | 0.5 |
| MCL | 9 | .955 | .964 | 0.5 |
| Medial Meniscus | 26 | **.978** | .971 | 0.5 |
| Lateral Meniscus | 23 | .894 | .916 | 0.5 |
| Medial OA | 15 | .981 | .991 | 0.5 |
| **Lateral OA** | 11 | **.776** | .865 | 0.15–0.2 |
| **PF OA** | 21 | .861 | .918 | 0.6–0.65 |
| Effusion | 35 | **.983** | .945 | 0.5 |
| **Synovitis** | 27 | .812 | .859 | 0.8–0.85 |
| Baker's | 12 | .973 | .969 | 0.3–0.35 |
| Contusion | 19 | .941 | .960 | 0.5 |
| **Fracture** | 18 | **.962** | .887 | 0.5 |

**Weakest findings for both halves:** Lateral OA, Synovitis, PF OA, Lateral Meniscus. Fracture is weak for R6.
- The LB-probed weights put Synovitis at 0.8 Raptor even though gold says R6 is better. This is consistent with the **OAI Synovitis/PF OA supervision in Raptor** helping on the hidden test [E].
- Report-label ceilings (747540) [C]: Synovitis 0.79, Lateral OA 0.83, Contusion 0.86, Fracture 0.88, Effusion 0.88, Lateral Meniscus 0.88.
- hoshinoyumemi: 87% of Synovitis targets are imputed from Effusion [C].

**Top teams' named findings**
- No 0.960+ team has said which findings they improved.
- hengck23: "some classes have more error … these classes have external data with external labels". This matches the OAI-supervised trio Lat OA / PF OA / Synovitis [C].

## Decision-relevant bottom line [E]

1. Re-weighting, gold-58-fitted weights, S6, and same-family siblings are all measured at 0 or negative on this base. Do not spend submissions on them.
2. The only public positive above 0.956 from an own model is a **knee-centred compartment-crop CNN student at 20% rank weight** (+0.001). An own member would need to be a different view of the knee (local crops) and strong on Lat OA, PF OA, Synovitis, Lat Meniscus or Fracture.
3. The public LB has a standard deviation of about 0.003 at this level, and 886 teams share 0.958. A +0.001 display gain may not survive the private 70%. Keep the plain 0.958 as one of the final selections.
