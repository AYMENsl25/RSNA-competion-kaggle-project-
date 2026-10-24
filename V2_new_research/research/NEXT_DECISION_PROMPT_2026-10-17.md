# Prompt: decide the next experiments (2026-10-17)

Act as a senior medical-imaging AI researcher who is also a Kaggle grandmaster. Give me a **decision** for the next experiments in the RSNA knee competition, not a survey of options. The final deadline is 2026-10-22 23:59 UTC, about 5.5 days away. Kaggle GPU quota has reset (~30 h this week). Hardware: laptop RTX 4050 6 GB. One Kaggle account only. Check the daily submission limit before planning.

## Verified facts (2026-10-17)

**Our leaderboard history.** In each pair, validation is our weak-label (report-derived) validation score and LB is the public leaderboard score:

| Validation | LB |
| --- | --- |
| 0.852 | 0.807 |
| 0.863 | 0.841 |
| 0.870 | 0.848 |
| 0.880 | 0.857 (best) |

- Our LB is always 0.02–0.045 **below** our validation. Forum reports say the LB is normally 0.02–0.05 **above** validation on report labels.
- Train/inference parity is ≤3e-7, which rules out a train-vs-inference mismatch. It does not rule out a problem in preprocessing, labels or validation design that both sides share.

**Built but never submitted:**
- The N1+N3a+N5 attention ensemble, weak-label validation 0.8944.
- The E12 blend: Ens4 0.30 + Raptor maxspan 0.35 + goodpjw reader 0.35. It passed GPU smoke tests.

**Latest controlled results** (fixed endpoints, paired group bootstrap on 885 validation studies):

| Run | Result |
| --- | --- |
| N3b transformer head | Worse. |
| N4 native 384 | Collapsed to public AUC ~0.55 at epochs 1–2; cause unknown. |
| N5 differential LR | −0.0092, CI [−0.015, −0.003]. |
| N5 uniform16 added to the ensemble | +0.0026, CI [+0.0014, +0.0039]. |
| N6 | Attention's advantage did not replicate on seed1. |

**Public leaderboard.**
- Top is 0.969; about 840 teams sit at 0.956.
- Every public notebook at 0.956–0.9586 is a re-blend of two model families, plus an optional third:
  1. **nartaa CoAtNet**, 384 and 224 px. 2.5D input, 140 mm crop, 5 sequence slots. Trained with OAI for PF OA, Lateral OA and Synovitis. Its epochs were selected on the 58 expert studies.
  2. **heliosli R6 ConvNeXt-S.** Knee-centred crops from an SKM-TEA anatomy network; probably a fastMRI auxiliary task.
  3. **goodpjw 2.5D ConvNeXt-Tiny reader.** 0.929 alone. Apache-2.0, with its training code available.
- Recommended fork: `goodpjw2008/rsna-knee-coatnet-r6-2-5d-mil-lb-0-958`. It has a "Part B" template for an extra model and an estimated 3–4 h of runtime headroom.

**What adding models has done publicly.**
- Adding a weak model hurts: the 0.929 reader at full weight took 0.956 down to 0.954.
- One author reports that each LB step came from a *newly trained, different* model.
- Errors within the public family correlate at 0.86–0.89. The author claims a ceiling of about 0.960 for that family.

**Rules.**
- Public checkpoints are allowed and prize-eligible if listed in the submission (discussion 744056).
- External data is allowed only if it is free and needs only registration (743416).
- The host permits OAI, but it is controlled-access.
- The core public weights carry the license "other".
- No rule forbids training on the 58 gold studies.

**Expert label definitions (host).** Borderline cases are graded negative.
- Meniscus tear: signal reaches the surface on at least 2 images.
- OA: at least about 1 cm of >50%-thickness cartilage loss.
- Effusion and Baker's cyst: moderate or large.
- ACL and MCL: high-grade or acute.

**How report labels compare with expert labels.**
- Report labels reach about 0.89 AUC against expert labels.
- Experts call effusion positive more often than reports do.
- When a report says nothing about synovitis, about 34% of those studies are still positive.

**Evidence files:** `research/external_evidence/deep_research_20261017/` (three reports and 20 pulled public notebooks), plus `research/N6_RESULTS.md`, `research/N4_EARLY_GATE_AND_N5.md` and `outputs/n5_lr/*.json`.

## Answer through these four lenses, briefly, then decide

1. What would a top professional do here that I am not asking for or doing?
2. What is the rawest material they would start from, instead of our digested numbers (weak-label AUCs)?
3. Which specific tools and methods would they use that we have not?
4. Other than you or me, what would show that a result is right? Name the arbiter for each decision, and say what each arbiter cannot tell us.

## Decisions I need

**A. Why our LB sits below our validation.**
- What is the single cheapest test that separates these three causes?
  - A submission-path or preprocessing bug.
  - Labels that disagree with the expert definitions.
  - A validation design that inflates our scores.
- The 58 gold studies are still unopened. Decide whether to spend them now.
  - Option: score our exact submission path and the goodpjw reader, which never trained on gold, on the same 58.
  - State the noise (SE ≈ 0.02–0.03) and what the decision costs us later.

**B. Base submission.**
- Fork the 0.958 stack unchanged first, to prove we reproduce it and that the stack meets the rules?
- Also submit the never-submitted N-ensemble alone, so we learn our own family's true LB?

**C. The one addition worth our GPU hours.** It must be *different* from the public family; a weaker copy will not help. Candidates:
- Retrain the Apache goodpjw recipe with expert-aligned labels: multi-source consensus, grade→probability mapping and the effusion/synovitis rules.
- Distillation from out-of-fold image predictions blended into the labels.
- Knee-centred or finding-routed crops.
- OAI.
- Drop our N4/N5 line.

Pick one or two. Give GPU-hours, the exact single change, the success metric and its noise, and a kill criterion.

**D. Blend weight.** How do we set the weight for the addition without overfitting the public LB? No source of out-of-fold predictions exists for the public stack on train. Fix the weight beforehand, or use a coarse LB check?

**E. Calendar.** A day-by-day plan to 22 Oct, with GPU-hours and submissions per day.

**F. Final two submissions.**
- One robust pick with uniform, pre-fixed weights.
- One best-on-LB pick, only if it differs by more than the LB noise (±0.003 on about 390 studies).

**G. What to stop.** Explicitly: is an N4 audit or a manual review of 50–100 reports still worth doing?

## Rules for your answer

- Make one change per experiment.
- Do not tune per-finding weights on the public LB.
- Do not use any second account.
- Label every number as verified, claimed by someone else, or estimated.
- If you disagree with a premise above, say so first.
