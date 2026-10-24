# Prompt: last experiment before the Oct 22 deadline

Paste everything below the line into a new session.

---

You are acting as a senior medical-imaging AI engineer and Kaggle Grandmaster on the
RSNA knee abnormality detection competition. This is our **last** experiment. Final
deadline: **2026-10-22 23:59 UTC**. Kaggle GPU left this week: **__ h** (I will fill this
in; if blank, check `kaggle kernels` history and assume ≤ 6 h). One Kaggle account only.
Local RTX 4050 6 GB can be used freely.

## Where we are (verified facts, do not re-derive)

- F0 = unchanged public 0.958 fork (nartaa Raptor-384/224 + heliosli R6 x4, mirror/flip TTA,
  per-finding rank weights). Public LB **0.958**. Top of LB **0.969**.
- F1 = F0 + 0.2 x S1B student (labels v3). Public LB **0.954**. S1B *passed* our gold-58 and
  parent-correlation gates and still lost 0.004. => Our offline gates do not predict
  LB gains when blending into F0. Gold-58 is burned for selection.
- K1 (Raptor fine-tunes) were clones of the parent (rank corr 0.986-0.988). K3A was different
  but weak (gold 0.842). K2/K3B/S1A died from fp16 NaN. K4B collapsed at encoder LR 2e-4.
- Our own from-scratch line peaked around LB 0.857-0.89 public-proxy. Not competitive.
- F2 (frozen DINOv2-S + geometry head) is running **locally, zero Kaggle GPU**; it measures
  weak-label agreement only.
- Full history: `experiments/registry.csv`, `research/K1_F0_PUBLIC_WEIGHTS_TRACK.md`,
  `research/F0_NOVEL_STRATEGY_RESEARCH_2026-10-20.md`.

## What I want from you, in this order

### 1. Deep, targeted search first (not wide)
Before deciding anything, search:
- The current competition leaderboard and **every public notebook scoring >= 0.950**:
  author, members, resolution, TTA, blend method, runtime, and whether its members are
  *independent* of nartaa/heliosli weights (different author, data, or backbone).
- Discussion threads: what the 0.96+ teams say helped (members, blending, post-processing,
  per-finding tricks, label sources), and any known LB/private shake-up risk.
- MONAI / MONAI Model Zoo, TorchIO, timm, and published knee-MRI checkpoints for anything
  usable **offline, inside the 9 h notebook limit, with no new training**.
- Past RSNA comps (2019 ICH, 2022 cervical, 2023 abdominal, 2024 lumbar): what the winners'
  *final-week* moves were (blending public stacks, TTA, rank weighting, second-slot choice).
Report only what changes the decision, with links.

### 2. Answer these four questions explicitly
1. What would a senior medical-imaging + Kaggle GM person do here that I am **not** doing
   or asking? (Think: using both final-submission slots, blending independent public stacks,
   per-finding weight shrinkage, TTA, stopping training entirely.)
2. What is the **rawest material** they would start from, instead of my digested
   summaries? (e.g. per-member, per-finding prediction files from F0's notebook outputs; the
   DICOM headers/series coverage of the test set; the public notebooks' source code; the raw
   LB history of every submission we made.)
3. Which **specific tools and methods** would they use that we did not? Name them concretely
   (library, function, notebook), not categories.
4. What, **other than you or me**, shows the result is right? (Public LB on one pre-declared
   submission, private LB at close, agreement with an independently authored public notebook,
   group bootstrap on OOF if honest OOF exists.) Say which of these we can actually get before
   the deadline.

### 3. Then decide ONE last experiment
Hard constraints for the decision:
- Fits the remaining Kaggle GPU hours with >= 30% headroom, **including a full hidden-test
  inference commit** (not the 3-study preview).
- Prefer inference-only or local-GPU work over new Kaggle training. Training a new member
  from scratch is rejected unless you show it fits and has evidence of beating 0.958.
- Must be judged by a **pre-declared** single public-LB submission vs F0 0.958, plus a
  pre-declared rule for which two submissions I select as final (default: F0 is always one of
  the two).
- Must not repeat what already failed (fixed-weight blend of a weaker self-trained member,
  Raptor clones, high encoder LR, fp16 without NaN guards).
- F2 joins only if it passes a gate you define *now*, before its result is known.

Output:
- The decision in one paragraph, with the expected mechanism of gain and the main risk.
- An hour-by-hour plan from now to 2026-10-22 23:59 UTC: what runs locally, what runs on
  Kaggle, GPU hours per step, submission slots used, and the cut-off time after which we stop
  and lock final selections.
- A kill rule: what result makes us abandon it and keep F0.
- The exact files/notebooks to create or fork, kept minimal.

Do not launch any Kaggle run or submission without my explicit go.
