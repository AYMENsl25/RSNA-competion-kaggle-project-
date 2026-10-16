# Prompt: decide the next experiments (RSNA Knee, 2026-10-15, 7 days to deadline)

You are acting as a panel of three: a senior medical-imaging AI researcher (knee MRI), a senior data scientist, and a Kaggle Grandmaster who has won RSNA imaging competitions. Decide **which experiments we run next and in what order**, then defend the decision. Do not summarise our history back to us; decide.

## Hard constraints
- Deadline 2026-10-22 23:59 UTC. Today 2026-10-15.
- Compute: local RTX 4050 6 GB (~55k studies/h ResNet18 @224, ConvNeXt-T 12 ep ≈ 1.7 h on 3 slots), ~1.9 Kaggle T4 GPU-h now, ~30 h after reset (~Oct 17). Free Kaggle CPU sessions. **Reserved:** E12 blend submission ~5–6.5 GPU-h, final-day reserve ≥ 4.5 GPU-h.
- No OAI data/weights. Public weights only run inside Kaggle (not downloaded locally). 58 expert studies sealed: **1 look left**.
- Two final selections: one prize-safe (our models only), one best-LB (E12 blend: ours 0.30 + Raptor maxspan 0.35 + goodpjw 0.35).

## Evidence (verified, numbers are ours unless marked)
- LB history: E01 0.769 → E1 0.807 → E4+E5 0.841 → Ens3 0.848 → Ens4 0.857. Val→LB gains have tracked ~1:1 for ensembles. Ens5 (+E3 ConvNeXt 6-slot) val 0.8846, not submitted.
- Single models val (public labels): E4+E5 ResNet18 0.8625; E8 ConvNeXt-T 0.8744; E3 ConvNeXt-T 6 slots 0.8778. Rejected: E2 (fixed 12-slab windows + attention head, −0.016; confounded — fixed windows removed random-window augmentation), E6 aug+16ep (−0.003), E7 320 px on a 256 px cache (−0.012, pure upsampling).
- Our read-out: 4 random 3-slice windows per plane (~12/study) train, 16/plane eval, **max-pool**, linear head, 160 mm crop @256 px (0.625 mm/px), labels = public LLM soft labels with rule relabel of mild Effusion/Baker's (v3).
- Public single models reach LB 0.928–0.95 (Raptor CoAtNet: 44 slots, 12 random windows train / 24–94 eval, **per-finding softmax attention-MIL**, 140 mm @336–384, backbone LR 3e-5/head 1e-3, 16 ep, top-3 avg; goodpjw ConvNeXt-T: 6 plane×fat-sat slots + slot & slice-position embeddings + 2-layer transformer + per-finding attention, 0.4 mm/px 384, labels = mean of 4 public LLM tables + grade→P mapping + Synovitis imputed from effusion grade; heliosli R7: 176 windows/study, aux graded-state + pairwise rank losses, EMA). Backbone choice reported to barely matter (ResNet50 0.940, DINOv2 0.942).
- Error analysis of Ens5 (96 worst cases read from raw reports): 43 label wrong/ambiguous, 49 genuine misses, 4 image-side. Label issues: "degeneration/mucoid ≠ tear" (ACL, menisci), severity floor (mild MCL thickening, minor fibrillation, trace Baker's, hedged contusion). Misses: mimics and **medial↔lateral compartment confusion** (6/16 lateral-meniscus/OA FPs have pathology in the opposite compartment). Lateral meniscus weakest (0.811); collapses to 0.61 when sagittal fat-sat is missing. Effusion AUC 0.882 on public labels vs 0.937 on v3 labels (label-limited). Sagittal crop clips the patella in ~27%. Member rank correlation: Effusion 0.91 (redundant), ACL 0.73 (diverse).
- Community: report-derived labels cap ~0.89–0.90 on gold (gold made from images); per-label blend tuning and cross-finding post-processing overfit (−0.020 on held-out gold); public split SD ≈ 0.003; weak-label OOF CV tracks LB r≈0.87, gold-58 r≈0.66. MONAI/TorchIO extra augmentation and medical pretrained weights judged ≈0 gain here.

## Candidate experiments (critique, merge, reorder, or replace)
A. Per-finding attention-MIL head with **random** windows (12/study train, 24–64 eval) on ConvNeXt-T, everything else = E8 (isolates the head; E2 confounded it).
B. Consensus labels: mean of public LLM tables + grade mapping + Synovitis imputation, plus our v3 severity/degeneration fixes (CPU only).
C. 6 slots + slot/slice-position embeddings (+ optional 2-layer transformer) on top of A.
D. Rebuild cache at ~0.4 mm/px, 140–150 mm, 320–384 px, 4–96 % slice span, wider anterior sagittal margin (CPU build + ~60 GB download).
E. Differential LR (backbone 3e-5–1e-4 / head 1e-3), OneCycle, EMA or top-3 average, 16 epochs.
F. 5-fold OOF on the train split → pseudo-label masked/under-mentioned cells (Synovitis, Fracture) → student.
G. Medial/lateral disambiguation (hemi-knee coronal crops or position embedding + aux loss).
H. More seeds/backbones of the best recipe for the ensemble.
I. Kaggle-side training of the best recipe at 384 px on T4 using the remaining GPU-h.

## What I want back
1. **The decision:** an ordered schedule (day by day to Oct 22) of at most 6 experiments, each with: exact change vs its control, the control, where it runs (laptop / Kaggle CPU / Kaggle GPU), hours, expected Δ val AUC with a range, and a **kill criterion**.
2. What a top practitioner would do here that we are **not** doing, and which of A–I they would drop.
3. The **rawest material** each experiment should start from (DICOM pixels/headers, raw report text, public training code, gold-58 images), not our digested summaries.
4. **External validation:** for each decision, what evidence other than us (private LB proxy, gold-58 look #2, an independent reproduction, paired bootstrap on OOF, a radiologist reading error cases) would show it is right — and when to spend the last gold-58 look.
5. How to choose the two final submissions to be robust to the private leaderboard (≈70 % of test), including which members/weights, given the public split noise.
6. The single biggest risk to our final score in the next 7 days and how to hedge it.
Be concrete and quantitative; say "unknown" where it is.
