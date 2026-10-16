# V3 nine-day plan (Oct 13 → Oct 22 2026)

Status: **proposed, not started.** Nothing below runs until the user confirms.
Inputs: four research agents (data, competition, literature, preprocessing), one innovator agent, two Codex review rounds. Raw notes: session scratchpad `reports/01–05`, `codex_round1/2_answer.md`.

## Facts that changed the plan

| Fact | Consequence |
| --- | --- |
| Deadline 2026-10-22 23:59 UTC; **0 submissions** | Exercise the offline submission path first (E01), before any new modelling |
| Kaggle GPU: ~3 h now, ~30 h after ~Oct 17 | Kaggle GPU is for submissions + jobs that need 16 GB VRAM |
| **Local RTX 4050 Laptop 6 GB** (verified `nvidia-smi`), 547 GB free disk | Most training moves local; Kaggle CPU kernels (free) build the cache |
| GPU waits on DICOM decoding 95% of the time; 2.4 h per 3 epochs | uint8 memmap cache is a prerequisite: ~10× faster, so 12–16 epochs become affordable |
| E01 (6 ep) beat P00 (3 ep) by +0.020; D01/S01 lost at 3 ep | 3-epoch ablations measure convergence speed, not quality. Compare at matched, longer budgets |
| ACL labels are clean (0.95 agreement) but ACL AUC is 0.73, and doubling epochs gave no gain | Image/sampling problem: 4 random windows rarely hit the 3–6 ACL slices, and images are downsampled to 224 |
| Effusion labels agree with the rubric 0.47, Baker's 0.70; mild-severity words fill the Effusion positives | Labels are mismatched against the host rubric. This is CPU/API work, not GPU |
| No ImageNet mean/std; p99 clipping saturates fluid; aspect-distorting resize, no antialias; no L/R canonicalisation | Fix all of these in the cache (one package, compared against E01) |
| LB 0.95–0.966; encoder size ~irrelevant; per-finding pooling, soft labels, OOF pseudo-labels, more inference windows are what top teams credit | No backbone sweep. Spend on labels, sampling, crops, ensembling |

## Hard rules (unchanged)
- The 58 expert studies stay sealed: at most 2 predeclared looks, never used to tune blends.
- Decisions use the frozen 863-study validation set: paired group bootstrap, per-finding AUC, macro and macro-without-Synovitis.
- fp16 on both laptop and T4; **never bf16 on T4**; T4 machine shape; offline pylibjpeg/gdcm.
- No OAI data, and no weights trained on OAI.
- One `canon_series()` function shared by cache build and offline inference, with a 20-study raw-vs-cache parity gate.

## Phase 0: infrastructure (Oct 13–15, ~1 h Kaggle GPU)

| ID | Work | Where | Gate |
| --- | --- | --- | --- |
| I1 | uint8 memmap cache: IPP order, canonical orientation, laterality flag, 160 mm physical crop, antialiased 256 px, foreground p0.5–99.8; all series kept with a per-slice index; 2–3 shards | Kaggle **CPU** kernels → private datasets → download | 20-study parity; checksums |
| I2 | Local bring-up: CUDA PyTorch, 1 cached epoch timed, 1 h thermal run, E01 predictions reproduced on 20 studies | Laptop | Kill local route if < 0.4× T4 or throttling |
| I3 | Offline submission notebook (E01 checkpoint, T4, fp16, IPP sort, no 0.5 fallbacks); measure runtime | Kaggle GPU ~0.8–1 h | **First real submission**: LB calibration of our proxy |
| I4 | Label audit: host rubric vs public labels on the 120 adjudicated reports (Effusion, Baker's, Lateral OA) | CPU | Defines targets for E4 |

## Phase 1: experiments (Oct 15–21)
Each row is compared paired against its stated control at the **same epochs and seed**. Expected gains are estimates on the proxy validation.

| # | Experiment | Hypothesis / recipe | Control | Where, cost | Expected | Kill if |
| --- | --- | --- | --- | --- | --- | --- |
| E1 | **Cached baseline v3** | I1 preprocessing package + ImageNet norm, P00 architecture, 12 epochs | E01 | Laptop ~1–2 h | +0.01–0.03 | worse than E01 → debug parity |
| E2 | **Full-slab plane routing + per-finding attention-MIL** | 24 central slices/plane; per-finding softmax over planes (ACL→sag, MCL/menisci→cor, PF→ax) | E1, matched windows/budget | Laptop ~3–5 h | ACL +0.05+, macro +0.02–0.04 | ACL gain < 0.02 |
| E3 | **Non-fat-sat sagittal for ACL** | add the PD/T1 sagittal series (5,197 unused) as an extra slot | best of E1/E2 | Laptop ~3 h | ACL +0.01–0.03 | no ACL gain |
| E4 | **Severity-aware rule relabel (no API; user declined paid LLM 2026-10-13)** | multilingual severity lexicon (reuse N01; small/minimal/trace/mild/leve/gering/klein…) downgrades mild-only Effusion/Baker's positives to soft ~0.3; check on 120 adjudicated reports first | public labels, same model | CPU + laptop run | +0.003–0.01 (Effusion/Baker's) | agreement gain < 5 pts on the 120 |
| E5 | **Loss/targets** | masked soft BCE (0.2–0.8 kept soft) + clipped pos_weight (1–10) vs hard BCE | best recipe | Laptop ~3 h | +0.005–0.01 | CI includes 0 and point ≤ 0 |
| E6 | **Augmentation + longer schedule** | anatomy-safe aug (rot ±10°, shift 8%, RRC 0.85–1, gamma, bias field, ±1-slice jitter; NO L/R flips), 16 epochs, top-3 checkpoint averaging | same recipe, 12 ep, no aug | Laptop ~5 h | +0.005–0.015 | — |
| E7 | **High-res ACL/MCL crops** | 320–384 px crops, central sagittal notch, medial coronal (laterality-aware) | best recipe at 256 | **Kaggle T4** ~4–6 h (VRAM) | ACL/MCL +0.01–0.03 | no per-finding gain |
| E8 | **One diversity backbone** | ConvNeXt-T or EffNetV2-S on best recipe (one, not a sweep) | ResNet18 best | Kaggle T4 ~5 h | ensemble +0.005 | low error-decorrelation |
| E9 | **OOF pseudo-labels** (conditional) | 4-fold teacher; teacher probabilities replace masked cells, conservative weights; under-mentioned Fracture/Synovitis | best single | Laptop ~12–16 h | +0.005–0.015 LB | student ≤ teacher OOF; skip if laptop < 0.6× T4 |
| E10 | **Inference sampling + TTA** | match train/infer windows; sweep 16/32/64 windows; anatomical mirror TTA | 16 windows, no TTA | Laptop eval only | +0.003–0.006 | runtime > budget |
| E11 | **Ensemble** | rank-average seeds/folds/E8 per finding, chosen on validation only | best single | Laptop + T4 inference | +0.005–0.01 | — |
| E12 | **Public-checkpoint floor** *(user decision)* | fork best non-OAI public notebook as a fallback / blend member | ours alone | Kaggle GPU ~1 h | LB → ~0.95+ | prize-eligibility unresolved (disc. 744056) |

Cut, with reasons: 12-logit stacker (measured 0.8215→0.8173); structural label fill (fabricates confident Synovitis); frozen DINOv2 (V1 0.566); backbone sweep (S→B +0.001); N4/Nyúl (cost, no evidence); 5-slot multiseries (S01 lost; risk); metadata features and transductive test pseudo-labels (≈0 gain, inference risk); OAI.

## Schedule

| Days | Laptop | Kaggle CPU | Kaggle GPU |
| --- | --- | --- | --- |
| Oct 13–15 | I2, I4, prep E4 pilot | I1 cache shards | I3 E01 submission (~1 h); rest held for fixes |
| Oct 15–17 | E1 → E2 → E3 | finish/verify cache | — |
| Oct 17–19 | E4, E5, E6 | label assets, inference assets | E7, E8 (~10 h) |
| Oct 19–21 | E9 if justified, seeds, E10 | final notebook dependency check | 2nd submission by Oct 21 (~1.5 h) |
| Oct 21–22 | E11 freeze | — | final submission + ≥4.5 h reserve |

Budget: Kaggle GPU ~1 h now + ~18–22 h after reset, with ≥4.5 h reserved. Laptop ~60–90 h if I2 passes.

## Progress log
- 2026-10-13: E01 submitted, public LB **0.769** (submission 57144276), far below the proxy→LB mapping others report.
- 2026-10-13: user approved spending **expert-label look #1** on E01 to diagnose the 0.769 (labels vs hidden-test pipeline); to be run when work resumes.
- 2026-10-13: cache built (5 CPU shards, ~62 GiB); download resumable via `notebooks/cache_v3/fetch.py`; paused at s0 4.3 GB for PC shutdown. On resume: rerun fetch.py, then queue E1 (command in the trainer report: `src/train_v3.py ... --out outputs/e1_v3 --epochs 12 --labels public --target hard --res 224 --mirror`).
- 2026-10-13: **expert-label look #1 spent** (E01, exact submission inference code on the 58 expert train studies, free CPU kernel `rsna-knee-v2-e01-expert-look1`). E01 expert-58 macro **0.735** (matches LB 0.769); 58/58 ok, 0 degraded/failed/fallback. Public LLM soft labels as predictors 0.878 (labels_v3 identical on these 58), hard >0.5 0.809. Verdict: **model**, not pipeline; labels are a secondary loss (hard thresholding costs ~0.07, Effusion 0.65 hard). Results in `outputs/expert_look1/`. Expert labels used for nothing else.

## Codex vs Claude/innovator, recorded disagreements
- **Labels before or after the first submission.** Codex: E01 submission first, then a paired label test. Accepted. Label audit runs on CPU in parallel, so no time is lost.
- **Full-slab routing.** Codex wants a matched-budget control, plus a check that peripheral findings are not lost. Adopted (E2 control = E1).
- **Structural fill.** Innovator for, Codex against. Cut.
- **Public checkpoint.** Both: user decision on prize eligibility before relying on it.
- 2026-10-13: E1 (cache preprocessing, ResNet18, 12 ep, public hard labels) val 0.8518 (+0.030 vs E01, CI [+0.022,+0.039]); **public LB 0.807** (sub 57157791).
- 2026-10-13: E4+E5 (labels_v3 + soft targets + pos_weight clip 10, otherwise = E1) val public 0.8625 / v3 0.869 at ep12 (outputs/e45_v3). Not yet submitted. E2 code ready (--slab 12 --head attn), not started.
- Next session: analyse research/public_nb/ (two public notebooks user asked about), integrate into plan; then E2 on top of E4+E5.
- 2026-10-14: read 2 public analysis notebooks (research/public_nb/). Top public LB (0.95+) is a monoculture of a few public weight datasets: nartaa CoAtNet SWA 384/224 (reportedly OAI-trained → avoid), heliosli ConvNeXt-small S6/R7 (fastMRI auxiliary, OrthoFoundation anatomy), goodpjw2008 2.5D ConvNeXt-T reader (0.929 alone); pilkwang baseline (DINOv2-S + own LLM labels) 0.891 in ~385 s. Blending them = E12 (user decision pending: prize eligibility / external-data provenance). Diversity, not copies, is what lifts blends.
- 2026-10-14: E4+E5 **public LB 0.841** (sub 57181257) — best so far.
- 2026-10-14: E2 (E4+E5 recipe + --slab 12 --head attn) val 0.8461 vs 0.8625: -0.016 [-0.025,-0.009], worse on 11/12 → rejected. Likely cause: fixed slab windows removed random-window augmentation (attention head untested alone).
- Next on resume (nothing running): write anatomy-safe augmentation, then E6 (E4+E5 recipe + aug + 16 ep + top-3 ckpt avg). Dashboard: .venv/Scripts/python src/dashboard.py → http://localhost:8050
- 2026-10-15: E6 (E4+E5 + --aug + 16 ep + avg last 3) val 0.8591 vs 0.8625: -0.003 [-0.009,+0.002] → no gain alone; checkpoint averaging = same (0.8591).
- 2026-10-15: rank-average ensembles on val (public labels) vs E4+E5: e45+e6 +0.005 [0.002,0.008]; e45+e6+e2 **0.8703, +0.008 [0.004,0.012]**; +e1 adds nothing. E2 (attention head) is useful as a diverse ensemble member despite losing alone. Next: multi-checkpoint ensemble inference notebook.
- 2026-10-15: Ens3 (rank-avg e45 best + e6 ep16 + e2 best; notebooks/v3_submission multi-ckpt, dataset aymenslimani/rsna-knee-v3-ens3) parity 20/20 max diff ≤2.4e-7 per model; GPU kernel rsna-knee-v3-ens3-gpu v2 (v1 failed: wrong dataset_sources, 30 s), visible 11 s, 0 failures; **public LB 0.848** (sub 57196314) vs 0.841 E4+E5 — best so far.
- 2026-10-15: Ens3 (E4+E5+E6+E2 rank avg) **public LB 0.848** (sub 57196314).
- 2026-10-15: read lavinwins/rsna-gold-medal (0.950) and rjings/rsna-versia-8 (0.957 recipe). Both = blends of public weights. gold-medal uses nartaa weights whose filenames contain "_oai_" (OAI-trained → excluded for us) + heliosli R6/R7 ConvNeXt + OrthoFoundation. versia = 13 public datasets: 20 DINOv2/v3 slot-attention members, A5 attention folds, RadImageNet ResNet-50 heads, Raptor CoAtNet views (maxspan/native384), per-finding routing. Transferable ideas for OUR models: 6 slots (plane x fat-sat) instead of 1 series/plane; dense inference windows (94); RadImageNet R50 / DINOv2 as diverse families; guard against studies with no usable slot (constant ties).
- 2026-10-15: E7 (E4+E5 at res 320) val 0.8505 vs 0.8625: -0.012 [-0.018,-0.006]; adds nothing to Ens3 (-0.0004). Rejected. Cause: cache is 256 px (160 mm FOV, 0.625 mm/px), so 320 only upsamples; real higher resolution needs a new cache (e.g. 384 px or tighter crop) — expensive, deprioritised.
- 2026-10-15: --slots {3,6} added (train_v3 + m00_data.select_slots + v3_submission per-model slots from log.json, default 3). 6 = plane x {fat-sat, non-fat-sat}, missing slots masked; pilot: 2/20 studies have all 6, 10 have 5, 8 have 4. Infer==train inputs on pilot (stubbed DICOM decode) for slots 3 and 6. max head only. E3 = E4+E5 + --slots 6, ~5 h (2x windows).
- 2026-10-15: E8 (E4+E5 recipe + --backbone convnext_tiny [timm fb_in22k_ft_in1k] --lr 1e-4 --wd 0.05, max head, fp16, bs 2, no grad ckpt; peak ~2 GiB) 12 ep in ~1.7 h. Val public **0.8744** vs E4+E5 0.8625: **+0.012 [+0.004,+0.020]**, biggest gains Fracture +0.039, Contusion +0.039, PF OA +0.020; MCL -0.007. 4-model rank avg e45+e6+e2+e8 **0.8800** vs Ens3 0.8703: +0.0097 [+0.0076,+0.0118], better on 12/12 (outputs/ens_e45_e6_e2_e8.csv). Submission build reads `backbone` from log.json (inference needs timm in the Kaggle image; build with pretrained=False). Not submitted.
- 2026-10-15: public-weights research (scratchpad pubweights/). Clean candidates: dreaddevelopment/raptor-knee-maxspan (CoAtNet 384, LB 0.928, comp-only data, CC0, training code public) and goodpjw2008/rsna-knee-2-5d-convnext-reader (LB 0.929, Apache-2.0, train code in dataset). Both trained on all non-gold studies (seen our val; gold-58 clean). Avoid: nartaa (OAI), heliosli bundles (license/withdrawn), RadImageNet heads (CC-BY-NC). Recommendation: E12 = inference-only rank blend Ens4 + Raptor + goodpjw (fixed weights, ~3-5 GPU-h after reset, expected LB ~0.93-0.945); fine-tuning public weights judged poor value (+0-0.01). Keep Ens4 as prize-safe final selection. Awaiting user decision.
- 2026-10-15: Ens4 (e45+e6+e2+e8 convnext) submitted: dataset aymenslimani/rsna-knee-v3-ens4, kernels rsna-knee-v3-ens4-{gpu,parity,preflight} v1. timm present offline in Kaggle image. Parity 20/20 max diff e45 2.4e-7, e6 1.8e-7, e2 2.4e-7, e8 3.0e-7; preflight 3/3, GPU 3/3, 0 failures. **Public LB 0.857** (Ens3 0.848, +0.009, matches val +0.0097). parity_compare.py now passes backbone.
- 2026-10-15: E12 blend kernel ready (notebooks/e12_blend, kernel aymenslimani/rsna-knee-v3-e12-blend-gpu v2): Ens4 .30 + Raptor maxspan .35 (CC0) + goodpjw reader .35 (Apache-2.0), per-reader subprocess, per-finding rank pct, weights renormalised over successful readers, 8.5 h guard. CPU preflight 3/3 and T4 smoke 3/3 all readers, 0 failures (~0.1 GPU-h). Extrapolated hidden runtime <=6.4 h (expected 3-5 h). Not submitted; submit after quota reset (~Oct 17), command in notebooks/e12_blend/README.md.
- 2026-10-15: E3 (E8 ConvNeXt-T + --slots 6, val-bs 1 after restart) val 0.8778 vs E8 0.8744: +0.003 [-0.002,+0.009] (MCL +0.021, Baker's +0.016, LatMen -0.035). Ens5 (ens4 + e3) val **0.8846**, +0.0045 [0.0034,0.0058] vs Ens4 → candidate replacement for the ens4 reader in E12 and for the prize-safe final.
