# The next six experiments — decision for October 15–22, 2026

**Decision:** run random-window attention first; test conservative consensus labels; then test sequence/position information, real pixel detail, differential learning rates, and one independent repeat of the winner. Keep the existing successful ensemble as a fallback throughout. Do not start five-fold pseudo-labeling or a backbone sweep this week.

This applies `NEXT_DECISION_PROMPT.md` as a decision framework. Its factual claims were checked against predictions, logs, source code and live read-only Kaggle queries. It is not an instruction to launch jobs. **No training, submission, expert-label evaluation or public checkpoint download was performed for this review.** Public reference Python source and metadata were downloaded, inspected, and not executed.

Prepared October 15, 2026. Schedule dates below use Istanbul time. The live Kaggle API confirms **October 22, 23:59 UTC = October 23, 02:59 Istanbul**. Treat October 22 as the repair day, not a training day. The stated GPU balance/reset is a planning assumption; it was not independently measurable here.

## What the evidence actually establishes

I recomputed these results from the saved prediction CSVs, with exact UID matching, finite-value checks, the frozen public-label mask, and 1,000 paired group-bootstrap draws. Current V3 validation has **885 studies in 843 groups**, not the older 863-study cohort. The split contains 3,457 train / 885 validation / 65 excluded studies; current runs actually train on 3,428 studies after availability/target filtering. Preserve that distinction when making comparisons.

| Existing experiment | Public-label validation AUC | Paired change versus stated control; 95% bootstrap interval | Decision implication |
|---|---:|---|---|
| E4+E5 | 0.862498 | Control | Existing ResNet baseline |
| E2: fixed windows + attention | 0.846145 | vs E4+E5: −0.016353 [−0.024526, −0.008544] | Reject this package; attention alone remains untested |
| E6, epoch 16 | 0.859106 | vs E4+E5: −0.003392 [−0.008610, +0.002030] | No demonstrated single-model gain; retain its ensemble contribution |
| E7: 320 input from 256 cache | 0.850510 | vs E4+E5: −0.011987 [−0.018100, −0.006339] | Does not test recovery of raw spatial detail |
| E8: ConvNeXt-T | 0.874449 | vs E4+E5: +0.011951 [+0.004471, +0.019802] | Best clean three-slot control |
| E3: six-slot ConvNeXt, epoch 11 | 0.877816 | vs E8: +0.003367 [−0.001696, +0.008773] | Standalone gain uncertain; lateral-meniscus AUC falls about 0.034 |
| Ens3 | 0.870347 | — | Existing ensemble |
| Ens4 | 0.880017 | vs Ens3: +0.009670 [+0.007597, +0.011835] | ConvNeXt adds useful diversity |
| Ens5 | **0.884561** | vs Ens4: **+0.004544 [+0.003409, +0.005801]** | Strong own-model candidate, pending deployment parity and LB |

Live Kaggle submission records confirm E01 **0.769**, E1 **0.807**, E4+E5 **0.841**, Ens3 **0.848**, Ens4 **0.857**. Ens5 has no recorded submission. The two ensemble increments are encouraging; they do not establish a universal 1:1 validation-to-LB conversion.

Important qualifications:

- These are **one-split weak-label** comparisons, not five-fold OOF results. Their intervals condition on already-trained, already-selected models. They exclude seed variation, selection bias, label error and test-distribution shift. Grouping mitigates known links; it does not prove patient independence.
- Ens5 exactly reconstructs from E45 epoch 12, E6 epoch 16, E2 epoch 12, E8 epoch 12 and **E3 epoch 12**. E3's `best.pt` instead corresponds to epoch 11 under v3-label selection. Substituting epoch 11 gives 0.884483, not the saved 0.884561. Match checkpoint and prediction hashes before deployment; do not silently call both versions the same model.
- Existing `relabel_v3.py` changes **Effusion and Baker's only**. ACL/meniscus degeneration fixes discussed in the prompt are not implemented. The four raw English error reports spot-checked for this review include mucoid ACL degeneration, explicit intact ligaments, and conflicting report statements. That supports targeted investigation, not automatic relabeling of all similar cases.
- The 43 label-wrong/ambiguous cases among 96 worst cases are a selected, report-based review. They are neither a population label-error estimate nor image adjudication by a radiologist.
- Synovitis's apparently excellent Ens5 proxy AUC, **0.97185**, covers just **150/885** observed labels: 123 positives and 27 negatives. It says little about the 735 masked cells. Effusion's 0.88170 public versus 0.93739 v3 AUC uses different targets; the difference is not a model improvement.
- The missing-sagittal-fat-sat subgroup has only **42 studies**. Its lateral-meniscus AUC of 0.60887 is an important warning, not a precise population effect. The reported 27% patellar clipping was not independently remeasured here; the cached montage shows edge truncation in some studies and warrants a raw-image retention audit.

Reproduction details, per-finding scores, hashes and reconstructed ensembles are in [NEXT_SIX_EVIDENCE_AUDIT.json](NEXT_SIX_EVIDENCE_AUDIT.json). The repeatable audit is [audit_next_six.py](../src/research/audit_next_six.py). The [read-only Kaggle snapshot](external_evidence/kaggle_readonly_snapshot_20261015.json) preserves the live checks and their limits.

## The six experiments, in execution order

The gain ranges below are **judgmental planning ranges, not confidence intervals or promises**. They are absolute macro-AUC differences on the fixed validation labels. Do not add them together. Hours are total active device time for the stated arms; implementation, queueing and transfers are separate. Budget training using full train-plus-validation epochs: measured E8 was 1.62 hours for 12 epochs and E3 was 3.15 hours, rather than extrapolating ResNet-only throughput.

Common defaults: frozen train/validation/exclusion manifests; ImageNet initialization already used by E8; fp16; no new augmentation; AdamW weight decay 0.05 for ConvNeXt; the existing soft v3 training targets and clipped positive weights unless explicitly changed; original 4 random windows per slot for training and 16 deterministic windows per slot for primary evaluation. All primary comparisons use matched epochs and the same seed. Save final-epoch predictions as well as any selected checkpoint. Never use a newly modified target table as the only validation judge.

| Order / ID | Exact intervention and control | Location / device hours | Expected validation change | Kill or promotion criterion |
|---|---|---|---|---|
| **1 / N1: attention with random windows** | E8, three slots, 12 epochs, seed 0. Change only max pooling to the existing per-finding gated attention head. **No `--slab`**; retain 4 random windows/slot, 16/slot evaluation, resolution 224, LR 1e−4 and the same labels. Control is E8 epoch 12. | Laptop **2–3.5 h**, including secondary evaluation. No Kaggle GPU. | Central expectation +0.003–0.010; plausible −0.010 to +0.015. | Do not promote as a better single if final paired gain <+0.003 or its CI includes zero. A failed single can survive only as a predeclared low-weight ensemble addition with ≥+0.002 ensemble gain and positive paired lower bound. Abort only for engineering failures early; a short-run AUC is not a reliable final ranking. |
| **2 / N2: conservative consensus targets** | Use the **E8 architecture and 12-epoch recipe**, changing training targets only. Equal mean of available, semantically compatible, non-unknown public probabilities when ≥2 independent sources address a cell; otherwise retain v3. Apply the existing mild-only Effusion/Baker's rules after merging. **No Synovitis imputation, no gold-based source weights, no invented grade values.** Include a degeneration correction only if independently justified in a fresh train-only audit and frozen before the image run. Control is E8 with current v3 targets. | Laptop / Kaggle CPU **2–4 CPU h**, then laptop **1.7–3 GPU h**. | Central 0 to +0.004; plausible −0.006 to +0.008. Lower priority than the attention hypothesis. | First audit 60 fresh candidate-train reports, stratified by language/disagreement, without model scores. Drop if correction agreement does not improve ≥5 percentage points, any rubric-critical stratum degrades, or the changes mostly convert silence to confident labels. Image-stage promotion requires +0.003 on the frozen public proxy with a positive CI, or predefined reviewed-target improvement with no overall degradation and later seed confirmation. |
| **3 / N3: six-slot sequence and position information** | If N1's attention remains credible, first train a **six-slot attention control without token embeddings**, repairing the three-plane-bias restriction. Then add learned slot IDs plus a small MLP of normalized canonical slice position. Both arms: same six slots, same masks, 4 random windows/slot, 16/slot eval, 12 epochs, same labels/LR/resolution. **No transformer and no auxiliary compartment loss.** This isolates the embedding package. E3's six-slot max model is a secondary reference, not a substitute for the matched control. | Laptop **6.5–9 h total** for the two arms. Cancel if N1 is clearly unhelpful and no ensemble signal survives. | +0.002–0.008 versus the matched six-slot attention control; plausible −0.006 to +0.012. | Require ≥+0.003 macro or ≥+0.015 lateral-meniscus AUC with macro no worse than −0.001, followed by positive ensemble gain. Reject if either meniscus loses >0.010 overall, or performance is obtained only by sacrificing a major compartment. Report the small missing-sagittal subgroup with counts/intervals; do not optimize directly to those 42 studies. |
| **4 / N4: true spatial detail and anatomy retention** | Rebuild pixels from raw DICOM at **384 px**, using 160 mm for coronal/axial. Candidate sagittal crop: **176 mm square, center shifted 8 mm anterior**, subject to a train-only raw-image retention pilot. This extends the old anterior boundary by 16 mm while retaining its posterior boundary. Use 320-px model input, E8's three-slot max model and unchanged 12-epoch labels/sampling/LR. Control is E8's 224 input / 256 cache / 160 mm crop. **This is explicitly a preprocessing package; a gain cannot be attributed to resolution alone.** Retain full original slice coverage. | Kaggle CPU **2–4 active h**, transfers separately; laptop **3.5–5.5 GPU h**. A measured local failure may move this same experiment to Kaggle GPU, capped at **4 h**, otherwise skip it. | Central +0.003–0.010; plausible −0.012 to +0.015. | Do not build the full cache unless a 50-study train-only pilot improves anterior retention without losing posterior structures, and 20 raw-vs-cache comparisons pass. Require ≥+0.004 macro or ≥+0.002 fixed-weight ensemble gain with positive CI. Skip if assets are not ready by Oct 18, or projected training/runtime exceeds budget. |
| **5 / N5: differential learning rates, fairly tested** | Freeze one parent: N1 attention if accepted, otherwise E8 max, keeping three slots and 224 input. Train **two fresh 16-epoch arms**: uniform LR 1e−4 versus encoder 3e−5 / all non-encoder parameters 1e−3. Both retain the same cosine multiplier, initialization, labels, weight decay and augmentation. Final epoch 16 is primary. **No simultaneous OneCycle switch, EMA, extra augmentation or top-3 checkpoint search.** | Laptop **4.5–7 h total**. | Central +0.002–0.006; plausible −0.005 to +0.010 versus the uniform 16-epoch control. | Kill for persistent nonfinite gradients or unstable loss after warm-up diagnostics. Final promotion requires ≥+0.003 with positive CI, or robust ensemble contribution. Compare uniform-16 to existing E8-12 only as a secondary schedule observation; do not attribute that difference to differential LR. |
| **6 / N6: independent seed confirmation and ensemble contribution** | Choose **one actually trained winning recipe** from N1–N5; do not combine untested winners. Repeat the winner and **its matched control with seed 1**, preserving all other settings, masks and initialization source. Compare winner−control separately for seed 0 and seed 1. Evaluate the fixed new ensemble described below. | Laptop **4–12 h total**, hard cap 12 h. No new backbone. If no new recipe passed, cancel and keep Ens5/Ens4. | Single-model expected improvement is **0** versus its seed-0 recipe; ensemble expectation +0.001–0.004, plausible −0.003 to +0.006. | Reject the claimed recipe improvement if the seed-1 matched difference changes sign materially. Promote the new ensemble only for ≥+0.002 versus Ens5, positive paired lower bound, macro-without-Synovitis no worse than −0.001, and no >0.010 regression on an adequately supported major finding. Otherwise keep the simpler existing ensemble. |

**Dense inference is a cheap secondary readout of N1/N6, not a seventh training branch.** On frozen checkpoints compare 16 centers/slot against a nested extension to 32 and then all valid centers up to 64. Always include the original 16 centers; the current `linspace` implementations for 16 and 32 are not nested. Log actual unique centers, mask duplicates, and preserve physically adjacent triplets within each series. Require ≥+0.001 validation gain, an own-model notebook projection ≤3 h, and an E12 whole-notebook projection ≤6.5 h with margin. More windows are not automatically better, especially for softmax attention. Do not change windows in the primary head comparison. Do not add horizontal-flip TTA this week.

**Cache correction:** the five existing shard summaries total **49.99 GiB of slice arrays**. The same slice inventory at 384 costs approximately **112.48 GiB uncompressed**, not another 60 GB. Compression, selected-series caching or a different inventory can alter download size; measure them, do not assume them. Use about ten shards, respecting the existing per-output size guard. Benchmark a 1-GiB transfer: 112.5 GiB takes about 3.2 h at 10 MiB/s or 16 h at 2 MiB/s, before overhead. Keep old and new cache versions side by side. At 320 input the candidate effective sampling is 0.50 mm/px for 160 mm crops and 0.55 for 176 mm crops; cached resolution alone is not model-input resolution.

N3/N4 require implementation and parity work. Current `train_v3.py` explicitly rejects six-slot attention, and the existing submission builder embeds one shared `canon.py`. A mixed old/new-cache ensemble therefore needs **per-checkpoint preprocessing versions**, not replacement of the global constants. This work belongs in the engineering allowance below. No command for these unimplemented branches is represented as runnable today.

## Day-by-day schedule and compute reservation

| Istanbul date | Laptop / local preparation | Kaggle CPU | Kaggle GPU after quota actually resets |
|---|---|---|---|
| **Oct 15** | Freeze hashes; verify the exact Ens5 checkpoint choice; implement/run N1. Start N2 train-only audit. | N4 50-study retention pilot and 20-study parity, using nonexpert studies. | Preserve the stated ~1.9 h current allowance for breakage; no full E12 yet. |
| **Oct 16** | N2 training after its audit; implement N3 masking/position handling and start matched six-slot arms if justified. | Build N4 cache shards only if the pilot passes. Download incrementally and measure transfer rate. | No training planned. |
| **Oct 17** | Finish N3; evaluate 16 versus nested denser windows. Prepare versioned preprocessing assets. | Complete cache and check index coverage/hashes. | **Run the already-prepared E12 control**: Ens4 .30 + Raptor .35 + goodpjw .35. Allocate up to 6.5 GPU-h. Do this early enough to expose hidden-runtime problems. |
| **Oct 18** | N4 if ready, then N5's two matched arms. Do not queue all heavy variants concurrently on the laptop. | Decoder and per-checkpoint preprocessing checks on a diverse nonexpert sample. | N4 may use ≤4 h only if a measured local pilot fails; skip if it cannot finish within that cap. |
| **Oct 19** | N6 matched seed-1 confirmation, then freeze at most one new own-model ensemble. | Package and compare raw-vs-cache predictions. | Submit the selected own-model candidate with up to 3 h allowance. If no new recipe qualifies, submit Ens5; keep Ens4 as the already-scored fallback. |
| **Oct 20** | Lock models, windows, weights and all acceptance criteria. | **Spend the last gold-58 look once**, as the single batch described below; no subsequent tuning. | One predeclared E12 comparator: **Raptor .50 + goodpjw .50**, zero ours. Allow up to 6.5 h. If the first E12 failed, repairing it takes priority over a second weight comparison. |
| **Oct 21** | Select the two completed final entries. Freeze source, dependency and checkpoint manifests by 18:00 Istanbul. | Check submission schema and failure summaries. | Repair/reproduce only. No architecture search. |
| **Oct 22** | Verify selected versions and completion status. | — | Repair only, with a start time that allows the full scoring run and queue margin before Oct 23 02:59 Istanbul. |

Local training/evaluation allowance is about **22–40 GPU-h** for all six branches including their specified controls, plus **12–20 h human/agent engineering** and cache transfers. This is not a 40-hour elapsed-time promise. If work slips, cut N4 first when data transfer fails its gate; cut N2 image training if its label audit is weak; preserve N1, N6 and completed submissions.

Post-reset Kaggle reservation, assuming the stated 30 h is actually available:

| Allocation | GPU-h ceiling |
|---|---:|
| Existing E12 control | 6.5 |
| External-only E12 comparator | 6.5 |
| One final own-model submission | 3.0 |
| Small GPU parity/preflight overhead | 1.0 |
| Optional relocation of N4 | 4.0 |
| **Final repair reserve, left untouched** | **7.0** |
| Total | **28.0**, leaving 2.0 unassigned |

Seven hours is deliberately more than the prompt's 4.5-hour minimum: a reserve smaller than the projected full E12 job does not cover one complete rerun. Do not add current pre-reset quota to a refreshed allowance unless Kaggle actually carries it forward. Published and hidden scoring runs can have different quota accounting; inspect usage after the first completed E12. If less quota returns, cancel optional N4 GPU training and the comparator before consuming the repair reserve.

## Start from the rawest useful material and validate beyond our summaries

| Experiment | Raw material to start from | Evidence that would support the decision, and its limits |
|---|---|---|
| N1 | Existing full-series cached pixels plus original per-series ordering/masks; E8 run config/weights; the actual attention implementation. Inspect raw DICOM triplets on the nonexpert pilot to confirm adjacency. | Primary MIL paper establishes the mechanism; public reader/trainer code provides a concrete independent implementation. Our paired 885-study gain and seed-1 result test this dataset. Neither a paper nor a notebook score predicts our gain. |
| N2 | Official **raw report text**, host rubric, individual public probability tables and their extraction semantics. Use fresh train-only reports, not the previously selected worst validation cases, for rule development. | Blinded report review by a radiologist would strengthen rubric fidelity; an independent second reviewer is useful if a radiologist is unavailable. It still assesses reports, not hidden image labels. Freeze the existing public/v3 validation targets. Last gold look is only a late veto, never a source-weight fit. |
| N3 | `train_series.csv` sequence flags, DICOM IOP/IPP/spacing/laterality and cache positions; reference `knee.py`; images where medial and lateral anatomy can be checked. | A separate reviewer verifies masks and canonical positions, especially after mirroring. Radiologist reading of a fresh nonexpert set can test whether apparent compartment errors are real. Require paired gains and seed confirmation; six selected opposite-compartment cases are insufficient to prove causality. |
| N4 | **Original DICOM pixels and headers**, including original FOV boundaries. Old 256-px pixels cannot restore anatomy or detail already removed. | Blinded before/after anatomy-retention review on 50 nonexpert training studies; exact input/prediction parity on 20; a separate 100-study stress sample with missing sequences, long stacks, coarse spacing and laterality uncertainty. Image retention is necessary, not proof of AUC gain. |
| N5 | Actual optimizer groups, initialization, stepwise LR logs, gradient diagnostics and full learning curves from a clean restart. | Public training source establishes that differential rates are a plausible recipe, not its causal benefit. Only the two matched 16-epoch arms and independent-seed direction test the hypothesis here. |
| N6 | Frozen checkpoint/prediction files for every member; exact per-checkpoint pixel pipelines; real offline runtime/failure outputs. | Seed 1 provides independent stochastic replication. Paired group bootstrap measures sampling uncertainty. A completed hidden-test notebook checks operational generalization; public LB is still only a partial test sample. Gold's final locked batch checks gross label-domain mismatch. |

There is **no verified independent private-LB proxy** available locally. Repeatedly bootstrapping the same 885 cases does not create one. A second split would require fresh matched training of control and candidate; do not call the existing single-split predictions OOF. Public reference checkpoints have seen substantial parts of our weak validation cohort, so **never choose their blend weights on our 885-study scores**. Their precise overlap with each development group is not established by the available metadata.

If a human radiologist is available, request a blinded read of 30 new nonexpert cases, balanced between routine cases and uncertain compartment/crop cases. Do not send reports or images to anyone without a separate instruction. There is no assumed radiologist resource in the compute budget, and this review did not obtain one.

**The last gold-58 look:** spend it on October 20 only after models, preprocessing, weights and windows are locked. One batch evaluates four predeclared composites: Ens5 incumbent, accepted new own-model ensemble if any, existing E12 .30/.35/.35, and external-only .50/.50. Generate each component's predictions once **inside Kaggle**, including public weights, and reuse them only for those fixed composites. Expert labels remain outside all trainers. Record macro, per-label support, macro without Synovitis, paired deltas and uncertainty; do not inspect a per-case error list to make new fixes. No new grid, label rule, threshold or per-finding weight follows this look. Use it as a failure/domain-shift veto: a substantial negative paired change with a CI excluding zero, or clear pipeline failure, can revert to the incumbent. Small differences do not override the frozen choice. These 58 are small and publicly available; they are not a secret private test set, and others may already have selected recipes against them.

## What a top practitioner would change, and what to drop

1. **Prioritize information reaching the head.** Random-window attention is a clean unresolved test. Additional sequence information deserves explicit identity/position handling. A bigger encoder is less urgent than checking these two mechanisms.
2. **Measure complementarity, not just standalone rank.** E2 is a losing single and a useful ensemble member. N6 should test incremental ensemble AUC with fixed weights rather than discard every lower-scoring model.
3. **Limit late selection freedom.** Keep the validation labels fixed; count missing labels; repeat the winning comparison; freeze two blend endpoints. Do not search twelve per-finding weights on 58 studies.
4. **Read the actual public recipe.** The downloaded goodpjw release is `cnxt_v0_*`; `train.py` defaults to `labels_v0`. Four-source averaging is v0. Grade mapping and effusion-to-Synovitis imputation are a separate `v1` code path requiring an additional own-grade table. Their presence in the same file is not evidence they caused the released checkpoints' score. The dataset describes three fold checkpoints, not a proven single-checkpoint 0.929 result.
5. **Protect deployment early.** Test actual reader failure/timeout behavior on a diverse sample, not three visible studies. E12 ranks over each reader's successful subset, renormalizes weights per study, and may shrink Raptor windows under deadline pressure. That is a useful emergency output path, but not the same predictor as the fixed three-reader blend. Treat missing-reader patterns and adaptive-window activation as material model changes.

Disposition of A–I:

| Candidate | Decision |
|---|---|
| A | **Run first** as N1, preserving random sampling and matched primary evaluation. |
| B | **Narrow and test** as N2. No automatic four-reader improvement claim; no Synovitis fabrication; no gold-weight fitting. |
| C | **Run as N3**, with a matched six-slot attention bridge. Drop the transformer for this week. |
| D | **Run conditionally as N4**. Use original pixels, preserve anatomy, budget 112.5 GiB for the same full slice inventory. Do not default to a smaller centered crop or narrower slice span. |
| E | **Keep only differential LR in N5** with matched 16 epochs. Drop the simultaneous scheduler/EMA/top-3 changes. E6 did not establish that longer training alone is harmful because it also changed augmentation. |
| F | **Drop for this deadline.** Five teachers plus a student, missing-not-at-random cells and no untouched image reference create too much runtime and self-confirmation risk. Preserve it as a post-deadline research question. |
| G | **Fold the cheap positional part into N3.** Drop hemi-knee crops and new auxiliary losses until laterality/compartment errors have independent image review. |
| H | **One matched independent seed in N6.** No broad seed/backbone sweep. |
| I | **Only relocation of N4 after a timed pilot**, within the same hypothesis and ≤4 GPU-h; no separate Kaggle architecture branch. |

## Choose the two final submissions

**Selection 1: own models, conservative eligibility route.** Start with Ens5's five individual rank-percentile predictions at **0.20 each**. These are E45-12, E6-16, E2-12, E8-12 and E3-12; make an inference-ready exact E3-12 state from the saved last checkpoint rather than substituting `best.pt`. Verify every model on 20 nonexpert studies with the exact deployment pipeline. If Ens5 fails deployment/LB checks, keep the already-completed Ens4 at **0.25 each**.

If a new recipe passes N6, evaluate exactly one challenger:

`own_final = 0.70 × Ens5 + 0.15 × new_recipe_seed0 + 0.15 × new_recipe_seed1`

Each new term is its per-finding rank percentile. This means each original member contributes 0.14. Require the N6 ensemble gate and a completed own-model submission before choosing it. Do not insert an untested mixture of all N1–N5 winners. If no new recipe qualifies, use Ens5, subject to its real score; do not force another member in. These weights apply identically to all twelve findings. Eligibility is still governed by competition rules and declared pretraining/label provenance; “ours” is not a guarantee by itself.

**Selection 2: strongest completed public-reader blend.** Compare only:

- **P30**, the existing E12 kernel: Ens4 **0.30**, Raptor maxspan **0.35**, goodpjw three-fold reader **0.35**.
- **P00**, the external-only comparator: Raptor **0.50**, goodpjw reader **0.50**.

Keep the exact public-reader preprocessing and source versions. Select the higher completed public-LB score when the gap is ≥0.005, subject to the frozen gold failure veto and acceptable runtime/coverage. If the gap is smaller, my default is P00: our last measured own LB is far below the reference readers' author-reported scores, so a 30% allocation needs evidence. **This 0.005 is a conservative decision threshold, not a measured standard error or guarantee about private ranking.** If only P30 completed successfully, use it rather than an unsubmitted comparator. Do not spend the remaining days searching intermediate weights or twelve label-specific mixtures. Replacing Ens4 inside P30 with a new own ensemble would be a third hypothesis requiring another complete validation/submission; it is not silently approved by improved local CV.

Why challenge 30% ours? An independent participant reports that blending a 0.860 own reader into a stronger public stack at 20% reduced LB from 0.944 to 0.937. That is an analogy, not a prediction of our result, but it directly refutes “diverse always helps.” [Independent implementation and reported experiments](https://github.com/sinhaaditya5/kaggle-rsna-knee-abnormality).

The prompt's approximately 70% private-test fraction is a planning assumption here; the relevant official page did not return readable text. No available method guarantees the private ordering. Broadly supported gains, different model families, fixed weights, and full coverage are more defensible than a small public-LB edge obtained through many trials. **Do not promise 0.93–0.95 for either blend before measuring it.**

## Biggest risk and hedge

**The biggest avoidable score risk is assuming that a proxy-validation improvement makes our reader worth 30% of a much stronger external blend.** Our model can be diverse and still dilute it. Protect a completed own-model entry, submit the existing E12 early, compare the external-only endpoint once, and retain a seven-hour repair reserve. Runtime/failure handling is the operational prerequisite: no AUC argument rescues a notebook that times out or changes reader coverage across the cohort.

My expectation is incremental progress for our own models, with **attention the best first modeling bet**. The strongest route to a much higher absolute score is likely the public-reader submission, but its actual score, complementarity and private generalization are **unknown**. Genuine pixel detail and sequence identity are worth testing; an elaborate pseudo-label pipeline is not the best use of this final week.

## External evidence and provenance

Sources below motivate tests; author-reported scores are not independent replications. Broad community numbers in the supplied prompt—CV/LB correlations 0.87/0.66, public-split SD 0.003, a −0.020 post-processing penalty and a strict report-label ceiling—were not independently established in this review and were not used as precise constants.

| Primary source checked | What it supports | Limit |
|---|---|---|
| [Ilse et al., attention MIL](https://arxiv.org/abs/1802.04712) | Learnable permutation-invariant instance aggregation; motivation for N1. | Not an RSNA-knee experiment and no predicted gain here. |
| [Public window trainer](https://raw.githubusercontent.com/mathischmp/challenge_rsna/main/src/rsna_knee/window_train.py) and [model/preprocessing implementation](https://raw.githubusercontent.com/mathischmp/challenge_rsna/main/src/rsna_knee/raptor.py) | Concrete random-window, per-finding attention and differential-LR implementations. | Independent adaptation; reference score is reported, not reproduced here. |
| [Independent pipeline README](https://raw.githubusercontent.com/mathischmp/challenge_rsna/main/README.md) | Reports failed equal-reader fusion and a loss from replacing inference views; useful counterevidence to blindly copying B or dense inference. | Observational results with changed recipes, not our matched ablations. |
| [goodpjw reader dataset](https://www.kaggle.com/datasets/goodpjw2008/rsna-knee-2-5d-convnext-reader) | Live API metadata: Apache-2.0, three fold checkpoints, 0.929 standalone-reader claim; actual `knee.py`, `make_labels.py`, `preprocess.py`, `train.py` inspected. | Dataset/code provenance is not a legal ruling on prize eligibility. Public checkpoint weights stayed in Kaggle. |
| [Independent ConvNeXt-style implementation](https://raw.githubusercontent.com/sinhaaditya5/kaggle-rsna-knee-abnormality/main/src/knee.py) | Checkable sequence/position and finding-specific attention implementation. | Shares lineage with goodpjw; not independent evidence that each component helps. |
| [Participant's original report-label discussion](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/743148) | Participants report that better gold extraction can fail to improve image LB and that labels are not an absolute ceiling on student performance. | Anecdotal, not a controlled study; no universal ceiling inferred. |
| [Official competition](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview) and [code requirements](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview/code-requirements) | Authenticated read-only CLI reconfirmed deadline, entry and our five completed scores. | Browser pages did not expose readable requirements; nine-hour/two-selection rules remain based on the project's prior official checks and supplied constraints. Recheck before deployment. |

Locally preserved public source: [external_evidence/goodpjw_reader_20261015](external_evidence/goodpjw_reader_20261015). Do not execute `make_labels.py` unchanged: it evaluates gold in its default main path, which would spend the reserved look. Only reimplement the audited nonexpert transformations inside the project's split guards.
