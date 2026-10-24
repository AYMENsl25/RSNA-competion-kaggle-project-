# Post-competition comparison: 21st and 22nd place

Read October 24, 2026. Primary sources were read in the browser because ordinary
web extraction could not retrieve them. Claims below are attributed to their
authors; no model weights, raw reports, per-study labels or expert records were
opened. Our final submission scores were checked with the Kaggle CLI.

- [21st-place solution, Driz Lo](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/writeups/21st-place-solo-gold-solution-llm-soft-labels)
- [22nd-place solution, matsurih](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/writeups/22nd-place-solo-gold-solution)

## Did they use public weights?

21st: yes, an explicit 10% probability contribution from nartaa's public
ConvNeXt-Small competition weights. The remaining 90% consists of two five-fold
ensembles the author trained: 45% continued on full-coverage LLM soft labels,
45% retrained from ImageNet initialization on self-distillation targets. The
architecture/preprocessing were based on heliosli's public multiview code.
These are 11 models in total. The independently trained models use generic
ImageNet-22k -> 1k ConvNeXt-Small initialization. Public code reuse, generic
pretraining and reuse of competition-trained checkpoints are different things.

22nd: explicitly reports no weights from other participants and no external
training data. All 56 diagnostic networks were trained by the author. However,
this is not training every component from random initialization: the resource
section names timm ImageNet-pretrained CoAtNet/ConvNeXt/ResNet backbones and a
pretrained TotalSegmentator MRI nnU-Net femur segmenter. The opening phrase
"no public weights" should be understood with these qualifications. It does
not mean the solution had no public pretrained components.

## 21st-place method

The main intervention was supervision. A remote image-reading tool exposed all
series and paginated slices, with optional native-resolution inspection/crops,
to GPT-6 Pro. Reports were supporting evidence; the competition's image-positive
criteria governed the probabilities. An initial rule forcing probability 1 for
report-confirmed findings was removed. Every training study eventually received
12 soft probabilities, allowing null for genuinely unassessable findings.

The author reports 58-study label AUC 0.89576 for four-source report consensus
and 0.93493 for image-first LLM probabilities. More persuasive evidence is the
matched ConvNeXt-Small comparison at epoch 16: OOF AUC against the LLM reference
0.93862 -> 0.95957 and expert-set AUC 0.91472 -> 0.94282 when training labels
were replaced. The evaluation reference is still weak; the small expert set was
used repeatedly. This is author-reported evidence, not an independently rerun
ablation.

Training used a multiview 2.5D ConvNeXt-Small, 384-pixel cache/352-pixel crops,
approximately 3.5-mm neighboring-slice offsets, up to 176 sampled training
windows per study, denser inference, class-specific spatial/window attention,
five grouped folds, soft BCE, separate encoder/head learning rates, warmup and
cosine decay, BF16 and gradient checkpointing.

Self-distillation targets were 75% LLM probability and 25% teacher probability.
Importantly, the teacher predicted its own fold's training partition: this was
in-sample self-distillation, not OOF target generation. The student's validation
fold remained held out, and validation retained original LLM references.
Continuation with distillation improved OOF AUC only 0.000477 versus a matched
two-epoch LLM-only continuation. Full student retraining had worse local metrics
but public 0.963 versus 0.962 for the base/continuation arms. The final fixed
45/45/10 probability blend is reported as the medal-winning entry; the write-up
does not state its final private AUC, so no exact private comparison is supplied.

The author used eight RTX 3090 GPUs on a shared server for independent folds and
experiments. Their claim of virtually free annotation describes their particular
LLM-client setup; it does not establish free GPU training or universal access.

## 22nd-place method

Reports public 0.965/private 0.954, rank 22/6,425. The inference ensemble has
56 networks in 12 groups, including all-plane models, separate plane models,
four compartment-model groups, ACL crops and a joint-ROI model.

1. Acquisition attributes: derive sequence weighting, fat suppression, echo time,
   obliquity and 2D/3D from headers plus the host table, with a limited fallback
   for missing metadata. Audit vendor quirks and mistaken names. Feed these
   attributes to the model. Select up to eight series for coverage of distinct
   acquisition types. Their final header-aware version had the same displayed
   public/private scores as the preceding version, so a separate leaderboard
   benefit of that last metadata change is not established.
2. Anatomy: run the femur segmenter, extract condyle/notch/joint-line/trochlear
   landmarks, and warp 64-mm crops from native DICOM pixels. A shared classifier
   processes one compartment at a time, mirroring the lateral side to a common
   frame. It cannot view opposite-compartment disease. This is materially
   different from feeding both approximate left/right tiles to one study head.
   Compartment classifier 4 contributes 74% of final lateral-OA prediction weight.
3. Supervision: use image readings for ten findings; retain report-based lateral
   OA labels; use a separate image-grading procedure for synovitis. Blend these
   with honest five-fold OOF predictions, 50/50. Confidence weights scale the
   loss. The author retained true labels for the 58 expert rows in training.
4. Models: use a residual window Transformer whose output and metadata embeddings
   are initially zero, preserving the starting pooling model; train separate
   plane and anatomical specialists to create genuinely different views.
5. Validation: tune on 4,342-study grouped OOF predictions against the image-read
   reference; predeclare acceptance thresholds and assess seed noise. The author
   says the 58 cases were diagnostic, not the model-selection metric. Their
   image reference still has systematic errors and does not replace an
   independent clinical reference.
6. Ensemble: probability averaging within/across groups, per-target weights
   chosen using OOF predictions and hill climbing/manual adjustment. This differs
   from 21st place's successful fixed global weights; the two solutions do not
   support a universal claim that per-target weights are always harmful.

Their score progression attributes 0.952/0.940 -> 0.958/0.947 to adding joint-ROI
and compartment models. Later steps bundle multiple changes, so the full private
gain cannot be attributed to one technique. The author reports that similar-view
head/loss/backbone variants generally did not help the ensemble; different plane
and crop views did. Training used RunPod RTX 4090s, up to 24 concurrently; total
GPU-hours/cost are not supplied. Inference decoded each study once, shared inputs
and split networks across two T4s within nine hours.

## What our records show

| Dimension | Our actual work | Gap exposed by these write-ups |
|---|---|---|
| Labels | Early hard/masked report targets; later soft BCE, report consensus and T1 OOF-denoised labels; no complete image-first annotation program | Better target semantics and full image-reading coverage were never established |
| Compartment models | K3/K4 fixed fractional tiles, resized from the existing crop, with tiles pooled together | No segmentation-grounded native-pixel crops or enforced one-compartment-only classifier |
| Distillation | K1: 50% converged parent target plus warm start; T1/S1: fresh ConvNeXt and OOF-denoised targets | Distillation was tried; our implementation did not reproduce 21st's new image supervision, multi-view coverage and five-fold final ensembles |
| Validation | Early fixed proxy split; later repeated small expert diagnostics and parent correlation/quality gates | No honest full-F0 OOF experiment; correlation and a small-set gain did not establish a good 20% full-stack blend |
| Coverage | S01 multi-series within a fixed budget; F2 geometry/context head; mature public inference inherited in F0 | Our own models did not develop the same comprehensive label/series/anatomical-specialist pipeline |
| Final experiments | F3 weight shrinkage; F5 reader removal; F6 Gaussian ranks; F7 squared ranks | These reweighted existing evidence and produced no gain at displayed private precision |
| Compute | Local laptop plus limited Kaggle allowance | Five-fold model families and iterative relabel/retrain cycles had substantially more resources in both write-ups |

K3's code stacks both compartment tiles into the window bag
(`notebooks/k3_compart/train_ft.py`, `compart_views`). The historical K1/K4/T1/S1
record is in `research/K1_F0_PUBLIC_WEIGHTS_TRACK.md`. We did recognize labels,
anatomy and OOF denoising; describing all three as ignored would be inaccurate.
Several early numerical failures also consumed budget before useful comparisons.

Final CLI scores, displayed precision:

| Submission | Public | Private |
|---|---:|---:|
| F0 public stack | 0.958 | 0.942 |
| F1 + S1-B student | 0.954 | 0.938 |
| F3 weight shrinkage | 0.957 | 0.942 |
| F5 no reader | 0.958 | 0.942 |
| F6 Gaussian ranks | 0.958 | 0.942 |
| F7 squared ranks | 0.958 | 0.942 |

F7 was already present in the account; this analysis did not create it. Equal
rounded scores do not imply identical predictions or unrounded AUCs. 22nd place's
reported private 0.954 exceeds F0's displayed 0.942 by 0.012. Our late changes
did not demonstrate improvement over the inherited public ensemble.

## Lessons for the next competition

Prioritize a bounded image-first label-quality pilot, preserving probabilities,
then establish independent grouped OOF predictions for every newly trained
model family. Build one strong stable multiview baseline and one anatomical or
plane specialist; compare actual paired ensemble gains on an untouched group
split. Audit acquisition/laterality and training/inference parity early. Use
compute measurements before committing to many folds; their entire ensemble
scale is not a realistic direct prescription for our former four-hour allowance.

I would change the late strategy: after F1 regressed and K1 produced near-clones,
make further fusion tweaks explicitly low-priority and spend development effort
on supervision and a structurally distinct view with an executable confirmation
plan. Neither write-up proves that another coefficient search would have closed
our gap. Future top-ten write-ups may change the ranking of hypotheses; these
two alone do not establish what the top ten did or which factor caused their rank.

Only this postmortem file was added during this analysis. No training, inference
job, upload or submission was launched.
