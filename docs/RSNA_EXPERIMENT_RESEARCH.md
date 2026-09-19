# Research decisions for the next RSNA experiments

Prepared 2026-09-14. This document distinguishes the project's historical observations, verified primary research, and proposed hypotheses. No new Kaggle score has been obtained by creating this suite.

## What the evidence actually says

The fresh tiny CNN ran on 120 training and 120 validation studies, with only 4–47 supervised examples per validation target in the supplied screenshot. Its best reported masked BCE was about 0.581. ACL/MCL AUCs were undefined. That run is a speed/functionality reference. It is not numerically comparable to an average per-target BCE, a new cohort, or the historical gold AUCs.

The local [experiment ledger](handoff/RSNA_Knee_Project_Handoff_Complete_2026-09-09/RSNA_Knee_Project_Handoff_2026-09-09/EXPERIMENT_LEDGER.md) records report-teacher AUC 0.76067, corrected ResNet18 MRI AUC 0.699358 and six-slot attention MRI AUC 0.644729. It explicitly records that lower pseudo loss did not yield better gold ranking. These are historical descriptions, not untouched evaluation results. The ledger supersedes suggestions to repeatedly use the same gold set as a go/no-go gate.

The [public notebook audit](handoff/RSNA_Knee_Project_Handoff_Complete_2026-09-09/RSNA_Knee_Project_Handoff_2026-09-09/PUBLIC_094_NOTEBOOK_AUDIT.md) found the two supplied public notebooks had matching principal inference stages. It documents one ensemble family and a source-displayed 0.940, not two independent replications or our achieved score. Original source notebooks are not present in the handoff; this review relies on the saved audit and provenance, not a fresh source equivalence check. The Kaggle evaluation webpage returned no readable body during this review; current leaderboard standing and detailed competition rules have not been independently verified here.

## Paper-by-paper assessment

| Source | What is supported | Experiment and limitation |
|---|---|---|
| [MRNet — Bien et al., 2018](https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.1002699) | Knee MRI can be classified by combining slice/series evidence with pretrained CNNs. | `r00` pools 2D encoder features from three planes; `r02` tests a sagittal-only restriction. This is inspired by MRNet, not a reproduction of its AlexNet/series-fusion pipeline or its three diagnostic tasks. |
| [Attention MIL — Ilse et al., 2018](https://proceedings.mlr.press/v80/ilse18a.html) | Learned, permutation-invariant aggregation can weight instances within a labeled bag. | `r01` uses target-specific slice attention instead of mean/max pooling. Its head is our extension, not the paper's exact architecture; prior project failure means attention must earn its complexity. |
| [DINOv2 — Oquab et al., 2023](https://arxiv.org/abs/2304.07193) | Self-supervised pretraining provides transferable visual features. | `m01` freezes a generic ViT-S/14 and trains a study head. Generic image pretraining is not knee-specific training. MRI fine-tuning, patch-local pooling and feature precomputation are later extensions. |
| [Focal loss — Lin et al., 2017](https://arxiv.org/abs/1708.02002) | A focusing factor reduces the contribution of easy examples in imbalanced detection. | `l02` tests MONAI sigmoid focal loss with gamma 2 against masked BCE. Transfer to sparse, noisy report labels is a hypothesis; hard examples can be labeling mistakes. |
| [Asymmetric loss — Ridnik et al., 2021](https://openaccess.thecvf.com/content/ICCV2021/papers/Ridnik_Asymmetric_Loss_for_Multi-Label_Classification_ICCV_2021_paper.pdf) | Separate positive/negative focusing and negative probability shifting address multilabel imbalance. | `l03` tests gamma-positive 0, gamma-negative 4, clip .05 with annotation masks. We do not convert unknown findings into negatives; this is a masked adaptation, not an exact benchmark reproduction. |
| [MONAI — Cardoso et al., 2022](https://arxiv.org/abs/2211.02701) | Medical-imaging transforms, network components and training utilities extend PyTorch. | The suite actually uses MONAI `FocalLoss`, `RandAdjustContrast`, and a compact 3D `DenseNet`. A framework choice by itself is not evidence of an accuracy gain. |

The implementation APIs were checked against [MONAI 1.5.1 DenseNet](https://raw.githubusercontent.com/Project-MONAI/MONAI/1.5.1/monai/networks/nets/densenet.py), [FocalLoss](https://raw.githubusercontent.com/Project-MONAI/MONAI/1.5.1/monai/losses/focal_loss.py), [intensity transforms](https://raw.githubusercontent.com/Project-MONAI/MONAI/1.5.1/monai/transforms/intensity/array.py), and [torchvision ResNet18](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html). MONAI 1.5.1 is the local test version, not a claim that it is the newest release. Installation cells preserve existing environments and record their actual versions.

## Highest priority: labels and validation

[RSNA describes reports in a dozen languages across 19 sites](https://www.rsna.org/artificial-intelligence/ai-image-challenge/knee-mri-ai-challenge). Our fresh rules match English phrases. They can miss non-English evidence and explicit normal statements; short-range negation can also misattribute findings. The first notebook exports a target/status review sample and flags absent positive/negative classes. New model training cannot repair these systematic labeling errors.

For the immediate comparisons, preserve the fresh labels and folds as a named control. Apply any reviewed training-label corrections through the explicit correction CSV, create a new supervision version, and keep validation labels fixed. To build a better multilingual teacher later, create a separate development/audit set before tuning extraction: test negation, uncertainty, compartment attribution, synonyms, translated reports and normal findings, then freeze that teacher. That larger labeling project is not implemented by these model experiments. A properly adjudicated evaluation set is still needed for stronger claims.

The normalized-report-hash fallback groups identical reports, but does not establish patient independence. The readiness notebook reports this limitation and checks for report collisions between known groups. Real patient linkage or a verified group map should be audited before a final evaluation round. No historical gold labels are read by this suite.

## Experiment matrix

Every model/loss/hyperparameter comparison first uses the same `p0` cache, cohort, validation labels, seed and selection criterion. Do not launch the Cartesian product.

| Family | Selectable settings | Main question |
|---|---|---|
| Preprocessing | `p0` full FOV/percentile; `p1` 130-mm crop; `p2` 150-mm crop; `p3` series z-score; `p4` CLAHE; `p5` central slice band; `p6` laterality canonicalization | Does physical context, contrast, depth coverage, or side normalization help? |
| Evidence/model | `r00` ResNet18 mean/max; `r01` attention; `r02` sagittal only; `r03` 16 training slices; `m01` frozen DINOv2; `m02` compact MONAI 3D | Is useful information being lost by the representation or pooling? |
| Strategy | `r04` frozen encoder; `r05` two-epoch frozen warm-up then unfreeze; `r06` shared MONAI intensity augmentation | How much adaptation can the weak labels support? |
| Loss | BCE control; `l01` capped training-derived positive weights; `l02` focal; `l03` asymmetric; `l04` BCE + .1 pairwise softplus ranking; `l05` .05 label smoothing | Which loss is robust to imbalance and imperfect labels? |
| Hyperparameters | `h01` lower LR; `h02` weight decay .001; `h03` 16-epoch budget | Is the control undertrained or overfitting? |
| Engineering | `s00` two-epoch tiny CNN | Does the complete cache/training/export path execute? |

The ResNet control uses AdamW, head LR 3e-4, encoder LR 3e-5, weight decay 1e-4, dropout .2, eight training/16 validation slices, batch size two with four-step accumulation (effective eight studies), cosine decay, gradient clipping and CUDA mixed precision. Runs use eight epochs maximum, minimum four before early stopping and patience three. These are initial hypotheses, not fitted optima. Ranking pairs are computed within each microbatch; accumulation does not create additional ranking pairs. Prefer testing `l04` with a genuinely larger batch if sparse positives give few pairs, and rerun a matched BCE control.

The MONAI 3D branch uses 16 depth positions and a small randomly initialized DenseNet. It processes each plane's stack independently, then combines plane features. It is not an isotropic physical volume and not a pretrained MedicalNet model. Compare it with `r03` (same 16 anchors), acknowledging changed initialization/capacity. DINO and MONAI are architecture-family comparisons, not strict one-factor parameter changes. Cropping may fail if required spacing or sufficient field of view is absent; failures must be resolved consistently across recipes, not silently removed from a cohort.

## Run order and promotion gates

1. Run readiness and inspect target/status examples, coverage and grouping warnings. Correct training-only errors if reviewed evidence is available; freeze the named control.
2. Run a 12-study `p0` timing pilot. Inspect all planes, geometry/masks/errors and extrapolate elapsed time and disk to the fixed cohort. A pilot is explicitly ineligible for training.
3. Build the full fixed cohort `p0` cache. Run `s00` if the environment is new; then `r00` and `r04`.
4. Test `p1` and `p3` with exactly the same `r00` settings. Add `p2`, `p4`, `p5` only if the pilot supports their data assumptions. Keep the whole cohort identical.
5. On the chosen named cache, compare BCE with `l01`, `l02`, `l03`; then try attention, dense anchors, staged unfreezing and intensity augmentation individually. Reserve ranking for batches with adequate positive/negative pairs.
6. Compare the frozen DINO and MONAI branches only after the control is usable. Try the small LR/decay/budget alternatives around a promising family rather than running every combination.
7. Expand to all eligible data (`max_train=max_val=0`) with a new cohort and rerun the control. Confirm leading recipes on another frozen fold/seed; assemble full OOF predictions in a later phase before tuning a final ensemble.

Promotion requires better unchanged validation metrics than both the matched control and the training-prior baseline, adequate per-target evidence, and no new data failures. Log failures and negative results too. AUC is reported only for targets with supervised positive and negative examples; absent targets are never assigned 0.5 or silently passed off as 12-target AUC. Validation BCE is always plain masked per-target BCE, regardless of training loss. AUC checkpoint selection is gated on all 12 classes having valid coverage; otherwise select by BCE and label the results exploratory.

Run comparisons use identical validation data and group-bootstrap paired differences. Broad or unstable intervals mean there is insufficient evidence to promote. Intervals from repeatedly selected validation results do not undo adaptive selection bias. Equal-probability ensembling is available as a diagnostic; it must be confirmed independently. There is no score guarantee.

## Deferred research, not claimed implemented

Exact public 0.940 reproduction, a trained six-slot fluid/T1 model, focal patch pooling, isotropic resampling, denoising/bias-field correction, pretrained MedicalNet/RadImageNet/CoAtNet, OOF rank stacking and final competition submission remain separate work. A header-only six-slot feasibility audit and conservative left/right canonicalization are now implemented as gates; neither is evidence of a score gain. The preprocessor otherwise prefers explicit fluid/FS evidence within three anatomical planes, which is not the complete public ensemble. Dice/Tversky losses are not included because this task supplies study labels rather than voxel segmentation masks.

## Architecture and acceptance evidence

The SPARC workflow fixed input/output contracts first. Separate workers implemented readiness, MRI caching and result comparison; the main implementation supplies registry, model/loss/training code and standalone notebook generation. Each stage embeds the same source package and hashes its artifacts. Acceptance checks cover unknown-label masking, missing-slice masks, constant-prior metrics, resume state, geometry/crop errors, cohort preservation and matched comparison. Actual Kaggle/GPU experiment results must be entered after execution; see the notebook README for the current validation record.
