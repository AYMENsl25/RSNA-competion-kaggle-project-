# Research strategy for improving F0 — 2026-10-20

## Recommendation

Test one independent, geometry-aware local/global knee-MRI specialist, starting with frozen DINO features and a small trainable head. Evaluate its incremental value against the complete ensemble. Reserve metadata-guided self-supervised adaptation for a longer project. These are proposed experiments; no improvement over F0 has been demonstrated.

The original contribution would be our geometry handling, localized representation, learning objective and controlled evaluation. Using generic pretrained weights is compatible with an original downstream method. Neither DINO + Transformer nor attention MIL alone is a new scientific idea. This search does not establish publication-level novelty for the proposed combination.

## Evidence from this workspace

Sources inspected: `research/K1_F0_PUBLIC_WEIGHTS_TRACK.md`, `research/DECISION_LOG.md`, and code-cell source in `notebooks/f0_fork_0958/fork.ipynb`. Existing training, inference and historical result files were not modified. No raw reports, label tables, expert records, credentials, model weights or final test labels were opened for this research.

- The preceding CLI check confirmed F0 public AUC 0.958 and F1 public AUC 0.954. This is a decline of 0.004 at the displayed precision, with no private score or patient-level uncertainty estimate.
- F0 already combines several public readers, multiple resolutions, anatomical mirror TTA and per-finding rank weights. Its notebook also describes study-level Transformer processing with series and position information. Generic sequence modeling, multi-series input and finding-specific attention are therefore insufficient novelty claims.
- K1 fine-tunes were extremely similar to the Raptor parent (prediction rank correlations about 0.986–0.988 in the documented development check).
- K3-A was more different but substantially weaker. K4 did not add measurable development value; its higher encoder learning rate collapsed.
- S1-B passed the previous standalone-quality and parent-correlation gates, yet F1 fell on the public leaderboard. Those gates did not demonstrate incremental value against the complete F0 ensemble.
- Some other backbone arms failed numerically. They do not constitute evidence that those architectures cannot work.

Do not interpret prediction correlation as error correlation. Two models can have highly correlated predictions and complementary errors, or low correlation because one is weak. Evaluate the actual paired blend difference.

## Time constraint

The official competition overview's indexed timeline lists the final submission deadline as **October 22, 2026, 23:59 UTC** (October 23, 02:59 in Istanbul). The current client date is October 20. It also lists offline inference and a nine-hour notebook ceiling. This is the published schedule, not a live account quota check. Preserve any stricter project-specific budget until explicitly revised. [1]

The deadline favors one bounded pilot with reusable features. It does not justify starting a large 3D foundation model or downloading a new large dataset without a measured path to completion.

## What related problems actually show

| Primary source | Method and reported evidence | Implication and limitation |
|---|---|---|
| Medical Slice Transformer, 2024 [2] | DINOv2 slice features plus Transformer; knee meniscus-tear AUC 0.85 ± 0.04 versus 0.69 ± 0.05 for its 3D ResNet comparator; knee cohort 1,199 patients. | Direct knee-MRI support for this representation. Different data, endpoint and comparator; it does not predict a gain over F0. |
| RSNA 2024 lumbar spine, fifth-place author's repository [3] | Localization followed by crop-sequence classification; anatomy-level context and staged training. The author reports that larger backbones did not help. | Spend model capacity on relevant anatomy. That competition provided localization supervision, which we cannot assume here. No numerical gain attributable to one component is established by this README. |
| RSNA 2019 hemorrhage, first-place write-up/code [4] | Image encoders followed by sequence models, combining different image input constructions. | Preserving inter-slice information is a proven competition pattern. CT windowing and slice labels do not transfer directly to weakly labeled MRI studies. |
| Multicentre knee study, eClinicalMedicine 2025 [5] | Meniscus localization, global/local attention and separate abnormality classifiers across 14,962 examinations. Primary-abnormality AUC decreased from 0.898 internally to 0.852 and 0.812 on two external sets. | Relevant anatomy-first precedent, and evidence that distribution shift matters. Its segmentation and expert supervision are stronger than ours; no claim that its AUC is comparable to F0. |
| Deep AUC Maximization, ICCV 2021 [6] | Margin-based AUC objective; reported CheXpert ensemble test AUC 0.9305, versus 0.9299 for a listed competing system. | Ranking objectives have real medical-imaging evidence. The leaderboard comparison is not an isolated loss ablation; noisy weak labels can make pairwise losses worse. |
| LP-FT, ICLR 2022 [7] | Head fitting before full adaptation reduced feature distortion and improved out-of-distribution results across the paper's benchmarks. | Supports a frozen-head-first experiment. It neither diagnoses our fp16 failures nor proves low-LR adaptation will improve F0. |
| Medical PEFT study, 2024 [8] | LoRA exceeded full fine-tuning in 13 of 18 radiography transfer tasks, by up to 2.9 percentage points, using fewer than 1% tunable parameters. | Small adaptation is a credible second step. Chest radiography is not knee MRI; LoRA is easiest on compatible Transformer layers. |
| FINO, 2026 preprint [9] | Metadata-guided self-supervised domain adaptation, including chest X-ray evaluation, with lightweight downstream probes. | Interesting route to learning our own domain representation without noisy task labels. Knee-MRI transfer and compute feasibility remain untested. |
| Masked Hard Instance Mining MIL, ICCV 2023 [10] | Masking highly attended instances improves weakly supervised pathology learning in the authors' experiments. | Supports testing whether a model overuses one obvious region. Histology results are not knee-MRI evidence; masking can also remove the only visible lesion. |

A same-competition two-stage DINOv3/DeBERTa repository exists [11], but its README provides no verified leaderboard gain. It is architectural precedent, not evidence of beating 0.958. Its existence also rules out calling a plain two-stage DINO pipeline novel.

## Immediate experiment: geometry-aware local/global specialist

### Hypothesis

F0 may lose useful information through fixed series slots, spatial pooling and coarse spatial detail. A new representation could recover some of that information and produce useful ranking corrections. This is a hypothesis about information loss, not a confirmed error diagnosis.

### Minimum viable implementation

1. Use one generic self-supervised encoder, initially frozen: DINOv2 or a compact DINOv3 variant. Select one before the pilot; do not run a backbone sweep. DINOv3 has its own license and access process [12]; assess availability and competition compatibility before depending on it. Generic pretrained weights would remain part of the method.
2. Cache image features once. For the first ablation, keep existing image coverage fixed so the representation/head comparison is interpretable. Retain both global and coarse spatial feature tokens instead of immediately averaging everything into one study vector.
3. Preserve series identity, plane, physical within-series positions, sequence type and missing-series masks. Use patient-space geometry; normalized index alone does not describe unequal slice gaps. Do not assume different acquisitions align merely because DICOM coordinates exist. Use verified registration for cross-series spatial matching, or fuse without asserting alignment.
4. Add a compact hierarchical head: process windows within each series, then combine series with finding-specific queries. This architecture is precedent-based; the candidate contribution is its physical geometry and local evidence handling.
5. Test learned local selection with a bounded patch/window budget and a global branch. Retain uniform exploration so early attention cannot permanently exclude a lesion. Without localization annotations, describe regions as weakly selected, not anatomically or pathologically verified.
6. Keep the current weak-label source fixed for the representation test. Changing targets simultaneously would obscure the cause of any improvement. A report's silence or agreement among LLM sources does not establish a true negative or expert certainty.

A spatial feature crop cannot recover details already lost when the input image was resized. If selected high-resolution pixel crops are later added, treat that as a separate experiment and measure their encoding cost. Likewise, additional series coverage is a separate ablation, not a hidden change bundled into the new head.

### Controlled comparisons, in order

| Test | One comparison | What it resolves |
|---|---|---|
| P0 | Frozen encoder + simple pooling | Does the new encoder have useful knee signal at all? |
| P1 | Same features + geometry-aware sequence head | Does geometry/context help beyond the encoder? |
| P2 | P1 + bounded local feature selection | Does retaining spatial evidence help? |
| P3, only after P2 succeeds | P2 with BCE versus BCE + a modest ranking term | Does metric alignment add value? |

For P3, construct within-finding positive/negative pairs only from observed, sufficiently confident training targets. Use uncertainty weights and avoid promoting large weak-label disagreements into hard ground truth. A memory of cached features can supply pairs without putting many full MRI studies in GPU memory. Keep a BCE anchor. AUC-oriented loss is optional; it is not inherently robust to our label noise.

First fit the head with the encoder frozen. Only consider last-block or parameter-efficient adaptation after a useful signal and a throughput measurement. Do not repeat aggressive full-backbone updates as the default.

### Which findings?

The historical parent-specific development results suggest lateral OA, PF OA and lateral meniscus as hypotheses for a spatial specialist. They do **not** identify the weakest findings of full F0. Obtain aggregate full-ensemble diagnostics under an authorized protocol before choosing a target subset. Keep a shared representation and small per-finding heads; twelve full models would multiply cost without established benefit.

## How to add the specialist to F0

First evaluate ordinary per-finding blending against the complete stack. A learned correction is a later option:

`score_c(x) = base_c(x) + alpha_c * delta_c(x)`

Here the base is a fixed, explicitly defined score scale; it is not automatically the raw logit of F0's rank output. A correction can be shrunk toward zero so findings without supported gains retain the base. Any rank transforms, calibration and correction coefficients must be fitted inside the training/inner folds. Evaluate rank blending on validation as an entire cohort, consistently with deployment.

This resembles boosting/ensemble-aware learning. It is not a new algorithm merely because we apply it to knee MRI. A forced disagreement penalty can reward incorrect answers; target actual loss reduction and corrected positive/negative orderings instead.

**Critical constraint:** do not train a residual against F0 predictions on cases F0's public members trained on and call that out-of-fold learning. A frozen model can still leak training information. F0's parent also used the small expert set for historical selection according to our notes.

## Validation that is possible, and what remains missing

1. Keep patient/report/duplicate-connected groups together. Preserve existing frozen comparisons; any additional site/scanner stress split is supplemental and explicit.
2. For models initialized only from generic pretraining, keep the outer fold out of supervised encoder/head fitting, early stopping, local selector fitting and label-teacher fitting. Generic frozen features can be cached without task-label fitting. If the encoder is adapted, do so within each outer training fold.
3. Score the whole pipeline on outer folds. A stage-2 head must not use outer validation to choose its upstream encoder checkpoint. Separate train and validation feature-generation distributions where possible.
4. Report standalone macro AUC, per-finding AUC, and incremental ensemble AUC on the same cases. Pairwise corrected and newly broken orderings provide more relevant diagnostics than a correlation threshold alone.
5. Bootstrap by patient/duplicate group for paired differences and report support. After model selection, a confidence interval on the same tuning set is optimistic; use nested selection or a separate confirmation set.
6. A newly held-out training fold cannot make the already trained public F0 independent. Honest OOF comparisons can use a retrained surrogate ensemble with documented provenance; improvement there is still a proxy for improvement over deployed F0.
7. Weak-label validation estimates weak-label agreement, not official image-label AUC. The repeatedly used 58-study development set cannot serve as an untouched confirmation set. New independent labels/data with aligned definitions would be needed for stronger evidence.
8. Without that clean confirmation set or honest F0 OOF predictions, a single predeclared leaderboard experiment is a limited practical test. Do not claim a statistically established improvement from a rounded aggregate score alone, nor treat the previously asserted ±0.003 as a measured uncertainty bound.

The earlier suggestion to simply obtain a larger independent validation set was incomplete: access, aligned reference labels, and the public models' training provenance must be resolved. A new random split of the existing cases does not solve this.

## Longer research direction: metadata-guided knee representation learning

Adapt a generic self-supervised encoder on training MRI only, guided by plane, verified sequence characteristics and physical position. Then train a small diagnosis head. FINO [9] motivates this separation between representation learning and noisy target supervision.

Possible knee-specific contribution: preserve anatomy/protocol information while reducing acquisition-specific shortcuts, then test transfer across scanner/site groups. Treat nuisance suppression as an ablation: scanner and disease can be confounded, and erasing scanner information can also erase useful anatomy. Zech et al. [13] demonstrate this problem in chest imaging.

Do not make all slices or sequences of one study identical in feature space. Focal disease is not visible everywhere; cross-sequence positives need genuine anatomical correspondence. Do not use final test images for adaptation in this proposal.

This is the more ambitious research program. It needs multiple controls, reproducible preprocessing and measured training cost. It is not the best first commitment with the published deadline two calendar days away.

## Resource and decision gates

- This document authorizes no training, GPU run, notebook push or submission.
- Before a future pilot: verify offline weights/software availability, source hashes, geometry parity, distinct artifacts and provenance; measure feature extraction, head training and added inference time on a bounded sample.
- Calculate total deployment time as F0 + specialist + I/O + save reserve, including inference on the full hidden-set scale. A successful three-study commit does not establish hidden-set runtime.
- Use a fixed pilot stop rule and the currently approved GPU cap. Runtime estimates are unmeasured; no hours or expected AUC gains are promised here.
- Promote only a prespecified comparison that improves the relevant validation endpoint consistently and fits deployment limits. If evidence remains proxy-only, label it so.
- Preserve F0 as a final candidate while testing one credible alternative. Final submission selection remains a user decision.

## Primary references

1. [Official competition overview, evaluation and timeline](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview). Indexed text retrieved October 20; direct page extraction is incomplete.
2. [Medical Slice Transformer](https://arxiv.org/abs/2411.15802), 2024. Abstract-level numerical evidence; full HTML was unavailable.
3. [RSNA lumbar spine fifth-place author's solution](https://github.com/siwooyong/RSNA-2024-Lumbar-Spine-Degenerative-Classification), 2024.
4. [RSNA hemorrhage first-place write-up](https://www.kaggle.com/competitions/rsna-intracranial-hemorrhage-detection/writeups/seutao-1st-place-solution-sequential-model-wins) and [author code](https://github.com/SeuTao/RSNA2019_Intracranial-Hemorrhage-Detection), 2019/2020.
5. [Multicentre knee abnormalities study](https://pmc.ncbi.nlm.nih.gov/articles/PMC12508577/), eClinicalMedicine 2025; DOI 10.1016/j.eclinm.2025.103534. Search-indexed article text was available; direct PMC access presented a challenge.
6. [Large-Scale Robust Deep AUC Maximization](https://openaccess.thecvf.com/content/ICCV2021/papers/Yuan_Large-Scale_Robust_Deep_AUC_Maximization_A_New_Surrogate_Loss_and_ICCV_2021_paper.pdf), ICCV 2021.
7. [Fine-Tuning Can Distort Pretrained Features](https://arxiv.org/abs/2202.10054), ICLR 2022.
8. [Less Could Be Better: PEFT for Medical Vision Foundation Models](https://arxiv.org/abs/2401.12215), 2024.
9. [FINO: Who Needs Labels?](https://arxiv.org/abs/2606.05107), 2026 preprint; [HTML](https://arxiv.org/html/2606.05107v1). Method/abstract support; no knee-specific result established.
10. [Masked Hard Instance Mining MIL](https://openaccess.thecvf.com/content/ICCV2023/papers/Tang_Multiple_Instance_Learning_Framework_with_Masked_Hard_Instance_Mining_for_ICCV_2023_paper.pdf), ICCV 2023. Indexed paper text; direct PDF fetch failed.
11. [Same-competition two-stage research baseline](https://github.com/mmatmath/rsna-knee-two-stage). Architecture precedent with no verified leaderboard gain in the inspected README.
12. [Official DINOv3 repository](https://github.com/facebookresearch/dinov3) and [license](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md).
13. [Zech et al., cross-hospital generalization](https://pmc.ncbi.nlm.nih.gov/articles/PMC6219764/), PLOS Medicine 2018.

Additional source considered: [official SKM-TEA dataset documentation](https://github.com/StanfordMIMI/skm-tea/blob/main/DATASET.md?plain=1). Its documented acquisition/download burden and protocol differences make it a poor immediate dependency. No dataset was downloaded. Generic repositories without demonstrated results were not treated as proof of performance, and a repository that echoed our project history was excluded as potentially non-independent evidence.
