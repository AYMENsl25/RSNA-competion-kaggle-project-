# RSNA Knee 2026: toolkit and methods report (5 days, ~30 T4 GPU-hours)

Date: 2026-10-17. Deadline: 2026-10-22. Our LB: 0.857. Strong public notebooks: ~0.958. Top LB: 0.969.

Legend: [V] = verified from a source in this session. [J] = my judgement or prior knowledge, not verified here.

---

## 0. Bottom line, ranked by expected value

| # | Action | Cost | Expected gain | Why |
|---|---|---|---|---|
| 1 | Reuse the strongest public notebooks' checkpoints, if the license and rules allow it. Rank-average them with ours, per finding. | ~0 training, 1-3 h inference | Large. This closes most of the 0.10 gap. | The 0.10 gap is about backbones, checkpoints and inputs, not labels. Our model is useful mainly as a diverse ensemble member. |
| 2 | Distill: use public-model OOF/test-free predictions on the 4.4k training studies as soft targets, mixed with report labels, and retrain our MIL student. | 6-10 GPU-h | Medium-large | Turns systematic report noise into the teacher's image-grounded signal. It is the cheapest form of label cleaning. |
| 3 | Input fixes: route findings to planes and sequences, and raise in-plane resolution to 384-448 px or a tighter crop. Add a DINOv3/ImageNet-22k ConvNeXt init. | 8-12 GPU-h | Medium | At 160 mm / 224 px = 0.71 mm/px, meniscal tears are sub-pixel to a few pixels. MRNet-style models use per-plane networks. |
| 4 | Label policy: align report extraction with the expert rule that "on the fence = negative" (severity-thresholded). Map equivocal or mild mentions to 0 or about 0.2-0.3 soft, instead of 0.5+. | CPU only | Small-medium, finding-dependent | The noise is systematic, not random [V homeshwarnelakurthi repo: about 82% agreement between report and image labels]. Random-noise methods (cleanlab, co-teaching) do not fix bias. |
| 5 | Ensemble hygiene: grouped folds by site or scanner, paired bootstrap on OOF, equal-weight rank average per finding, and at most a coarse per-finding weight grid. | CPU | Avoids losing 0.005-0.01 on private LB | Random K-fold inflates AUC by about 0.05 here [V same repo]. |
| 6 | MONAI | n/a | Near zero for a 2.5D timm pipeline | The only possible use is MRI intensity augmentations. See section 1. |

Do not do in 5 days: MONAI Label, Auto3DSeg, training nnU-Net meniscus/ACL segmenters, 3D foundation models (Triad, KneePreM, MedicalNet), SAM-based ViT at 1024 px (MRI-CORE) on a T4, or cleanlab/co-teaching rewrites.

---

## 1. MONAI: what is worth using

**Version** [V]: The latest release is MONAI 1.6.1 (27 Sep 2026). 1.6.0 came out on 11 Jun 2026 and 1.5.2 on 29 Jan 2026. 1.5.1 added PyTorch 2.7/2.8 support. 1.4 dropped Python 3.8. https://github.com/Project-MONAI/MONAI/releases

**Offline install on Kaggle** [J]: MONAI is a pure-Python wheel whose only hard dependencies are torch and numpy. Online, run `pip download monai==1.6.1 --no-deps -d wheels/` and upload the result as a Kaggle Dataset. In the notebook, run `pip install --no-index --no-deps /kaggle/input/<ds>/monai-1.6.1-*.whl`. Check first whether the Kaggle image already ships MONAI with `pip show monai`. Never let pip resolve dependencies, because that can change the image's torch.

**Component-by-component verdict for 2.5D classification:**

| Component | Verdict |
|---|---|
| `LoadImaged` / `Orientationd` / `Spacingd` | Useful only if you rebuild preprocessing. We already crop at a fixed physical size (160 mm). For a 2.5D pipeline, a per-series resample in pydicom/SimpleITK is equivalent and faster. The real risk is multi-series DICOM sorting, meaning ImagePositionPatient along the slice normal, and MONAI does not solve that better. Skip. |
| `ScaleIntensityRangePercentilesd` (for example 0.5-99.5 per series) | Good idea, but it is one line of numpy. Use percentile normalization per series. Do not use a z-score on the whole volume, which is dominated by background. |
| `RandBiasFieldd`, `RandGibbsNoised`, `RandKSpaceSpikeNoised`, `RandRicianNoised` | The only MONAI components with plausible value. They simulate multi-site MRI artifacts, and the data comes from 16 sites on 5 continents [V RSNA]. They run on the CPU in numpy/torch and can be called on a 2.5D stack. Expect at most +0.002-0.005. Bias field is the most useful of the four. Only add them if dataloader CPU time allows. The transforms have existed since MONAI 0.6 [V docs]. |
| `RandAffine` / `Rand3DElastic` | Albumentations or torchvision on 2.5D stacks covers this. No gain from MONAI. |
| `monai.networks.nets.MILModel` [V] | A 2D MIL wrapper with a ResNet50 backbone by default. Input is [B,N,C,H,W]. Modes are mean/max/att/att_trans/att_trans_pyramid. It was built for pathology. Our finding-specific attention MIL is already at least as good. Its `att_trans` (a transformer over instances) is a design we could borrow, but do not swap frameworks now. https://monai.readthedocs.io/en/1.2.0/_modules/monai/networks/nets/milmodel.html |
| SwinUNETR / ResNet3D / MedicalNet | 3D, slow, and weak evidence for knee classification. Skip. |
| Model Zoo / bundles | No knee-MRI bundle found [V search]. Skip. |
| MONAI Label, Auto3DSeg | Need annotation and segmentation labels plus many GPU-days. Skip. |

---

## 2. Pretrained weights relevant to knee MRI

| Weights | Type | License | Downloadable | Evidence on knee MRI | Verdict for us |
|---|---|---|---|---|---|
| **DINOv3 ConvNeXt-T/S/B/L, ViT-S/B/L** (LVD-1689M) | 2D, natural images | Custom "DINOv3 License": royalty-free, use/redistribute with the license attached. Needs a legal or competition-rule check. [V] https://ai.meta.com/resources/models-and-libraries/dinov3-license | Yes, HF `facebook/dinov3-*`, plus timm ports [V] | No knee data. On chest X-ray, DINOv3 roughly matches DINOv2/ImageNet at 224 px but helps at 512 px, especially with ConvNeXt [V arXiv 2510.07191] | **Best drop-in.** ConvNeXt-Tiny DINOv3 (29M) uses the same architecture as our current backbone. Swap the init and raise the resolution to at least 384. |
| **DINOv2** ViT-S/B | 2D natural | Apache 2.0 | Yes | Mixed on MRI: a glioma study found DINOv2 below ImageNet on clinical MRI [V search summary] | Public notebooks use it. Good as a diverse ensemble member, not as a primary choice. |
| **ImageNet-22k ConvNeXt / CoAtNet (timm)** | 2D | Apache 2.0 | Yes | Used by public 0.95+ notebooks (per the brief) | Safe baseline. Prefer in22k-ft-1k weights over in1k. |
| **RadImageNet** ResNet50 / DenseNet121 / InceptionV3 / IncResNetV2 | 2D, 1.35M CT/MR/US slices including MSK MRI | MIT (GitHub) [V] https://github.com/BMEII-AI/RadImageNet ; HF mirror https://huggingface.co/Lab-Rasool/RadImageNet | Yes (Google Drive / HF) | The paper reports higher AUC than ImageNet on MRNet ACL and meniscus tasks, mainly for small fine-tuning sets. The gap shrinks with more data [V PMC9530758 summary]. | Old architectures. At 4.4k studies, expect parity with ImageNet ConvNeXt, not a gain. Worth one run only as a diversity member, and only if GPU time remains. |
| **MRI-CORE** (Duke, 2025) | 2D ViT-B (SAM init), DINOv2-style SSL on 6M MRI slices from 110k volumes, 18 body regions | Apache 2.0 [V] https://github.com/mazurowski-lab/mri_foundation | Yes (Google Drive) | Includes MSK/bone tasks and KNEEMRI linear probing. No ImageNet-ConvNeXt fine-tune comparison. | Native 1024 px ViT-B is heavy on a T4. Interesting, but too risky for 5 days. |
| **Triad** (3D, 131k MRI volumes) | 3D autoencoder | Check the repo (not verified) | Yes, GitHub wangshansong1/Triad [V] https://arxiv.org/abs/2502.14064 | Gains mostly for segmentation (+2.5% vs nnU-Net scratch) | Skip. |
| **KneePreM** (2026) | 3D U-Net MAE on 19k OAI knee series | Release not stated [V] https://arxiv.org/abs/2609.31461 | Unclear | Beats random init / SuPreM on fastMRI+ classification (macro AUC ~0.72 at 50% data). No ImageNet comparison. | Skip. Absolute numbers are low and it is 3D. |
| **OrthoFoundation** (DINOv3 backbone, 1.2M knee X-ray + MRI) | 2D | Unknown, no repo found | No [V search] | Preprint claims first place in MRI structural injury detection | Not available. |
| **MedicalNet** 3D ResNet10-50 | 3D, 23 seg datasets | MIT [V] https://huggingface.co/TencentMedicalNet/MedicalNet-Resnet18 | Yes | Weak evidence for knee classification | Skip. |
| **BiomedCLIP** ViT-B/16 | 2D, PMC figures | MIT label, but the model card says research-only and that deployment is out of scope [V] | Yes | Poor fit for MRI slices | Skip. |
| **MedImageInsight** (Microsoft) | 2D embedding | Paper CC BY 4.0. Weights are via Azure, with terms to check. | Azure | General-domain | Skip. |
| **MedSAM / SAM-Med3D** | Segmentation | Apache 2.0 | Yes | Segmentation-only. Would need prompts for ROI. | Skip. |
| Merlin, RadFound | CT / report-centric | n/a | n/a | Not knee MRI | Skip. |

**Competition rule check** [J, could not fetch the Kaggle rules page]: Kaggle usually allows external pretrained models that are "publicly available and free to all". The DINOv3 custom license and public-notebook checkpoints usually qualify, but read the competition's External Data thread and each notebook's license before relying on them. One public repo notes that Rule 4.b forbids sending report text to hosted LLM APIs [V homeshwarnelakurthi].

---

## 3. Knee-MRI tricks from the literature

**MRNet (Stanford, Bien et al., PLOS Med 2018)** [V]
- Data: 1,370 exams, one AlexNet per plane (sagittal, coronal, axial), slice features max-pooled, then logistic regression across planes.
- Internal AUC: 0.937 abnormal, 0.965 ACL, 0.847 meniscus.
- External (KneeMRI, Croatia): ACL AUC 0.824 with no retraining, 0.911 retrained.
- Lesson 1: meniscus is the hardest finding. Lesson 2: per-plane models plus late fusion is a strong, simple prior.
- https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.1002699

**MRPyrNet (MIDL 2021 / CMIG 2022)** [V]
- Adds FPN plus Pyramidal Detail Pooling because "anomalies are localized in small-sized knee regions".
- Improves ACL and meniscus results on MRNet and KneeMRI.
- Takeaway: the model needs higher-resolution feature maps, not just global pooling.
- https://github.com/matteo-dunnhofer/MRPyrNet

**Localization**
- A YOLO meniscus-localizer plus EfficientNetV2 tear classifier reached AUC 0.97 (sagittal) and 0.98 (coronal) on 642 knees, with no no-crop ablation [V PMC11792105].
- A U-Net study suggested cropping to the menisci would help, but did not test it [V].
- No public, ready-to-use nnU-Net knee meniscus/ACL weights were found [V search]. Panfilov/Tiulpin (Oulu) and SKM-TEA have OAI-DESS segmentation code, but DESS differs from clinical PD-FS.
- **In 5 days:** do not build a segmenter. A cheap proxy is a second, tighter fixed crop centred on the joint line (about 100-110 mm field of view, or the centre of the existing 160 mm crop) at 384 px. Feed it as extra MIL instances or as a second model. Alternatively, use our MIL attention to pick top-k slices and re-crop.

**Sequence and plane routing** (standard MSK practice) [J]

| Finding | Primary plane / sequence | Secondary |
|---|---|---|
| Medial/lateral meniscus | Sagittal PD (FS and non-FS) | Coronal PD-FS (fat-sat preferred for DL contrast [V Mask R-CNN study, cited in PMC11792105 search results]) |
| ACL | Sagittal PD/T2-FS | Coronal, axial |
| MCL | Coronal PD-FS/T2-FS | Axial |
| Medial/lateral OA (cartilage, osteophytes) | Coronal | Sagittal |
| Patellofemoral OA | Axial PD-FS | Sagittal |
| Effusion, synovitis | Axial and sagittal fluid-sensitive (FS) | n/a |
| Baker's cyst | Axial FS | Sagittal |
| Bone contusion, fracture | Fat-sat or STIR in any plane (marrow edema) | T1 for fracture lines |

Implementation: classify each series by orientation (from ImageOrientationPatient) and by fat-sat (from SeriesDescription/ScanningSequence keywords). Tag every MIL instance with an embedding for plane and sequence. Alternatively, give the finding-specific attention a learned prior over plane/sequence. This is cheap and usually worth +0.005-0.02 on plane-specific findings [J].

**Resolution** [J]: Clinical knee PD in-plane resolution is about 0.3-0.5 mm, and our pipeline uses 0.71 mm/px. Meniscus, ACL and MCL benefit most from 384-512 px. OA and effusion are fine at 224-256. DINOv3 ConvNeXt gains appear at 512 px on chest X-ray [V arXiv 2510.07191]. On a T4, ConvNeXt-T at 384 px with fp16 and channels_last is feasible.

---

## 4. Noisy report-derived labels vs expert image labels

**What we know about this dataset** [V homeshwarnelakurthi repo]
- The test set was labelled by 2 MSK radiologists plus an adjudicator, using severity thresholds, with "on the fence" graded negative.
- Report labels agree with these only about 82% of the time, and the disagreement is systematic.
- On the 58 gold studies, one text labeler scored 0.791 and a first vision model 0.674. Conclusion: vision capacity is the bottleneck.

**Methods by expected value:**

1. **Teacher distillation** (highest). Run the strong public checkpoints, or our 5-fold OOF ensemble, on all 4.4k training studies. Train the student on `y = a*report_soft + (1-a)*teacher_prob`, with a in roughly 0.3-0.5 per finding. This is the noisy-label method with the most competition evidence [J]. Teacher predictions are image-grounded, which matches how the test labels were made. Use OOF teacher predictions to avoid leakage when the teacher was trained on the same studies. Public checkpoints trained on all of train are acceptable in practice.
2. **Policy-level label correction**. The expert rule biases toward negatives on borderline cases, so recode report labels:
   - "mild / small / possible / cannot exclude / degenerative signal without tear" maps to 0-0.3.
   - "definite / complete / high-grade" maps to 1.0.
   - The CheXpert experience: the best uncertainty policy differs per pathology, and label-smoothed uncertain labels (U-Ones-LSR) beat hard 0/1 [V arXiv 1911.06475: 0.930 mean AUC on the expert test set]. https://arxiv.org/abs/1911.06475
3. **Label smoothing / soft labels** (ε about 0.05-0.1) with BCE. These are cheap and robust. LLM-confidence soft labels are fine where the LLM gave calibrated confidence.
4. **cleanlab / confident learning** (low). In retinal imaging its per-sample error scores had only 0.54-0.67 AUC for identifying true errors, which makes it a review-prioritization tool, not an auto-cleaner [V Research Square rs-10775714]. It assumes class-conditional random noise, not systematic severity bias. Optionally, down-weight (do not drop) the top 5% of OOF disagreements.
5. **Co-teaching / ELR / Co-Correcting** (low for 5 days). On medical benchmarks the gains are inconsistent: plain CE sometimes beats co-teaching and ELR on NIH nodule [V arXiv 2205.04723], and LNMBench finds most methods degrade under real-world noise [V arXiv 2512.09315]. Do not rewrite the training loop now.

**Using the 58 gold studies correctly:**
- **AUC is invariant to any monotone per-finding transform.** Temperature or Platt scaling per finding cannot change per-finding AUC, so macro-AUC is also unchanged. Calibration only matters when blending models with different score scales. Rank averaging removes even that need.
- With 58 studies and low prevalence, many findings have fewer than 10 positives. Per-finding AUC SE is about 0.05-0.10, and macro SE is about 0.02-0.03 [J, Hanley-McNeil]. It cannot select between options 0.002-0.01 apart.
- Use gold for:
  - (a) sanity checks such as catastrophic failures and inverted findings;
  - (b) one coarse decision per finding family, such as label policy A vs B or "teacher-heavy vs report-heavy", pooled across folds;
  - (c) the diagnostic of how well report labels match gold per finding, to set a per finding.
- Never fit per-finding blend weights on 58 studies.
- Also put the 58 gold studies into training at the end with hard expert labels and upweighting (x3-5) [J]. They are the only samples drawn from the test label distribution.

---

## 5. Ensembling and validation

**Folds**: Group by site/scanner (and patient if repeated). Random K-fold inflates AUC by about 0.05 here because pixels leak site information [V homeshwarnelakurthi]. If site is not given, cluster on DICOM Manufacturer, ModelName, InstitutionName and field strength.

**Blend**:
- Per finding, convert each model's predictions to ranks (`scipy.stats.rankdata / n`) and average them.
- Equal weights first. Then try a coarse grid (weights in {0, 0.5, 1, 2}) or greedy hill-climbing with replacement on OOF.
- Accept a non-uniform weight only if it beats equal weights in **nested** CV, meaning weights fit on 4 folds and scored on the 5th.
- With 12 findings × M models, per-finding hill-climbing on about 4k noisy-label OOF rows overfits to report noise. Prefer global weights, or per-finding weights shrunk toward global.
- Hill-climbing reference: https://endgame-ml.readthedocs.io/en/latest/guides/ensembles.html
- Mixing public checkpoints without OOF: the public models have no OOF on our folds, so use equal-weight rank averaging, with our model at weight 0.25-0.5 relative to the best public model. Check on gold and the public LB only for gross errors.

**Is 0.002 real?** [J, numbers from Hanley-McNeil]
- On ~900 test studies, a single finding with AUC 0.90, 100 positives and 800 negatives has SE ≈ 0.021. With 30 positives, SE ≈ 0.038.
- Macro over 12 roughly independent findings: SE ≈ 0.008-0.010 per model.
- The paired difference between two correlated models is smaller, typically 0.002-0.005 SE.
- So a 0.002 macro gain is roughly 0.5-1 SE: not distinguishable.
- Use a **paired bootstrap** over studies on OOF (resample study IDs, recompute macro-AUC for both models, 2,000 reps). Accept a change only if the 95% CI of the difference excludes 0, or if it is consistently positive across all 5 folds. DeLong is fine per finding, and the bootstrap handles the macro average [V pROC docs https://rdrr.io/cran/pROC/man/roc.test.html; MetricGate guidance].

**Public LB**:
- The public LB is a subset of an already small test set, so its macro SE is probably about 0.01-0.015. LB moves under about 0.005 are noise.
- Use the LB only for large decisions, such as "public checkpoint blend vs ours" (0.10 apart).
- Final two submissions:
  - (1) the robust blend: equal-weight rank average of the best public models plus our distilled student, chosen by CV and gold sanity checks;
  - (2) the best-public-LB variant, if it differs by more than 0.005.
- Do not pick between options 0.001-0.003 apart on LB.

**Inference budget** (9 h on a T4) [J]:
- Each 2.5D ConvNeXt-T model at 384 px over all series costs roughly 20-40 min for about 900 studies, depending on DICOM decode.
- Decode once, cache the arrays in RAM or /tmp, and share them across models.
- Budget 4-6 models plus a hflip TTA only.

---

## Sources
- RSNA challenge: https://www.rsna.org/news/2026/august/ai-challenge-knee-mri , https://www.rsna.org/media/press/2026/2669
- Competition analysis repo (gold set, 82% agreement, grouped folds): https://github.com/homeshwarnelakurthi/RSNA-Knee-Abnormality-Detection
- MONAI releases: https://github.com/Project-MONAI/MONAI/releases ; MILModel: https://monai.readthedocs.io/en/1.2.0/_modules/monai/networks/nets/milmodel.html ; intensity transforms: https://monai.readthedocs.io/en/1.1.0/_modules/monai/transforms/intensity/dictionary.html
- RadImageNet: https://github.com/BMEII-AI/RadImageNet , https://huggingface.co/Lab-Rasool/RadImageNet , https://pmc.ncbi.nlm.nih.gov/articles/PMC9530758/
- DINOv3: https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m , license https://ai.meta.com/resources/models-and-libraries/dinov3-license ; DINOv3 resolution study: https://arxiv.org/abs/2510.07191 ; DINOv3 medical benchmark: https://arxiv.org/abs/2509.06467
- MRI-CORE: https://arxiv.org/abs/2506.12186 , https://github.com/mazurowski-lab/mri_foundation
- Triad: https://arxiv.org/abs/2502.14064 ; KneePreM: https://arxiv.org/abs/2609.31461 ; OrthoFoundation: https://www.researchgate.net/publication/400083610
- MedicalNet: https://huggingface.co/TencentMedicalNet/MedicalNet-Resnet18 ; BiomedCLIP: https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 ; MedImageInsight: https://arxiv.org/abs/2410.06542
- MRNet: https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.1002699 ; MRPyrNet: https://github.com/matteo-dunnhofer/MRPyrNet , https://proceedings.mlr.press/v143/dunnhofer21a.html
- Meniscus localization + classification: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11792105/ ; 3D Mask R-CNN meniscus: https://pubmed.ncbi.nlm.nih.gov/36372871/
- nnU-Net knee generalizability: https://cds.ismrm.org/protected/23MProceedings/PDFfiles/2468_VvVh2ZQ8z.html
- CheXpert / uncertainty label smoothing: https://arxiv.org/abs/1901.07031 , https://arxiv.org/abs/1911.06475 , https://arxiv.org/abs/2508.02495
- Cleanlab retinal audit: https://www.researchsquare.com/article/rs-10775714/v1 ; co-training noisy-label medical: https://arxiv.org/abs/2205.04723 ; LNMBench: https://arxiv.org/abs/2512.09315
- LLM report labels (spine MRI, MICCAI 2024): https://papers.miccai.org/miccai-2024/087-Paper1510.html
- AUC comparison: https://rdrr.io/cran/pROC/man/roc.test.html , https://metricgate.com/docs/proc-bootstrap-auc-comparison/
