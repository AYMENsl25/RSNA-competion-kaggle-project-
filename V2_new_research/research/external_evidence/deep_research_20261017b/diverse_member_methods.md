# Adding ONE diverse member to a strong correlated ensemble (RSNA knee, 0.958)

Context: the public ensemble (CoAtNet-2 MIL plus ConvNeXt-S) scores 0.958. Our warm-start self-distilled clone has rank correlation 0.986 with its parent and adds 0 to the blend. The budget is about 23 T4-hours over 5 days. Some numbers below are my own derivations, not values from a paper; each one is labelled.

---

## 1. Theory: when does one new member help?

**Binormal / LDA view (derived; best for reasoning about AUC).**
- Model each member's score as Gaussian, with class separation `d` in SD units. Then AUC = Φ(d/√2).
- For an existing ensemble E (d_E) and a new member M (d_M) whose errors correlate with E at ρ, the best linear blend gives:

    d_blend² = (d_E² + d_M² − 2ρ·d_E·d_M) / (1 − ρ²)

- The gain is exactly zero when **d_M = ρ·d_E**. The gain grows like (d_M − ρ·d_E)² / (1 − ρ²).
- So a weaker member helps only if it is weaker by less than its correlation would imply. Lowering ρ is worth more than raising accuracy.
- This is the classification form of Krogh & Vedelsby (1995): ensemble error = mean member error − ambiguity. Rank averaging approximates the linear blend, and a learned weight comes close to the optimum.

**Worked numbers (derived).** E has AUC 0.958, so d_E ≈ 2.44. Take a new member with single-model AUC 0.940 (d ≈ 2.20):

| ρ (error corr. with the *ensemble*) | blended AUC | gain |
|---|---|---|
| 0.90 | 0.958 | 0 (this is the clone case) |
| 0.88 | 0.958 | ~0 |
| 0.75 | 0.9616 | +0.0036 |
| 0.60 | 0.967 | +0.009 |

- Our clone had prediction rank correlation 0.986, so its error correlation was about 0.97. A gain of zero is exactly what this formula predicts.
- Correlation with the *ensemble* is higher than correlation with any single member. Measure ρ against the blend, not against one model.

**Empirical evidence on what lowers ρ, strongest first.**
- **Different training methodology, pretraining data or objective decorrelates far more than a different seed or hyperparameters.** Gontijo-Lopes, Dauphin & Cubuk, *No One Representation to Rule Them All*, ICLR 2022 (arXiv 2110.12899):
  - The more two models' training methods differ, the less correlated their errors are.
  - Lower-accuracy but divergent models improve high-accuracy ensembles.
  - Changing the pretraining dataset gave the most diversity.
  - Changing the architecture family gave a moderate amount.
  - Changing hyperparameters, augmentation or seed within one method gave the least.
- **Independent random initialisations explore different modes.** Subspace or local sampling around one solution (SWA-style, dropout, small perturbations) stays functionally similar. Fort, Hu & Lakshminarayanan, *Deep Ensembles: A Loss Landscape Perspective*, 2019.
- **Fine-tuning from the same pretrained checkpoint stays in one basin.** The fine-tuned models are linearly mode-connected, which is why "model soups" work: Neyshabur et al., *What is being transferred in transfer learning?*, NeurIPS 2020; Wortsman et al., *Model Soups*, ICML 2022. The flip side is that fine-tunes of one checkpoint are poor ensemble partners.
- **Negative correlation learning** adds a penalty −λ·(p_i − p_ens)² so the member is pushed away from the ensemble (Liu & Yao 1999; Brown et al. 2005 survey; Buschjäger et al. 2020 "Generalized NCL"; Ross et al. AAAI 2020 "locally independent" ensembles).
  - It works, but the gains in deep nets are modest and λ needs tuning.
  - It needs out-of-fold (OOF) ensemble predictions on the training set. The public models were trained on this data, so their in-sample predictions are overconfident and would bias the penalty. Use it only with honest OOF predictions. Medium EV, and it costs tuning runs.
- **Boosting-style reweighting toward examples the ensemble gets wrong is negative EV here.**
  - Our training labels are noisy LLM report labels, so the ensemble's "errors" against them are largely label noise.
  - Convex boosters are provably defeated by random label noise (Long & Servedio 2010).
  - AdaBoost degrades under label noise (Dietterich 2000).
  - Residual fitting would mostly learn the LLM's labelling mistakes. Avoid it.
- **Kaggle practice.** Top medical-imaging solutions get diversity from:
  - different backbone families;
  - different aggregators (GRU/LSTM vs attention-MIL vs transformer);
  - different inputs (plane/sequence, crop, resolution, slice sampling);
  - auxiliary or segmentation-guided crops.

  They do not get it from seeds. Examples:
  - RSNA 2023 abdominal 1st place: 2.5D triplets, CoaT-Lite and EffNetV2-S backbones, GRU head, auxiliary segmentation, crop-based inputs (survey summary, arXiv 2512.12887; yu4u slides).
  - RSNA 2024 lumbar 1st place ("Avengers"): a multi-stage design with series/level localisation, then crop classifiers on varied backbones (Zenn summary of top solutions).

  I could not fetch the original Kaggle write-ups for detail.

**Decorrelation per GPU-hour, my ranking:**
1. Different input view (plane/sequence subset, ROI crop vs full field of view, different slice sampling or resolution). This costs no extra GPU, because it is the same training run with different inputs. **Highest.**
2. Different backbone family plus a fresh ImageNet init. Avoid ConvNeXt and CoAtNet because both are already in the blend. Use EfficientNetV2-S or a ResNet/ResNeSt/NFNet-style CNN, or MaxViT-T. High.
3. Different aggregator (bi-GRU or a small transformer over slices instead of attention-MIL). This is free and moderate.
4. Different label treatment (hard-thresholded labels, label smoothing, a different soft-label temperature). This is small to moderate.
5. NCL against honest OOF predictions. Moderate, and needs tuning.
6. Seeds, augmentation changes, or self-distillation from the same family. Near zero.

---

## 2. Why our warm start produced a clone, and how to warm-start but still diverge

**Why it cloned.**
- The low learning rate from the parent's weights stays inside the parent's basin (Neyshabur 2020; Wortsman 2022).
- The target of 0.5 × parent prediction is explicit self-distillation, so the loss directly rewards agreement with the parent. That is the opposite of negative correlation learning.
- Same data, same view, same architecture: there was no source of diversity at all.
- Ash & Adams (NeurIPS 2020) also show that warm-started nets keep the old solution's biases.

**Ways to diverge, with evidence:**

| Method | Evidence | Expected divergence | Cost |
|---|---|---|---|
| Fresh ImageNet init, different family, labels-only target | Fort 2019; Gontijo-Lopes 2022 | High | Full training |
| Same parent, but **different input view** (e.g. one plane only, or ROI crops at higher resolution) | Input diversity drives Kaggle top blends; the features must re-adapt | Medium–high | Moderate (a few epochs) |
| **Shrink-and-perturb**: θ ← λ·θ + N(0, σ²), with λ ≈ 0.3–0.5 and σ ≈ 0.01, then normal LR, labels only | Ash & Adams 2020 (closes the warm-start gap; reproduction confirms it) | Medium | Cheap |
| **Re-init the head plus the last stage**, higher LR (5–10× the parent's fine-tune LR), labels only | RIFLE (Li et al., ICML 2020): repeated re-init of the FC layer escapes the pretrained solution; Zhou et al. 2022 "Fortuitous forgetting" (ICLR) | Medium on the head, low on the backbone | Cheap |
| Distill from the **other** family (ConvNeXt → CoAtNet student) | No direct evidence of a gain; it moves the student *toward* the other member | Raises ρ with the blend. **Avoid.** | — |
| Labels only, no parent term | This alone is necessary but not sufficient at low LR | Low | — |
| Stochastic weight perturbation / SWA / dropout | Fort 2019: stays in the same mode | Low | — |

**Rule:** if we warm-start at all, combine three things: shrink-and-perturb or partial re-init, a different input view, and labels-only targets. Before any LB submission, check that rank correlation with the current blend is ≤ 0.95 on held-out data.

---

## 3. Fast training: a competitive 2.5D model from ImageNet in ≤ 10 T4-hours

**FLOPs per image at about 224 px** (to compare with CoAtNet-2, which is about 15 GFLOPs at 224 and about 30 at 320):

| Backbone | GFLOPs (approx.) | Relative to CoAtNet-2 |
|---|---|---|
| ConvNeXt-T | ~4.5 | 3–4× cheaper |
| EfficientNetV2-S | ~3 (8.8 at 384) | — |
| MaxViT-T | ~5.6 | — |
| ResNet-50/ResNeSt-50 | ~4–5.4 | — |

Tricks, roughly in order of payoff:
- **Use both T4s with DDP rather than one GPU.** About 1.8× throughput, the largest single win. Our measured 56 min/epoch was on one T4.
- **AMP fp16 (T4 tensor cores; T4 has no bf16) plus channels_last.** About 1.5–2× speedup for CNNs. Little gain for attention-heavy models.
- **Use fewer slices per study during training.** Randomly sample 16–32 slices or triplets out of 96 each epoch, then use all slices at test time. This is roughly linear speed-up, and the random sampling also regularises the MIL head. It is common in RSNA 2.5D solutions.
- **Train at lower resolution, then fine-tune at higher resolution:**
  - Progressive resizing: fast.ai DAWNBench; EfficientNetV2 progressive learning (Tan & Le 2021).
  - FixRes (Touvron 2019): train at low resolution, then fine-tune for 1–2 epochs at the test resolution.
  - For example, 224/256 for most epochs and 320 for the last 1–2.
- **One-cycle LR schedule** (Smith & Topin, "super-convergence", 2018). Good results in 8–15 epochs instead of 20–30.
- **Freeze the stem and stage 1 for the first epoch or for the whole run.** About 10–20% faster backward pass. Minor accuracy loss for in-domain grayscale.
- **Store data as uint8 memmaps and do augmentation on the GPU.** Otherwise the CPU data loader is the bottleneck on Kaggle's 4 vCPUs.

**Expected outcome (my estimate).**
- About 10–15 min per epoch on T4x2 at 224–256 with 24 sampled slices, so about 10–12 epochs in 3–4 hours. Add 1–2 epochs at 320.
- Expected single-model AUC is about 0.5–1.5 points below full training (≈ 0.935–0.945 vs the 0.949 CoAtNet).
- That is enough: per §1, a 0.94 model with ρ ≈ 0.75 adds about +0.003–0.004.
- The new member must not come from a family already in the blend. Prefer EfficientNetV2-S or a ResNeSt/NFNet-style CNN.

---

## 4. The 58 expert studies and the public LB as decision tools

**Standard error on 58 studies (Hanley–McNeil, derived).**
- For one label at AUC ≈ 0.95 with 15 positives and 43 negatives, SE ≈ 0.04.
- With fewer than 5 positives, SE is 0.08–0.15.
- The macro mean over 12 labels has SE ≈ 0.015–0.025, because label errors are correlated within a study. Treat the 12 labels as about 4–6 independent ones.

**Paired difference between two highly correlated blends.**
- SE_diff ≈ SE·√(2(1 − r)). With an estimate correlation r ≈ 0.9, that is ≈ 0.007–0.01.
- So the 58 studies can only detect differences of **≥ ~0.015–0.02**, and only with a paired bootstrap over studies (resample studies, compute the macro-AUC difference, read off the CI) or DeLong per label.
- Their real uses are:
  1. **Veto** obviously broken models (drops of ≥ 0.02).
  2. Measure **ρ of a candidate against the blend** on clean labels. Correlation estimates are far more stable than AUC differences.
  3. Check the sign and size of a candidate's effect at a few blend weights.
- **Never** tune blend weights, thresholds or per-label choices on them.

**Public LB (~390 studies).**
- Macro-AUC SE is about 0.007 absolute. The paired SE for a small blend change is about 0.002–0.003, which is consistent with the stated ±0.003.
- P(a gain is real, i.e. the sign holds on private), with a flat prior and paired SD ≈ 0.003:
  - +0.001 → Φ(0.33) ≈ **0.63**
  - +0.002 → Φ(0.67) ≈ **0.75**
  - +0.003 → ≈ 0.84
- The winner's curse makes this worse. The best of k submissions is biased upward by about σ·E[max of k normals] (≈ +0.004 for k = 10).
- Public and private are different studies, so a public-LB-tuned weight partly overfits.

**Decision rule.**
- Accept a change only if it has a prior reason (lower ρ, or a new view or family), it is **≥ +0.003 on LB**, and it is **not negative on the 58-study paired bootstrap**.
- For blend weight, use a fixed, theory-guided weight (rank-average with the new member at 0.2–0.33). Do not run an LB sweep.
- For the final 2 picks, take one "safe" blend (the current 0.958) and one "diverse" blend. Do not take two LB-maxed variants.

---

## 5. MONAI / TorchIO extras: an honest assessment

**MRI artefact augmentations.**
- Options: MONAI `RandBiasField`, `RandGibbsNoise`, `RandKSpaceSpikeNoise`, `RandRicianNoise`; TorchIO `RandomMotion`, `RandomGhosting`, `RandomSpike`, `RandomBiasField`.
- They help mainly under **scanner/site domain shift** (TorchIO paper, Pérez-García et al. 2021; Shaw et al. MIDL 2019 on k-space motion augmentation).
- Here train and test come from the same competition source, so the expected direct gain is about **0 to +0.003**, and it could be negative if overused (probability > 0.2–0.3, strength kept low).
- Their real value is **as a cheap diversity lever** for the new member: a different augmentation regime lowers ρ a little, for free.
- Costs:
  - TorchIO `RandomMotion`/`RandomGhosting` are FFT-based and slow on the CPU for 96-slice volumes, so they could make the data loader the bottleneck.
  - Bias field and Rician/Gaussian noise are cheap; do them on the GPU.
- Advice: use bias field, mild noise and Gibbs; skip motion and spikes.

**MONAI ensemble utilities** (`MeanEnsemble`, `VoteEnsemble`, `EnsembleEvaluator`) add nothing beyond our existing rank-average code. Zero value; skip them.

**MONAI 3D networks** are an option only if a genuinely 3D member is wanted for diversity. It would be a different inductive bias, but training from scratch on T4s within the budget is risky. Low EV per GPU-hour.

---

## Sources
- [Gontijo-Lopes, Dauphin, Cubuk — No One Representation to Rule Them All (ICLR 2022)](https://www.arxiv.org/pdf/2110.12899v3) · [ICLR poster](https://www.iclr.cc/virtual/2022/poster/6754)
- Fort, Hu, Lakshminarayanan — Deep Ensembles: A Loss Landscape Perspective (arXiv 1912.02757)
- [Ash & Adams — On Warm-Starting Neural Network Training (NeurIPS 2020)](https://arxiv.org/pdf/1910.08475) · [reproduction](https://chameleoncloud.org/experiment/share/5b5717df-9aa9-470f-b393-c1e189c008a8)
- Neyshabur, Sedghi, Zhang — What is being transferred in transfer learning? (NeurIPS 2020); Wortsman et al. — Model Soups (ICML 2022)
- Li et al. — RIFLE (ICML 2020); Zhou et al. — Fortuitous Forgetting (ICLR 2022)
- Krogh & Vedelsby 1995; Liu & Yao 1999 (NCL); Brown et al. 2005 diversity survey; Buschjäger et al. 2020; [Ross et al. AAAI 2020](https://object.cloud.sdsc.edu/v1/AUTH_da4962d3368042ac8337e2dfdd3e7bf3/ml-papers-txt/AAAI/2020/ensembles_of_locally_independent_prediction_models__c59f7510.txt)
- Long & Servedio 2010 (random noise defeats convex boosters); Dietterich 2000
- Hanley & McNeil 1982; DeLong et al. 1988
- Smith & Topin 2018 (super-convergence); Tan & Le 2021 (EfficientNetV2); Touvron et al. 2019 (FixRes)
- [RSNA 2023 abdominal 1st place summary (arXiv 2512.12887)](https://arxiv.org/pdf/2512.12887) · [yu4u slides index](https://speakerdeck.com/nanachi/bai-bei-jie-fa-korekusiyon-expertdatutaqing-nizu-rinakatutazhi-shi-toji-shu)
- [RSNA 2024 lumbar results](https://www.rsna.org/news/2024/november/2024-ai-challenge-winners) · [Zenn top-solution summary](https://zenn.dev/mkj/articles/ba630a8837ee72)
- Pérez-García et al. 2021 (TorchIO); Shaw et al. MIDL 2019 (k-space motion augmentation)
