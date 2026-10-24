# Lessons from Top Solutions: Reusable Playbook

Source: what the 2nd, 5th, 7th and 8th place teams did in RSNA Knee Abnormality Detection (Kaggle, Oct 2026).
Our score: 0.857 private. Theirs: ~0.958. Most of that gap came from **labels, not models**.

Use this as a checklist when starting any new supervised-learning project, especially one with noisy or weak labels.

---

## 0. The core idea

> **Fix what the model learns from before you change how it learns.**

All four teams found that label changes moved the score far more than model changes:

| Team | Gain from labels | Gain from model/ensemble changes |
|---|---|---|
| 5th | +0.008 private | anatomy crops +0.003, everything else ≤ 0.001 |
| 7th | most of the total | 4-model ensemble added +0.001 public, 0 private |
| 8th | public 0.937 → 0.954 | ranking loss +0.001, head retrain +0.006 gold |

---

## 1. Ask first: do my training labels match how the test is graded?

**Problem pattern:** training labels come from source A (text reports, weak rules, user clicks), but the test is graded by source B (experts looking at the actual data).

- RSNA: training labels came from radiology reports; the test was graded by experts reading images.
- Reports leave out mild findings. On 58 expert-labelled exams the experts marked **240 positives where the reports mentioned only 154** (PF OA: 21 vs 2).
- Report-only labels capped at **~0.90 AUC** against the expert labels, however good the LLM reading them was.

**Checklist**
- [ ] Write down where the train labels come from and where the test labels come from.
- [ ] If they differ, estimate the ceiling: how well do the train labels score against test-style labels?
- [ ] If the ceiling is low, work on labels first.

---

## 2. Keep a small trusted set and score your labels against it like a model

Every team used the **58 expert-labelled exams ("gold")** as an answer key, both for model predictions and for **each version of the labels**.

- Treat each label version like a model submission: compute its AUC on the gold set.
- A small gold set is noisy (58 exams ≈ ±0.02 by chance). The 2nd place team **tuned on 24 and kept 34 held out** to confirm each frozen version.
- Any rule you derive from the gold set (thresholds, floors) is a clue, not proof. Mark it as such.
- 8th place: gold AUC and public LB correlated well across 22 submissions (Pearson r = 0.92), so gold was a usable fast proxy.

**Checklist**
- [ ] Find or create a small high-quality labelled set (even 50–100 items).
- [ ] Split it: a tuning part and a held-out part.
- [ ] Version your labels (`labels_v1.csv`, `v2`, ...) and score each one on it.

---

## 3. Use big models offline to make labels; ship a small model

LLMs and vision LLMs **never ran at test time**. They only produced training labels; a normal CNN was trained on those labels and submitted.

- Cost: the 2nd place team trained all final models for ~100 GPU hours (~$100), with labelling on top of that.
- The host explicitly allowed commercial LLM APIs for labelling. **Check the rules first.**

**Pattern:** expensive teacher (LLM / VLM / large ensemble) → soft labels → cheap student model.

---

## 4. Let the vision model look at the raw data, not just the text

| Label source | Gold AUC (RSNA) |
|---|---|
| LLM reads report only | 0.89–0.91 |
| VLM reads images only | 0.919 |
| VLM reads report + some key slices | 0.935 |
| VLM reads **every slice** + report | 0.942 |
| VLM **verifies** report against images | **0.947** |

Lessons:
- **Show it everything.** An early pilot with 44 of 134 slices said "images don't help." With all ~277 images, they gave the largest gain.
- **The prompt matters:** "The report is a draft that may contain errors; check each finding against the images" beat every other prompt.
- **Image-first option (8th):** the model judges the images alone first and must give supporting slices and bounding boxes for every positive. Only then does it see the report, and on disagreement the images win.
- **Don't show the model the answer key.** When the 2nd place team showed Astra the expert labels, it copied them, and a rule it wrote from them scored 0.80 when used blind.

---

## 5. Look for patterns in the label noise and correct them directly

Label noise is often **systematic**, not random. Find the pattern and fix it.

- **Definition mismatch (5th):** the host's guide counts an MCL injury only if it is an acute high-grade tear, so low-grade tears in reports were really negatives (0 of 5 were positive for the experts).
- **Missing ≠ negative (5th):** one public label table gave no answer for 27% of pairs (84% for synovitis). Don't map "not mentioned" to 0.
- **Overconfident hedges (5th):** hedged "yes" labels (~0.75) were correct only 38% of the time.
- **Hospital habits (7th):** the LLM marked synovitis almost exactly when the report used the word, and word use depended on the hospital. Fix: use the scanner model as a stand-in for the hospital, then shift each site's labels so the site's mean matches what its effusion level predicts. The order within a site stays the same.
- **Language = hospital (2nd):** they read each report in its original language, because a translation slip would teach the model the hospital instead of the anatomy.
- **"Unclear" is informative (2nd):** all 12 exams the reader called "unclear" on PF OA were positive for the experts, so they set a floor of 0.8.
- **Mixed subjects (8th):** 8 studies contained both knees. They split them into per-knee examples.

**Checklist**
- [ ] Group label errors (vs gold) by category, site/source, language and wording.
- [ ] Read the official label definitions and compare them with how the source text uses the words.
- [ ] Handle "not mentioned" separately from "negative".

---

## 6. Use soft labels, combine sources per class

- Use values in [0, 1], not hard 0/1. Map the LLM's answer categories (yes / probably / unclear / probably no / no / not mentioned) to numbers.
- **Weight the sources differently for each class.** 8th place blended `S = w·Image + (1−w)/2·(TextA + TextB)`:
  - Fracture / Synovitis / Baker's cyst: w = 5/6 (images are better)
  - PF OA / Lateral OA / MCL: w = 1/6 (text is better)
  - Pick w per class by sweeping it and keeping the best gold score. Don't over-tune: a later ±1/6 search changed nothing.
- **Rank-blend sources while keeping each class's prevalence (7th):** convert scores to ranks, blend, then map back onto the original label distribution.

---

## 7. Self-distillation: feed out-of-fold predictions back into the labels

1. Train K-fold models on the current labels.
2. Collect **out-of-fold** predictions, so each sample is scored by a model that never saw it.
3. **Calibrate** them to the label scale (8th: per-class isotonic regression fitted on the other folds).
4. New label = `0.6 × original + 0.4 × calibrated OOF`. Use 0.5/0.5 or 0.7/0.3 depending on how much you trust the source.
5. Optionally repeat once. **Always blend with the original labels**, not the previous round, or the original signal fades.

Results: 7th place 0.947 → 0.952 gold; 8th place 0.931 → 0.935 gold, then 0.941 → 0.943 in round 2.

**Warning:** this leaks. A teacher may have trained on your validation fold, so local CV goes up artificially. Judge on gold or the LB, not on CV.

---

## 8. Validation that doesn't lie

- **Group folds by source** (hospital, scanner, user, device). 5th place made 20 site clusters. A model scored 0.019 higher locally when the same hospital was also in training, but not on the expert labels.
- **CV against noisy labels rewards copying the noise.** Two label sets scored the same in CV (0.873) but differed by +0.008 on the LB.
- **Check one class at a time on the LB:** submit real predictions for one class and 0.5 for the rest. With N classes and mean AUC: `AUC_class = N × LB − (N−1)/2` (12 classes: `12 × LB − 5.5`).
- **Test changes in pairs:** two submissions that differ in only one thing (7th).

---

## 9. Input pipeline: check this before touching the architecture

The cheapest big wins were input fixes:
- Reading **4 → 15 slices per series: +0.033** (2nd place, bigger than most model changes).
- **64 → 168** cached slices: public 0.930 → 0.937 (8th).
- At test time, spread samples across the whole volume and average a few offsets (e.g. 0, 1, 2).
- Normalize orientation: **mirror right knees into left** (or the equivalent for your domain). Careful with flips when left/right matters.
- **Regions of interest:** crop where the target is, at higher resolution, and **keep the full view too**. A close-up alone lost 0.010 on Lateral Meniscus until the whole knee was added back (2nd).
  - Fixed boxes (7th: 13 fixed regions) work fine.
  - Learned anatomy (5th: bone segmentation → 4 crops) gave +0.003.

---

## 10. Architecture patterns that worked (for any "bag of images → labels" task)

Works for 3D medical scans, video, multi-page documents, product photo sets, etc.

- **2.5D input:** stack 3 neighbouring slices as RGB channels so a normal 2D ImageNet backbone can read them.
- **Attention MIL pooling** (Ilse et al. 2018): one attention per class over all windows. Use two levels: windows within a series, then series within a study.
- **Tag each input** with its type (plane, contrast, crop region) as a learned embedding.
- **Label queries** (Query2Label / ML-Decoder): K learned queries per class that cross-attend over all tokens (7th).
- **3D position embeddings in millimetres**, so different views line up in one shared space (7th).
- **Sequence model across slices:** a bidirectional GRU in depth order (8th).
- **Global + local features:** average pooling plus per-class spatial attention (8th).
- **Ranking loss on top of BCE** when the metric is AUC: penalize pairs where a higher-label sample gets a lower score (8th: gold 0.926 → 0.930).
- **Retrain the head on a bigger bag:** freeze the backbone, **reset the head**, retrain on many more windows (8th: gold 0.945 → 0.951; continuing with the old head overfitted).
- **Domain pretraining helps a little:** fastMRI, or the OAI grades (5th: +0.001).
- **Stabilizers:** EMA or SWA of the weights, AdamW with a cosine schedule.
- Things 7th place tried that **did nothing**: multi-scale tokens, a slice mixer, a bigger decoder, loss balancing, heavier augmentation, larger crops.

---

## 11. Ensembling

- **Rank average per class**, not probability average. AUC only cares about order, and ranks put differently calibrated models on the same scale.
- **Fixed, simple weights.** Weights tuned on the gold set lost 0.002–0.007 on held-out exams (2nd).
- **Diversity is what makes an ensemble help.** Use other backbones, inputs, crops or **label sources**. Models that look alike add nothing: a 0.959 model added 0 to the blend (2nd), and 4 models with the same pipeline added 0 on private (7th).
- **Specialists:** a model that votes only on the weakest classes, with a small weight (~0.2).
- **Fail-safe inference:** write the core submission first, give each extra member a time budget, and drop it if it fails. Share preprocessing across models and use both GPUs.

---

## 12. Start-of-project checklist

1. [ ] Read the rules: external data, external APIs (LLMs), pretrained weights.
2. [ ] Train labels vs test labels: same source? Estimate the ceiling.
3. [ ] Find or build a small gold set, split into tune and held-out parts.
4. [ ] Fix the input pipeline: enough samples per item, orientation, crops plus the full view.
5. [ ] Baseline: strong public notebook + 2.5D backbone + attention MIL. **Starting from a good public notebook is fine:** 2nd place built its best model on one.
6. [ ] Label iteration loop: new label version → score on gold → train → LB pair test.
7. [ ] Look for systematic noise (definitions, sites, wording, "not mentioned").
8. [ ] One or two rounds of self-distillation.
9. [ ] Only then: architecture tweaks, ranking loss, specialists.
10. [ ] Ensemble diverse members with rank averaging and fixed weights.

---

## References

- 2nd place: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/752370
- 5th place: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/752190
- 7th place: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/752217 (all label versions public: https://www.kaggle.com/datasets/songqizhou/rsna-knee-llm-labels)
- 8th place: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/752219
- Ilse et al., Attention-based Deep MIL, ICML 2018 (https://arxiv.org/abs/1802.04712)
- Query2Label (https://arxiv.org/abs/2107.10834), ML-Decoder (https://arxiv.org/abs/2111.12933)
- Born Again Neural Networks (self-distillation), ICML 2018
