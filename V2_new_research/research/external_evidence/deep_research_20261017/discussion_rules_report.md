# RSNA Knee Abnormality Detection: rules, labels, CV/LB, top-team hints, and past RSNA lessons

Research date: 2026-10-17 (5 days before the 2026-10-22 23:59 UTC deadline).

**Method.** The Kaggle discussion HTML does not render through WebFetch. I read the threads verbatim through Kaggle's internal JSON API (`discussions.DiscussionsService/GetForumTopicById`, `GetTopicListByForumId`, `competitions.PageService/ListPages`), using read-only GET/POST calls with no auth. I listed all 198 topics in the forum and read about 90. Raw dumps are in `scratchpad/kg/*.md`: one file per topic id, plus `page_rules.md` and the other `page_*.md` files.

Discussion URLs below take the form `https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/<id>`.

**Legend.** **[VERIFIED]** means the official rules page, the competition metadata, or a post marked HOST in the API. **[CLAIM]** means something a participant posted that I could not verify.

---

## 0. Key competition facts [VERIFIED, competition metadata and pages]

**Format**
- Code competition: notebook submissions only, internet off, a 9 h CPU or 9 h GPU limit, and the output file must be `submission.csv`.
- 5 submissions per day. Maximum team size is 5.
- `leaderboardPercentage = 30`, so the public LB is 30% of the test set. The test set has "about 1300 studies" (data page), which makes the public LB about 390 studies and the private LB about 910.
- The entry and team-merge deadline was 2026-10-15. `kernelsPublishingDisabledDeadline` was also 2026-10-15, so no new public notebooks can appear. Existing public datasets and notebooks stay usable.
- Teams pick 2 final submissions, and both are considered for the main track and the efficiency track ([CLAIM] Chris Deotte, 743356, consistent with the Efficiency page).

**Data**
- Train has 4,407 studies. Only 58 have the 12 labels; the rest have only the report text.
- `test.csv` has no Report field. Reports are a training-time resource only.
- The data page says: "prevalence of abnormalities is not guaranteed to be the same across the training, public leaderboard, and final evaluation datasets."
- DICOMs are stripped to 86 tags and come in mixed transfer syntaxes (JPEG Lossless, JPEG 2000 and others).

**Prizes and metric**
- Main track: $9k / $7k / $6.5k / $6k / $5.5k, then $5k each for 6th–10th.
- Efficiency track: $7k / $6k / $5k. The efficiency metric is AUC / (Benchmark − maxAUC) + RuntimeSeconds / 32400.
- Winners must deliver training code, inference code and weights. Weights go out as a public Kaggle dataset, plus a video and a forum post.
- The winner license field says CC-BY-NC 4.0, but the same paragraph's text describes "an OSI-approved license that in no event limits commercial use". This contradiction is unresolved (739905 has no host answer).
- Scale: 6,127 teams, 7,309 competitors, 134,511 submissions so far.

---

## 1. Rules

### 1a. External data and models: the rule text [VERIFIED, page_rules.md §2.6, Code Requirements page]

- **§2.6.a:** External Data must be "publicly available and equally accessible to use by all Participants … at no cost", or else meet the Reasonableness criteria.
- **§2.6.b:** "The use of external data and models is acceptable unless specifically prohibited by the Host." It must be "reasonably accessible to all" and of "minimal cost", and it explicitly names "geo restrictions" as a concern.
- **Code Requirements page:** "Freely & publicly available external data is allowed, including pre-trained models."
- **§3.x Public Code Sharing:** code shared publicly in the competition forum or notebooks is deemed licensed under an OSI license that does not limit commercial use.
- **§3.x Use of Open Source:** open-source code used must be under OSI licenses that do not limit commercial use.
- **§2.5.a.4:** if input data or pretrained models carry an incompatible license, the winner need not open-license that data or model.

### 1b. Public Kaggle notebooks' checkpoints and datasets in a submission

- **[VERIFIED HOST]** Chris Carr (RSNA Director of Informatics), 2026-10-14, thread **744056**: "1) submissions incorporating publicly shared weights are eligible for awards, and 2) challenge participants should include the shared weights and their source datasets in the list of resources included with their submissions."
  - So using other participants' public checkpoints, including ones trained on competition data, is explicitly allowed and prize-eligible. You must disclose them.
  - The host did not say what satisfies §2.8 "training code" for third-party weights. Practical approach: link the source dataset and notebook. Several weight authors have since published their training code, for example nartaa's `training_code/` (749359).
- **License caveat [VERIFIED rule, CLAIM on specifics].** Public weights carry mixed licenses:
  - nartaa: CC BY-NC 4.0 or Apache-2.0 code
  - mattiaangeli: CC0
  - RadImageNet ResNet-50: CC BY-NC-SA
  - §2.5.a.4 exempts incompatible-license pretrained models from the open-license grant. Questions 735121, 739905 and 743420 got no host answer.
- **What external data is hidden inside popular public blends [CLAIM]** (edteoh, 749858, read from manifests):

| Public resource | External data inside |
|---|---|
| nartaa CoAtNet 384 / 224 (`nartaa/rsna-knee-publication-swa-weights-20261007`) | 2,399 OAI knees, used for PF OA, Lateral OA and Synovitis |
| heliosli R7 ConvNeXt-S | fastMRI + fastMRI+ boxes, as an auxiliary task |
| heliosli `anatomy.pt` crop locator (`heliosli/rsna-orthofoundation-fold0-epoch7-20261004`), used by R6, R7, S6 and nartaa S6fall | SKM-TEA |
| heliosli OrthoFoundation ViT-L | OAI-pretrained |
| goodpjw "read the report then the knee" | fastMRI+ ViTDet heads |
| dreaddevelopment Raptor CoAtNet; heliosli S6 compact-v1 | none found |

  Bottom line: anyone who forked the 0.956/0.957 public blends is almost certainly carrying OAI, fastMRI and SKM-TEA.

### 1c. OAI (Osteoarthritis Initiative)

- **[VERIFIED HOST]** Po-Hao Chen, 2026-09-21, thread **741819**: OAI "does not violate the … rules if it is generally accessible to all participants without an escalated (i.e. institutional) review and sign-off or access to a legal team, and is free of charge. We do not own or control the OAI dataset and cannot comment on whether this challenge complies with OAI data-use terms."
- **[VERIFIED HOST]** Po-Hao Chen, 2026-08-27, thread **733965**:
  - Non-commercial-only licenses do not by themselves exclude a dataset, and prize money is not "commercial".
  - "Datasets that can be accessed through a straightforward registration or click-through data-use agreement would generally satisfy" accessibility.
  - Institution-specific approvals, IRB or negotiated agreements may not.
- **[VERIFIED HOST]** Chris Carr, 2026-10-14, thread **743416**:
  - The host refuses to rule dataset by dataset.
  - "Datasets that require only registration and acknowledgement of licensing terms would clearly meet these criteria. Datasets that require an institutional research agreement or limit access to users with institutional affiliations clearly do not."
  - Asks teams to disclose datasets used, and to rely on "reasonableness".
- **[VERIFIED HOST]** KneeCoT is **banned** (734109, 2026-08-25), because it requires a formal agreement with the hospital.
- **[CLAIM] Facts on OAI access:**
  - NDA access needs only a Login.gov account plus click-through terms. The NDA permissions page shows "Institutional Approval Required: No" (Eesh saxena, 743416).
  - An NDA help-desk reply says it is open to all (Tucker Arrants, 743416). A UMass reply OKs the use (Satwik, 746792).
  - **But** NIH notice NOT-OD-25-083 and NDA's security notice bar "countries of concern" (China, Russia, Iran and others). Several China-based users get Login.gov error LG22 (746792, 748276, 749145).
  - §2.6.b names geo-restrictions. Richard_61 (743416) asked the host to rule on this; there is **no host answer as of 2026-10-16**. hengck23 advised: "assume the host cannot and would not make changes at last week and move on."
- **Risk assessment (my inference).** Under the 10-14 RSNA wording, OAI is most likely allowed: registration plus terms, no institutional sign-off. The open geo-restriction question is a non-zero DQ risk for prize-eligible teams, though not for medals.
  - Public OAI-trained weights are now inside many top public notebooks.
  - Several strong competitors (azamat1ch at 0.959 single architecture, diet1236364, YYama) say they use **no external data**. So external data is not required to reach about 0.96.
- **Licensing of OAI-derived artifacts [CLAIM].** NDA terms forbid redistributing OAI-derived label rows; nartaa did not publish them. 3DReasonKnee (rajpurkarlab, HF, CC BY 4.0) supplies boxes, segmentations and series matching for OAI knees, but the images remain NDA data.
- **Other datasets (no host ruling; my inference from the host criteria):**
  - **fastMRI / fastMRI+:** the application lists an institution and intended use, which is a risk.
  - **MRNet:** the research-use agreement forbids derivative works.
  - **KneeMRI:** CC BY-NC-ND.
  - **SKM-TEA:** Stanford AIMI registration.

### 1d. Hosted LLMs on reports [VERIFIED HOST]

- Thread **733965** (Po-Hao Chen):
  - Commercially hosted LLMs and APIs are permitted for processing reports, for example extracting labels.
  - Sending report text to an external LLM API "will not, by itself, be considered prohibited PRIVATE SHARING."
  - They must be "reasonably accessible … and of minimal cost". The host reserves the right to judge cost and access.
- 2026-08-27: "You can use LLM API, such as those from OpenAI, to read the reports to generate the labels."
- LLM-derived labels or embeddings may be used at training and inference time, but test-time inputs are images only.
- **Open question (748925, no answer):** labels from a *proprietary, non-public* model. Safer to disclose and publish the label file.

### 1e. Using the 58 expert studies, and hand-labeling

- **No rule restricts training on the 58 gold studies [VERIFIED by reading the rules].** They are ordinary Competition Data, and the host intends them to "help participants surface" the report-versus-image gap (733826).
  - Practice: most strong posters exclude them from gradient training and use them for selection or diagnostics (nartaa 746792; colum2131).
  - The public JEV label set was tuned on gold-58 (746792, 747540). Any pipeline using JEV-derived labels has gold-58 contamination, so gold-58 is no longer an independent check for it.
- **§3.4(b) [VERIFIED]:** "submissions may not use or incorporate information from hand labeling or human prediction of the validation dataset or test data records."
  - The test set is hidden, so this applies only to test.
  - Hand-labeling *training* MRIs is considered allowed by participants (740718; PC Jimmmy: "Modification of train data has NEVER been considered as external data"). **There is no host confirmation.**

---

## 2. Label insight

### 2a. How the expert labels differ from report labels [VERIFIED HOST]

- **Process** (733343, 733491, 733826):
  - Labels are assigned **from the images, independently of the report**, using a rubric.
  - Two subspecialty MSK radiologists read each study, and a third adjudicates disagreements. The same process was used for test.
  - "If image interpretation and report text disagree … image-derived label [is] authoritative."
  - Negative means annotated as absent, not "not annotated".
  - "Ambiguous or borderline findings ('on the fence') were graded as negative to favor specificity."
  - Bilateral studies were reviewed individually, and report or DICOM was adjusted to disambiguate. Seven bilateral studies exist in train (735639); whether test contains any is unknown.
- **Official definitions** (733343, pinned overview):

| Finding | Positive | Negative |
|---|---|---|
| ACL | High-grade partial (>50% of fibers) or full-thickness tear | Mild signal, degeneration, thickening |
| MCL | High-grade partial or complete **acute** tear: disrupted fibers plus edema in or around the ligament | Low-grade sprain, chronic or remote changes |
| Medial / Lateral Meniscus | Abnormal signal that **definitely contacts the surface on ≥2 images**, or a morphologic abnormality (truncated, diminutive, displaced fragment) | Intrasubstance degeneration that does not reach the surface |
| Medial / Lateral / PF OA | **Moderate or large area (≈≥1 cm) of high-grade cartilage loss (>50% thickness)** in that compartment, with or without subchondral marrow change | — |
| Effusion | **Moderate or large** fluid distending the joint | — |
| Synovitis | Inflammation and thickening of the synovial lining. No size or contrast criterion given; the 737155 question about MOAKS-style or Hoffa scoring is unanswered | — |
| Baker's cyst | Moderate or large fluid collection in the typical popliteal location | — |
| Contusion | Bone-marrow-edema-like signal from impact **without** a discrete fracture line | — |
| Fracture | **Acute** cortical break or fracture line | — |

  On fracture, an osteochondral fracture in a Bulgarian report was labeled 0 (733826).

- **Measured report-versus-gold gaps [CLAIM]:**
  - Strict human report-only labeling of 20 gold studies agreed with the 240 gold cells 82.5% of the time; positive PPV was 73%, recall 80% (733826).
  - **Effusion:** the expert panel calls fluid positive more often than reports do.
    - Report "absent/normal" → 10/26 (38%) gold-positive.
    - Report "small/equivocal" → 14/18 (78%) gold-positive (colum2131, 745949).
    - Gold has an Effusion=1 where the report says "No knee effusion" (hengck23).
    - Identical English "small effusion" sentences received opposite labels (k256.dev, 735826).
    - So the "moderate/large" threshold is **not** applied the way the report wording suggests. Do not hard-map "small" to 0.
  - **Synovitis:**
    - Only about 16% of reports mention it (737566); 73–84% are "not addressed" (733932, 745949).
    - 27/58 gold studies are positive.
    - When the report is silent, 31–34% are gold-positive.
    - P(syn | eff) = 0.63 versus 0.22 without effusion (733932).
    - Label sets score only about 0.68–0.80 AUC on gold for this finding.
  - **"Silence" means different things per finding** (stevenleehans, 733932). Gold-positive rate when the report is silent versus when it speaks:

| Finding | Report silent | Report speaks |
|---|---|---|
| Baker's | 3% | 44% |
| Medial OA | 0% | 36% |
| PF OA | 21% | 41% |
| Synovitis | 34% | 76% |

    So silence equals negative for Baker's and OA, but carries no information for Synovitis.
  - **Fracture** appears in 7% of reports but 31% of gold labels (741310). Gold is enriched: every gold study has at least one positive, mean 4.14 positives per study (733932).
  - **Per-finding report-label AUC versus gold** (best public sets, 747540): ACL 0.99, MCL 0.98, Medial Meniscus 0.95, Medial OA 0.93, Baker's 0.95, PF OA 0.90, Lateral Meniscus 0.88, Effusion 0.88, Fracture 0.88, Contusion 0.86, Lateral OA 0.83, Synovitis 0.79.
  - **Image models beat the reports** on Effusion, Contusion, Baker's and Synovitis. Reports are stronger on ACL, MCL, menisci and PF OA (NTejas-1 GitHub table; Will 740610). This argues for per-finding blending of the "teacher" (report) and the "student" (image OOF).

### 2b. Label-generation methods used by strong participants [CLAIM]

**Extraction**
- **Single-prompt LLM extraction is the baseline everyone uses.** Gold-58 macro AUC of the label sets:

| Label source | Gold-58 macro AUC |
|---|---|
| Regex | 0.81 |
| Qwen3.6-35B | about 0.83 accuracy |
| Claude Opus 4.5 | 0.892 |
| stevenleehans v4 | 0.893 |
| 6-set consensus | 0.889 |

  - All eight public sets are statistically indistinguishable on 11/12 findings (747540, bootstrap CIs).
  - Cheap models suffice: under $5 of API spend for a 0.949 team (Raymond Yuen); Qwen locally gives "low 0.95 LB" (Tucker Arrants); Gemma 4 locally gives 0.949 single model (Scott Willis).
- **Ask for probabilities plus an explicit "not addressed" or "I don't know" state** (733932, NTejas-1, hengck23 745861). For AUC, ranking matters more than accuracy, so grade the output: explicit-present > equivocal > not-mentioned > explicit-absent (colum2131).
- **Prompt with the host's image-based definitions.** nartaa used Gemini 3 Flash (low thinking) with those definitions. Include few-shot gold examples: Ali Ozan Memetoglu saw local −0.004 but public LB +0.002.

**Target handling**
- **Soft labels beat hard labels.** "Rounding the labels to 0/1 also hurt" (Archit Konde; NTejas-1). FHZ982 measured SOFT +0.014 on gold-58 across 3/3 seeds, but the CI crosses zero (734105).
- **Sharpen** soft targets that cluster near 0.5 (738339). Fill unaddressed cells with **measured priors** rather than 0 (Kirderf; nartaa S6 "prior fill").
- **Targeted cross-finding imputation:** fill only *undecided* Synovitis cells from Effusion. This gave +0.11 on that finding and +0.009 macro. Blanket imputation of all 12 findings was worse (733932).
- **Masking:** nartaa masks only cells with no label. Prateek Grover kept only cells where the LLMs agree and the label is >0.9 or <0.1, dropping about 25% of cells, and gained on CV and LB (745861). Tom Aindow and Archit Konde found masking noisy cells did nothing.
- **Blend multiple label sources.** nartaa's 0.949 used the mean of its own Gemini labels, stevenleehans v4 and JEV HYBRID. That table step was worth +0.002 public (746792).

**The main lever: image-model pseudo-labels or distillation (self-training)**
- **Archit Konde (0.950 single fold):** take *out-of-fold* image-model predictions and blend them into the report labels. Use a heavy model weight: "50/50 already jumped, going heavier on the model added a bit more." In-fold predictions just parrot the labels; replacing labels outright was worse than mixing.
- **colum2131:** ConvNeXt-T 0.953, ViT-S 0.951, ensemble 0.955, with no external data. Image pseudo-labels from the same submission-time input, blended with the LLM labels at a fixed ratio.
- **Raymond Yuen (0.949):** train teacher models on multiple weak-label sources, generate pseudo-labels, train the final model on them. "Different teachers help different targets."
- **azamat1ch:** ConvNeXt-T 5-fold 0.959, EffNetV2-S 0.956, CoAtNet 0.952, no external data. "Short answer is labels"; "AI, not by hand. And don't rely on reports alone."
- **Counter-evidence:**
  - Nicolai Karcher: 50/50 soft bootstrapping lifted gold and CV but not LB.
  - TranBaDat2607: OOF pseudo-labels hurt both gold and LB.
  - Tucker Arrants: extraction improvements and corrections never moved LB, while the image "student" consistently beat the "teacher".
- **VLM image-based relabeling** (report plus MRI to a multimodal LLM, 745949, colum2131 numbers on gold):

| Source | Effusion AUC | Synovitis AUC |
|---|---|---|
| LLM, reports only | 0.835 | — |
| LLM, reports + MRI | about 0.92 | 0.84–0.86 |
| Image model trained on pseudo-labels | 0.986 | 0.859 |

  Kirderf's mosaic-to-VLM benchmark showed high but unreliable runs (747658).
- **Other ideas:**
  - hengck23: active labeling, using the expensive annotator only on the most uncertain cells.
  - hengck23: "random images (definitely negative) help suppress FP".
  - hengck23: a tiny-parameter head trained on the 58 on top of a frozen pretrained prior (claims no shake-up).

### 2c. How much label quality moved the LB [CLAIM]

- Switching to a better public label set, same model:
  - SpeedSci: 0.931 → 0.942.
  - Yann Majewski: combining own labels with a public set gave +0.015.
  - Abdullah Wasee: "a bigger difference than I expected".
- Raymond Yuen: 0.938 → 0.949 from "still label improvements" (pseudo-label blends).
- Label-table blending plus JEV: +0.002 (nartaa).
- Several top posters (Tucker Arrants, tennogh) say refined extraction moved nothing beyond noise. **The consensus is that the gain comes from image-based relabeling (OOF pseudo-labels, multi-teacher distillation), not from better text extraction.**

---

## 3. CV versus LB

- **Public LB size and noise.** About 390 studies, scores truncated to 3 decimals.
  - [CLAIM] Simulated SD of about **0.003** at AUC ≈ 0.955 (varshitha kolupuri, 748011; nartaa 749359).
  - About 135 teams per 0.001 between 0.941 and 0.956 (749570).
  - Inside a tie block, rank still follows the unrounded score, so rank movement is a finer paired signal than the displayed score (748011).
  - Blend-weight sweeps produced 13 identical 0.937 readings (740439).
- **Correlation with LB** (colum2131, 748011, 15 submissions, 5-fold):

| Local score | r with public LB |
|---|---|
| OOF AUC on the ~4,300 LLM-labeled studies | **0.87** |
| Gold-58 AUC | **0.66** |

  - Tucker Arrants: CV on report labels was "strongly correlated" with LB. CV of 0.87–0.90 from different label sets all mapped to about 0.94 LB.
  - Nicolai Karcher: an extremely strong correlation, until label bootstrapping broke it, because CV on modified labels is no longer comparable.
- **Typical mapping: LB > OOF on report labels by about 0.02–0.05, and LB ≈ gold or slightly above.**

| Source | OOF (report labels) | Gold-58 | Public LB |
|---|---|---|---|
| datnt114 | 0.937 | 0.938 | 0.961 |
| Raymond Yuen | — | 0.925 (label 0.89) | 0.947 |
| Scott Willis | — | 0.930 (labels 0.89) | 0.949 |
| Archit Konde | — | 0.930 | 0.950 |
| nartaa S6 | 0.907 | 0.931 | 0.950 |
| nartaa CoAtNet | 0.890 | 0.919 | 0.949 |
| Cody_Null | 0.90 | 0.92–0.93 | 0.931 |
| bishnoiyash (HF blog) | 0.854 | 0.857 | 0.903 |

  - hengck23 rule of thumb: labels at 0.89 on gold, model train AUC 0.95, about 0.92 gold, about **0.930 LB**.
  - Many posters (736635) show OOF systematically *below* LB, nonlinear at low AUC.
- **Diagnostic for our team.** Our weak-label validation is 0.89, which by the community mapping should give about 0.93–0.95 LB, but our LB is 0.857. A gap that large matches **inference-path bugs**, not model quality. Documented causes:
  - **bf16 autocast on T4/P100.** It is emulated, about 50× slower. It hits the time guard and fills 0.5 for most studies: 0.851 → 0.710 for one team (744230). Use fp16; check `torch.cuda.is_bf16_supported(including_emulation=False)`.
  - **Slice order by filename (SOP UID).** This is random order; fixing it is worth +0.028 (745759, 734105). InstanceNumber runs opposite to position in more than a third of series (741310). Sort by IPP projected on the slice normal.
  - **Preprocessing contract mismatch** between training and inference, for example normalization constants (735154).
  - **Decode failures** on JPEG2000 or JPEG-lossless. Need pylibjpeg/gdcm fallbacks and log the count of fallback studies.
  - **Missing-series handling.** 3–14 series per study; the most common combination covers only 40%.
  - **Laterality** missing in about half of studies; derive it from geometry.
  - The fix recipe: run the *exact* submission path on held-out train studies and score them against gold (744230).
- **Shake-up expectations [CLAIM].**
  - Many medal-zone teams run forks of a few public blends, so shake-up risk is high around the medal lines (742327, 748181).
  - Kaggle's Ryan Holbrook: low-effort public blends "tend to drop on the private leaderboard".
  - Historic data (Georgy Mamarin, 742327): teams just inside or just outside the bronze line medal at about 24% either way.
  - Per-label weights tuned on public, such as `Lateral Meniscus = 1.00` in the most-forked notebook, are flagged as the likeliest give-back (749570, 742050).
  - Private split is unknown: site-stratified versus held-out sites was not answered (734681). Prevalence may differ.
  - **Advice from multiple posters:** select one robust, uniformly weighted submission and one best-public submission; do not trust ±0.001.

---

## 4. What the top teams (0.955–0.969) say or hint

Top public LB is about 0.969; strong public notebooks are about 0.954–0.958. Most top teams are silent until write-ups. Hints:

- **Labels first.** "model is not the key. it is your labels" (hengck23). "Short answer is labels … AI, not by hand. And don't rely on reports alone" (azamat1ch, 0.959 single architecture, no external data). A Chinese user posted a single submission at 0.960 and described it on Rednote as "basically a single model" (748111).
- **No external data is needed for about 0.96** (azamat1ch, diet1236364, colum2131 at 0.955, datnt114 at 0.961).
- **OAI as extra supervision** for weak classes:
  - nartaa recipe comparison: 0.940 → 0.945, then 0.948 with 384 full-field. This is not a controlled ablation.
  - OAI rows: MOAKS full-thickness cartilage loss ≥1 in any PF subregion → PF OA; same rule over lateral tibiofemoral subregions → Lateral OA; Hoffa-synovitis (MSYIC) ≥1 → Synovitis. All other findings are masked (746792).
  - hengck23: "there are some classes that have more error than others, and these classes have external data with external labels"; "kaggle annotation > external annotation > negative".
- **Architecture: small 2.5D backbones** with three adjacent slices as RGB, attention MIL over windows, and per-finding attention heads.
  - Reported scores: ConvNeXt-T, EffNetV2-S, CoAtNet-rmlp-2-384, DINOv2 ViT-S, ResNet-50 (0.954, 5-fold at 224).
  - Bigger encoders give a null result (DINOv2-S → B: +0.0011, 735154). Qwen3.5-2B VLM fine-tune: 0.950 single fold.
- **Resolution and field of view:**
  - 224–288 px is a "sweet spot" for ResNet and EffNet (Tucker Arrants, tennogh).
  - CoAtNet benefits from native 384 with a 320 center crop; narrower crops lost 0.001–0.002.
  - Physical FOV around 140–160 mm beats 100 mm (Cody_Null, Parag).
  - Slice geometry and the number of windows matter more than resolution. Raptor: 42 → 62 windows gave +0.003; 94 versus 24 windows gave 0.940 versus 0.934 (737696, 746792).
- **Localization and ROI crops:** heliosli S6 uses full FOV plus 100 mm knee-centered local crops placed by an SKM-TEA anatomy V-Net, which gave 0.950 retrained on nartaa labels (749359). 3DReasonKnee offers OAI boxes and segmentations. Explicit 2×2 region tokens: null (740610).
- **TTA:**
  - Anatomical mirror: sagittal mirrored by reversing slice order, coronal and axial by width flip. +0.001.
  - Naive all-plane width flip hurt. Never flip without swapping medial and lateral labels.
- **Weight averaging:** 3-epoch weight averaging selected on gold (nartaa). Raptor saw SWA ≈ no gain.
- **Ensembling:** decorrelated families help (Dread Development, Chris Deotte).
  - The 224 crop model plus the 384 model moved rank without changing the displayed score.
  - colum2131: ConvNeXt + ViT = +0.002.
  - Rank-average blending is standard.
  - Public 20-member stacks reach 0.941; better stacks reach 0.954–0.958.
- **Pseudo-labeling the test set:** no one reported it. Reports are absent at test, and the test runs inside the hidden kernel, so it would have to be in-kernel self-training. Not observed.

---

## 5. Lessons from past RSNA winners that transfer

### RSNA 2024 Lumbar Spine
Sources: 1st, NANACHI 540091; 2nd, Yuji/Ian Pan/Brendan/Kevin 539452; 3rd, Moyashii 539453; 4th, tattaka/yu4u 539443; 5th, siwooyong 539472; summary, 541279.

- **Two-stage localize-then-classify was used by almost every top team.**
  - Stage 1 predicts slice index (instance number) and in-plane keypoints, using 3D ConvNeXt or 2D+LSTM regression or classification, CenterNet, UNet heatmaps with Gaussian-expanded targets, or YOLOX boxes.
  - Stage 2 classifies **crops around each keypoint**.
  - Crops let you treat levels and sides as extra samples (5–10× data).
  - Knee analog: localize medial and lateral menisci, ACL notch, MCL, PF joint, popliteal fossa, then crop per structure. Medial and lateral can share weights via mirroring.
- **Stage-2 model:** 2D CNN encoder per slice, then **BiLSTM/GRU or transformer**, then **attention-pooling MIL**.
  - ConvNeXt-S and EffNetV2-S were best. Small models beat large ones (1st: convnext-large < base < small). CNNs beat ViTs for the 1st-place author.
  - Short training (7–14 epochs).
  - **Auxiliary attention or slice-level losses** helped.
- **Robustness to stage-1 error:** random shift of keypoints (±10 px) and slice index (±2) during training. The 2nd-place team used 27 shifted crops at train time and as TTA.
- **Multi-view fusion:** concatenate per-view features before the head (1st) or a transformer across all views' slice features (3rd, 4th). Neither was clearly superior.
- **Noise removal:** the 2nd-place team dropped training samples where |OOF − label| ≥ 0.8, worth +1% on public and private. The summary author could not reproduce it, so validate it.
- **Metric alignment:** the 5th-place team pretrained without metric weights to maximize AUC, then **froze the backbone and fine-tuned only the head** with the weighted loss. Cutmix with p=1.0 was their strongest regularizer.
- **Stacking:** 4th-place used LGBM/XGB stacking and a Nelder-Mead-guided MLP. External coordinate data (brendanartley's dataset) helped stage 1.

### RSNA 2023 Abdominal Trauma
Sources: 1st, Team Oxygen 447449; 2nd, Theo Viel 447453; 3rd, Yuji 447464.

- **3D segmentation** produces organ crops; then a 2.5D CNN (3 adjacent slices) + RNN works on crops at 384 px.
- **Slice-level soft labels:** study label × organ-visibility ratio from segmentation (1st). The analog is weighting slices by structure visibility.
- **Auxiliary segmentation loss** on the classification encoder stabilized training (1st).
- **Frame sampling guided by organ presence** reduces label noise (2nd). Heavy augmentation plus cutmix. A separate RNN aggregator was trained on frozen per-frame features to optimize the metric.
- **Masked crops** (box × mask) for irregular organs; multiple crop sizes; per-target weighted averaging. Stacking overfit (3rd).

### RSNA 2022 Cervical Spine
Sources: 1st, Qishen Ha 362607; 2nd, Ryan R 365115; 3rd, Darragh 362643.

- 3D segmentation trained on only **87** masked studies was enough to pseudo-label all 2k studies. Then per-vertebra crops; 15 slices × 5 channels plus the mask as an extra channel; 2.5D CNN + LSTM.
  - Analog: a small hand-annotated or anatomy-model set (SKM-TEA anatomy, 3DReasonKnee) can bootstrap knee ROI detection.
- Darragh: a bounding-box regressor to zoom in on the anatomy, plus slice-level visibility-ratio targets, with 2.5D CNN + 1D RNN + attention.
- The 2nd-place team: 2.5D UNet seg (3-channel), then a CNN + BiGRU + attention classifier.
- Training 3D CNNs directly underperformed 2.5D.

### Transfer checklist for the knee task
1. Fix the inference path first (fp16, IPP sort, decoders).
2. Labels: soft LLM labels plus OOF image pseudo-label blending, per finding.
3. 2.5D adjacent-triplet ConvNeXt/EffNetV2/CoAtNet with attention MIL and per-finding heads, many windows.
4. Anatomy-guided crops as a second view.
5. Anatomical mirror TTA with label swap.
6. Diverse small-model ensemble with rank averaging, uniform weights.
7. Pick one public-LB-best and one robust submission.

---

## Source URLs (primary)

**Competition**
- Rules, data, efficiency and prize pages: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection (via API `competitions.PageService/ListPages`, competitionId 154281)

**Host rulings**
- 733343 overview and definitions; 733491; 733826; 733965 hosted LLMs and non-commercial datasets; 734109 KneeCoT ban; 741819 OAI; 743416 RSNA external-data guidance; 744056 public weights prize-eligible

**OAI and external data**
- 746792 nartaa 0.949 with OAI recipe; 749359 0.950 S6; 749858 external data in public blends; 748276, 749145 OAI geo-restriction; 747455; 743420; 739905; 735121; 748925; 740718

**Labels**
- 733932 "not addressed"; 737566 synovitis; 745949 effusion and synovitis; 747540, 737454 label-set benchmarks; 734105 soft versus hard; 745861 hengck23 tricks; 747658 LLM and VLM benchmark; 745214

**CV/LB and shake-up**
- 735304 best single model; 748011 CV versus LB; 736635; 742327 shake-up; 749570, 742050 public-blend overfit; 740439 3-decimal resolution; 748181

**Pipeline bugs and modeling**
- 744230 bf16 bug; 745759 slice order; 741310 data traps; 735154 encoder scaling; 740610; 738339; 737597; 737696 Raptor; 733517, 734004 metadata

**Past RSNA solutions**
- rsna-2024-lumbar-spine-degenerative-classification/discussion/{540091, 539452, 539453, 539443, 539472, 541279}
- rsna-2023-abdominal-trauma-detection/discussion/{447449, 447453, 447464}
- rsna-2022-cervical-spine-fracture-detection/discussion/{362607, 365115, 362643}

**External pages**
- https://huggingface.co/blog/bishnoiyash/rsna-competetion
- https://github.com/TranBaDat2607/RSNA-Knee-Abnormality-Detection
- https://github.com/willmurray9/kaggle-rsna-knee-abnormality-detection
- https://github.com/Sibusiso-K/RSNA-Knee-Abnormality-Detection (README only; details in docs/)
- https://github.com/NTejas-1/RSNA-Knee-Abnormality-Detection
- https://www.rsna.org/news/2026/august/ai-challenge-knee-mri
