# RSNA Knee: what the top public notebooks are made of (snapshot 2026-10-17)

Sources: 20 notebooks pulled read-only to `scratchpad/public_nb/` (flattened text in `scratchpad/flat/`), dataset manifests/READMEs in `scratchpad/ds_meta/`, dataset metadata in `scratchpad/dsmd/`, kernel output logs of the 0.9586 notebook and the efficiency LB csv in `scratchpad/kout/`. Nothing was run, submitted or uploaded.

Legend: **[code]** = verified in notebook code / manifest / metadata; **[claim]** = author text only; **[est]** = my estimate.

## 1. TL;DR

* Every public notebook at 0.956–0.9586 comes down to **two model families**. Everything else is weighting:
  1. **nartaa "Raptor" CoAtNet** (dreaddevelopment recipe, retrained with OAI): two checkpoints, 384 px and 224 px (crop 0.8), in one dataset.
  2. **heliosli "compact" ConvNeXt-small** (OrthoFoundation code): **R6** (bundle `fe055406`) in the 0.958 stacks; R7 (`8c7b666e`) and S6 (`209a32c8`) are siblings.
  3. *(optional, ≤3.5 % weight)* **goodpjw2008's 2.5D ConvNeXt-Tiny MIL reader** (0.929 alone). It is the only member trained independently.
* Best public: `heliosli/this-is-best-public-dataset-new-version-0-9586` (claims 0.9586) ≈ `goodpjw2008/rsna-knee-coatnet-r6-2-5d-mil-lb-0-958` (0.958). They use the same models. The only differences are the per-finding weights, whether the 224 model gets the mirror view, and which dataset copy they read.
* **Recommended fork: `goodpjw2008/rsna-knee-coatnet-r6-2-5d-mil-lb-0-958`**. The reasons are in section 5.
* Trap: the latest version of `heliosli/rsna-knee-r6-frozen-fe055406` now holds **R7**, not R6. **[code]** The 0.9586 notebook asserts `bundle_id == fe055406`, so it fails if the dataset version is bumped. goodpjw reads `rostigor/rsna-knee-r6-fe055406`, a versioned copy that is byte-identical to R6 v1 (manifest sha `6508ce69…`) **[code]**.

## 2. Component models

| Model (where) | Backbone / head | Input | Folds / TTA | Labels | External data / weights | Alone LB |
|---|---|---|---|---|---|---|
| **Raptor-384** `nartaa/rsna-knee-publication-swa-weights-20261007` → `raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt` | timm `coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k`, 3-slice windows → per-finding attention MIL **[code]** | 2.5D. 96 slices in 5 slots (Sag-fluid 26, Sag-other 22, Cor-fluid 18, Cor-other 12, Ax 18). 140 mm physical crop, 384 px corpus, 320 px centre crop at inference, K=94 windows **[code]** | 1 model, SWA of epochs 14/11/12. Anatomical-mirror TTA, mean of probabilities **[code]** | `labels_v1_blendjev.parquet`: soft LLM report labels (stevenleehans + pilkwang + riadmohamed42 JEV, CC0) **[claim, RUNS.md]** | **OAI**: 2,399 knees, MOAKS-derived, supervising only PF OA / Lat OA / Synovitis. SWA epochs **ranked on gold-58** (gold not trained) **[claim]** | 0.949 |
| **Raptor-224** (same dataset) `…_c08_r224_swa.pt` | `coatnet_rmlp_2_rw_224` | Same 96 slots, 0.8 centre crop → 224 **[code]** | SWA 15/13/12. Mirror TTA in goodpjw **[code]**; single view in heliosli 0.9586 **[code]** | same | OAI, same run recipe and seed | 0.945 |
| **R6** `rostigor/rsna-knee-r6-fe055406` = `heliosli/rsna-knee-r6-frozen-fe055406-w1` = v1 of `heliosli/rsna-knee-r6-frozen-fe055406` | convnext_small (in22k→1k), first 16 blocks frozen. Per-finding spatial attention + window-attention MIL, plus categorical "state" heads. Post-hoc **fluid stacker** (Effusion) and **state stacker** (Baker's) **[code: manifest]** | 2.5D 3-slice windows, size 384, crop 352, 3.5 mm slab. Views "v4" = full FOV + knee-centred locals placed by an **SKM-TEA anatomy V-Net** (`anatomy.pt`). Side (L/R) classifier `side_cls.pt`. About 530 windows per study **[code]** | 4 folds. Flip TTA (medial/lateral outputs swapped back, logit mean, MCL plain pass) **[code]** | `probe_targets_opusv11b_rr6_plr6_cdr.csv`: report-derived soft and state targets, "opus" = likely Claude-Opus report reading **[code name; meaning est]**. `gold_in_training: 0` **[code]** | Trainer v9.8 (same md5 as R7) has a `--fm-aux` **fastMRI+ lesion-box auxiliary** task. Tag `cnx_r6fm` suggests R6 used it **[est]**. Uses SKM-TEA and KneeXNet pretrained weights (in the OrthoFoundation dataset) **[code]** | ≈0.957 with Raptor (50/50) **[claim]** |
| R7 `heliosli/rsna-knee-compact-r7cnxfsss` | as R6, plus extent / pair-rank losses | as R6 | 4 folds | `…rr7_plr7_cdr.csv` | fastMRI aux explicit (`fm_aux_weight 0.3`) **[code]** | 0.956 with Raptor |
| S6 `heliosli/rsna-knee-compact-v1` | as R6, no stackers | as R6 | 4 folds | `…rr3_plv21…` | — | 0.954 with Raptor |
| S6fall `nartaa/rsna-knee-s6fall-weights` | S6 architecture, retrained by nartaa, size 320 / crop 288 | v4 views | 1 model, all data | blendjev soft labels, gold excluded | no OAI **[claim manifest]** | 0.950. Costs 0.001 when added next to R6 **[claim goodpjw]** |
| **goodpjw 2.5D reader** `goodpjw2008/rsna-knee-2-5d-convnext-reader` | convnext_tiny encoder → 2-layer study transformer → 12 attention pools **[code]** | Canonical volumes at 0.4 mm/px, 384 px (153.6 mm FOV), up to 64 slices. 6 series slots, 16 triplets per series **[code]** | 3 of 5 folds (EMA) | soft labels from 4 public LLM label tables. 5 folds grouped by report text, gold not trained **[claim]** | competition data only **[claim]** | 0.929 |

## 3. Blends

| Notebook | Members | Blend | Claimed LB |
|---|---|---|---|
| heliosli/this-is-best-public-dataset-new-version-0-9586 | R384 + R224 (no mirror) + R6 flip + goodpjw reader | Per-finding percentile rank. Raptor half = 0.65·r384 + 0.35·r224. Final = w·Raptor + (1-w)·R6, with w = 0.5 except Lat OA 0.15, PF OA 0.65, Synovitis 0.85, Baker's 0.30. Then the reader rank-blended at 0.003–0.035 per finding **[code]** | 0.9586 [claim] |
| goodpjw2008/rsna-knee-coatnet-r6-2-5d-mil-lb-0-958 (v27 = v25) | R384 + R224 (with mirror) + R6 flip + reader | Same, w: Lat OA 0.2, PF OA 0.6, Synovitis 0.8, Baker's 0.35 **[code]** | 0.958 [claim] |
| heliosli/blend-mine-in-private-lb-luck-to-all-upvoters | R384 + R224 + R6 flip (from the -w1 copy) | w as goodpjw | 0.958 [claim via goodpjw] |
| goodpjw2008/rsna-knee-does-s6-help-next-to-r6-lb-0-957 (v7) | as goodpjw 0.958 | Per-finding w fitted on gold-58, clipped to 0.1–0.9 ("aggressive", overfit risk) | v4 (with S6) 0.957; v7 pending |
| heliosli/0-957-latest-dataset-score-in-one-click, evgendvorkin/rsna-versia-7 (clone) | R384 + R7 | 0.5/0.5 rank | 0.956 |
| picond/rsna-knee-9576-r6-v1-fixed-readable | R384 + R224 + R6 flip | 0.5/0.5 (no per-finding w) | 0.9576 [claim] |
| bliverrigpdoge/r7-raptor-knee-centred-convnext | R384 + R7 + **private** knee-centred ConvNeXt-S / EffNetV2-S | 0.8 base / 0.2 own | 0.957. A fork gives 0.956 because the weights are private |
| lavinwins/top-1-gold | R384 + S6 + reader | 0.62/0.26/0.12 | parent 0.954 |
| romantamrazov/rsna-knee-dinosaur-the-last | R384 + S6 + reader + residual tweaks | 0.7/0.3 base | ~0.95x |
| nartaa/rsna-knee-0949-anatomical-mirror | R384 alone | — | 0.949 |
| nartaa/rsna-knee-s6fall-solo | S6fall alone | — | 0.950 |
| prvsiyan/the-bee-s-knees-final-rsna-push | re-implementation of nartaa 0.949. The current version is a CPU process-control demo and does not score | — | 0.949 (older version) |
| mattiaangeli speedy-raptors, lavinwins/rsna-gold-medal, tonylica, dreaddevelopment | Older generation: DINOv2/v3-small, RadImageNet ResNet-50 heads, Raptor v5/v8/v10, CoAtNet D4/Global96/res-gated; DINOv2 via Kaggle Models | rank stacks | 0.941–0.943 (superseded) |
| ryanholbrook efficiency-lb | leaderboard only | — | — |

There is no post-processing beyond rank averaging. The R6 stackers (Effusion, Baker's) are built into the model bundle. Nothing is fitted at test time. Every blend uses a percentile rank per finding, and a model that fails on a study is replaced by the other model's rank for that study.

## 4. Lineage: what is actually independent

```
dreaddevelopment Raptor recipe (CoAtNet + per-finding attention MIL, LLM soft labels)
  └─ nartaa retrain (+OAI, blendjev labels, gold-58 SWA selection) ──► R384, R224   [family A: 2 ckpts, same run recipe]
heliosli OrthoFoundation compact (ConvNeXt-S, SKM-TEA knee anchors, state heads)
  ├─ S6 ─► nartaa S6fall (retrain)                                                  [family B]
  ├─ R6 (+fastMRI aux [est], stackers)  ◄── used by every 0.958
  └─ R7 (+fastMRI aux, extent/pair rank)
goodpjw2008 2.5D ConvNeXt-T + study transformer                                     [family C, independent, weak]
```

* The 0.958 / 0.9586 notebooks are A + B(R6) + a little C. Every other ≥0.956 notebook re-blends the same checkpoints with different weights. None of them is independent of the others.
* Gold-58 AUCs per finding (goodpjw, **[claim]**), Raptor half vs R6 flip: ACL .969/.996, MCL .955/.964, MedMen .978/.971, LatMen .894/.916, MedOA .981/.991, LatOA .776/.865, PFOA .861/.918, Effusion .983/.945, Synovitis .812/.859, Baker's .973/.969, Contusion .941/.960, Fracture .962/.887. These are optimistic for Raptor because its SWA epochs were selected on gold-58.
* leoprovorov estimates the shared-error correlation between the public models at ρ≈0.86–0.89, which puts the ceiling of this family at about 0.960 even with 8 equal models **[claim/est]**.

## 5. Recommended fork point

**`goodpjw2008/rsna-knee-coatnet-r6-2-5d-mil-lb-0-958`**, with these datasets attached:

| Dataset | Size | License | Needed? |
|---|---|---|---|
| `nartaa/rsna-knee-publication-swa-weights-20261007` | 543 MB | other: "research and educational use, including the RSNA Knee competition"; OAI acknowledgement required | yes |
| `rostigor/rsna-knee-r6-fe055406` (or the identical `heliosli/rsna-knee-r6-frozen-fe055406-w1`) | 823 MB | other | yes (R6) |
| `heliosli/rsna-orthofoundation-fold0-epoch7-20261004` | 1.34 GB | other: "no new open-source license", upstream SKM-TEA/KneeXNet terms | yes (R6 code, wheels, `anatomy.pt`) |
| `goodpjw2008/rsna-knee-2-5d-convnext-reader` | 353 MB | **Apache-2.0** | yes (Part B; also ships the offline pylibjpeg wheels) |
| `heliosli/rsna-knee-compact-r7cnxfsss` | 823 MB | other | **attached but unused**, can be detached **[code]** |

Why this one rather than heliosli 0.9586:

* It reads R6 from a dataset whose latest version is R6. heliosli's own slug now serves R7 at its latest version, and the code asserts the bundle id.
* The 224 stage is time-guarded: it is skipped if less than 30 min would remain, and on any failure the notebook falls back instead of raising, as heliosli's does.
* It has a ready-made insertion cell, "Part B": a separate process, a `> 8.2 h` skip guard, a per-finding `OWN_W_PER_TARGET` rank blend, and a fallback that restores `submission.csv`.
* To add our model, copy Part B as a "Part C" with our infer script and weights.
* To match 0.9586 exactly, optionally paste heliosli's `W_RAPTOR_C = {'Medial OA':0.5,'Lateral OA':0.15,'PF OA':0.65,'Synovitis':0.85,"Baker's":0.30}` and turn off the 224 mirror. The difference is below 0.001 [est].

**Runtime headroom.** No notebook states a full hidden-test runtime. Facts:

* Hidden test = 1,284 studies **[code, nartaa]**.
* Raptor-384: about 200 studies per 3 min, so about 20–25 min **[claim, nartaa]**.
* R6 measured in the 0.9586 commit log: about 20 s of GPU per study per T4 (4 folds × flip, ~530 windows), with decode overlapped. That gives about 1,284 / 2 GPUs × 20 s ≈ **3.5–4 h** **[est]**.
* Total ≈ 4.5–5.5 h, so roughly **3–4 h of headroom** on 2×T4 **[est]**. This is unverified; budget our model at ≤2 h with its own deadline and fallback.
* Guards in the code: each Raptor stage times out at 2–2.5 h; R6 deadline is 8.3 h from notebook start; Part B is skipped after 8.2 h.

**Expected value of adding our model (0.857):**

* Public evidence says extra readers on these bases help only at tiny weights. goodpjw's 0.929 reader at full weight took 0.956 → 0.954; at 1/10 weight it was neutral. hoshinoyumemi's additions all went down.
* Expect ~0 gain unless our model is genuinely different (other data or labels). Use per-finding weights ≤0.05 and select them on gold-58, keeping in mind that gold-58 is biased toward Raptor.

## 6. Compliance flags

* The Raptor weights were trained with **OAI**, which is controlled-access under an NDA Data Use Certification. Training code is Apache-2.0 per nartaa. nartaa mentions a "leakage-controlled retraining request (discussion 750180)", and SWA epochs were selected on gold-58.
* All the core weight datasets carry license "other". Only goodpjw's reader is Apache-2.0.
* heliosli once removed `compact-v1` after being told it "wasn't allowed", then restored it.
* Medals are probably fine. Prize eligibility for an OAI/"other"-licensed stack is uncertain; read discussions 750180 and 749359 before relying on it.

## 7. Evidence on the 0.965–0.969 teams

* Efficiency-LB csv (ryanholbrook output, scores as of about 15 Oct): 1× 0.967 (CentralyDS), 2× 0.966, 3× 0.965, 3× 0.964, 8× 0.963. Compare 840 teams at 0.956 and 190 at 0.957.
* Several efficiency-rank leaders score 0.964–0.965: rank 1 Claudio Verdú Ruiz 0.964, rank 3 "RB & Matheus & Tucker & Tony" 0.965. This suggests the top teams reach 0.96x with fast or compact pipelines, not giant public stacks. It is an inference only: the csv's score may be the team's best overall, not the efficiency entry.
* No public notebook discloses their methods. Hints from notebook text:
  * Each visible LB step came from a **newly trained, different model**, not from re-weighting (hoshinoyumemi).
  * The report labels agree with expert labels at only about 0.89 AUC; better report reading is named as the main lever (leoprovorov).
  * 87 % of Synovitis targets are imputed from Effusion, and Synovitis, Lat OA and Contusion are the weakest findings (hoshinoyumemi).
  * The heliosli line gains from knee-anchored local views, fastMRI auxiliary lesion boxes, categorical state targets and stackers. A private team likely pushes these further (more folds/seeds, better labels, more external data).
