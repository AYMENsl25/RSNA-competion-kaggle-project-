# After N3a: the next route toward 0.90

Prepared October 16, 2026, Istanbul time. User requested a small fresh search and a recommendation. Agent Reach's Exa route was used for five targeted searches, followed by primary-source fetches and inspection of the already-downloaded public reader code. No training, checkpoint download or expert-label evaluation was launched in this review.

## First, N3a was useful in the ensemble

N3a completed 12 epochs in **1.6291 hours**. Best public and v3 epoch was 12: **0.8884626 public**, **0.8943030 v3**. At fixed epoch 12, N1 scored **0.8867645** public. Their difference is **+0.0016981**, paired 95% interval **[−0.0028633, +0.0061984]**. This is insufficient evidence of a standalone improvement.

The predeclared addition to the existing ensemble did improve it:

`base = 0.80 Ens5 + 0.20 rank(N1_epoch12)` -> **0.8893508**

`candidate = 0.80 base + 0.20 rank(N3a_epoch12)` -> **0.8918068**

Difference **+0.0024560 [0.0015578, 0.0033240]**. Macro without Synovitis also improves **+0.0026245**. This clears the predeclared ensemble gate, although seed variation and private-test generalization remain unmeasured. Keep N3a as an ensemble candidate rather than discard it.

The principal tradeoff is meniscus performance: N3a improves ACL **+0.00887**, MCL **+0.01504**, and Fracture **+0.01187** relative to N1, but loses **0.02360 Medial Meniscus** and **0.00705 Lateral Meniscus**. More views alone did not resolve these errors. Those point changes do not identify their anatomical cause.

All scores here use the fixed weak-label development set of 885 studies / 843 known groups. They are not leaderboard scores. The remaining gap from the current ensemble to 0.90 on that same public-label metric is **0.0081932**. Changing validation labels or comparing another project's gold/LB score would not establish that improvement.

## Decision: try a sequence-aware contextual MIL head

The best next architectural hypothesis is **N3b contextual MIL**: keep ConvNeXt-Tiny as the image encoder, but explicitly identify each MRI contrast and slice position, then allow window features to exchange information before finding-specific pooling.

Our current head has only three learned plane biases, repeated across the six contrast slots. It has no explicit fat-sat/non-fat-sat identity, no slice-position encoding, and no interaction between windows before pooling. This is a concrete gap relative to the public ConvNeXt reader, rather than a reason to increase the encoder size blindly.

Proposed recipe:

1. Encode exactly the same three-slice windows as N3a with the same ConvNeXt-Tiny ImageNet initialization.
2. Project the 768-dimensional window features to **384 dimensions** with LayerNorm.
3. Add a **six-slot learned sequence embedding** and an MLP of normalized slice-center position, expressed after the existing laterality transform.
4. Apply **two pre-norm transformer encoder layers**, eight heads, feed-forward width 768, dropout 0.1, with padded windows masked.
5. Retain finding-specific attention pooling and independent classification weights for each finding. The transformer operates on image-window tokens, not label tokens; do not add cross-finding post-processing or fixed clinical priors.

Keep 12 epochs, seed 0, 224 input, the existing cache, 4 random windows/slot for training, 16 deterministic windows/slot for validation, v3 soft training labels, unknown masks and the same optimizer schedule/LR. Control is **N3a epoch 12**. Save selected-best checkpoints as a secondary result. This tests the **contextual-head package**; it cannot attribute an effect separately to position embeddings, LayerNorm or the transformer.

Allow **2–3 laptop GPU-hours provisionally**, after a real-data peak-memory and timing check. The measured N3a baseline was 1.63 hours; the additional head should be modest relative to the encoder, but its cost and benefit have not been measured here. No new DICOM download/cache or knee-trained external checkpoint is required.

A judgmental planning range for single-model gain is **+0.003 to +0.010**, with negative outcomes entirely plausible. This is not a confidence interval and does not promise 0.90. Useful tests are:

- Standalone gain at least **+0.003** over N3a, paired lower confidence bound above zero.
- Or a fixed 20% addition to the **0.8918068** ensemble improves it by at least **+0.002**, with positive paired lower bound.
- Examine Medial Meniscus and Lateral Meniscus separately. An overall win should not obscure another substantial meniscus regression.
- Do not stop or promote from a three-epoch screen. Smoke checks establish correctness; all 12 matched epochs establish the initial comparison.
- Confirm an adopted recipe with a second seed before calling it robust. Keep the last gold-58 look reserved.

If we want cleaner component attribution instead, run depth=0 with the new embeddings first and depth=2 afterward. That costs a second controlled run; it is not required to test the practical head package, and it is not a fifth-layer backbone sweep.

## Ranked alternatives

| Priority | Candidate | Why / next action |
|---|---|---|
| 1 | **Contextual MIL head above** | Directly tests missing sequence identity, position and context using existing inputs. Strongest next hypothesis from our code and reference implementation. |
| 2 | **Differential encoder/head LR** | Matched uniform versus differential-rate arms on the winning head, same total schedule. Public reader uses 1e−4 encoder / 5e−4 head; this motivates a test, not copying several training changes at once. |
| 3 | **Real spatial detail / crop retention** | Rebuild from raw DICOM if the remaining small-structure errors justify it. Old-cache upsampling is not this test. Preserve anterior/posterior anatomy and budget the approximately 112.5-GiB full 384 cache. |
| 4 | **One alternative backbone for diversity** | CoAtNet-0 ImageNet weights already exist locally and are a concrete candidate with the same window/head recipe. Require provenance, strict loading, a 6-GB memory probe and one matched run. Performance and runtime unknown. CoAtNet-2/public 0.93 scores are not evidence that this smaller backbone will improve our model. |

ConvNeXt-V2-Tiny is another legitimate architecture, using masked-autoencoder pretraining and global response normalization. The official release establishes the mechanism, not an RSNA gain. I found no reproducible matched evidence that a swap to V2 beats our current V1 recipe. Do not start both CoAtNet and V2 as a sweep.

To cross 0.90, prioritize a genuinely complementary model plus a locked ensemble over claiming a new single model must jump there. A head improvement, an optimization improvement and independent-seed confirmation may cumulatively help, but their gains are not additive. None guarantees a private-LB improvement.

## What the search actually supports

- **Released public ConvNeXt reader:** source contains a 384-dimensional projection, slot and position embeddings, a two-layer window transformer and per-finding attention. Its dataset reports **0.929 public LB for three folds**; that number bundles preprocessing, labels, training and ensembling. It does not prove that the transformer alone adds a particular amount. [Dataset and code](https://www.kaggle.com/datasets/goodpjw2008/rsna-knee-2-5d-convnext-reader). Local source: `research/external_evidence/goodpjw_reader_20261015/knee.py` and `train.py`.
- **Controlled negative architecture evidence:** an independent author's 872-study comparison reports CoAtNet-RMLP-1 **0.8754** versus ConvNeXt-nano **0.8755**, with a 90% difference interval crossing zero and roughly three times the cost. Slot-awareness alone was neutral; raw 336-pixel detail gave a small gain. These are author-reported results on another split/recipe, not reproductions or a universal verdict. [Original experiment ledger](https://raw.githubusercontent.com/TranBaDat2607/RSNA-Knee-Abnormality-Detection/main/docs/phase2_notes.md).
- **Backbone instability counterexample:** another original experiment journal reports CoAtNet wins followed by a large regression on a new seed, while frozen DINOv2 and OrthoFoundation screens did not qualify. This argues for matched repeats; it does not disprove full fine-tuning of those model families. [Author's October 2 journal](https://www.shashankpadala.com/blogs/learning-model-training-kaggle-rsna-experiments).
- **Peer-reviewed sequence-fusion precedent:** CoPAS uses cross-plane/cross-sequence attention and reports AUC **0.812** on its own multicenter cohort. That supports the plausibility of contextual fusion, not comparison with our 0.888 knee-competition validation. Its larger 3D pipeline is not a drop-in choice for this week's laptop budget. [Nature Communications paper](https://www.nature.com/articles/s41467-024-51888-4), [official code](https://github.com/zqiuak/CoPAS).
- **ConvNeXt-V2:** official architecture/training implementation exists. No task-matched improvement was verified. [Meta's official repository](https://github.com/facebookresearch/ConvNeXt-V2).
- **Speculative candidate rejected as evidence:** a public ConvNeXtV2/label-token-transformer model card describes a plausible network but provides no comparable validated score or clear expert-study exclusion. Its weights are not recommended as a clean local initialization. [Model card](https://huggingface.co/leminhhung0101/knee-model).

High-scoring notebook titles, community blend scores and weakly documented model cards were not treated as architecture ablations. No private leaderboard proxy was found. No gold-label look was spent to evaluate search results.
