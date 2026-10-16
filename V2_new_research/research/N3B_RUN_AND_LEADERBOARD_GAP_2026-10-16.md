# N3b contextual MIL: launched 2026-10-16

N3a epoch 12 public-label AUC: 0.8884625760. Existing fixed blend: 0.8918068136. These are weak-label validation scores, not Kaggle leaderboard scores.

Architecture package: ConvNeXt Tiny ImageNet encoder, 384-dimensional projection and LayerNorm, six learned sequence-slot embeddings, normalized slice-center MLP, two pre-norm transformer layers (8 heads, FFN 768, dropout 0.1), finding-specific attention and independent finding classifiers. Positions refer to the already canonicalized/mirrored volume. Only valid windows enter the image encoder; padding is masked in transformer attention and pooling.

Matched N3a settings: 3,428 training / 885 validation studies, same frozen groups, seed 0, same six slots, four random adjacent-slice windows per slot in training, 16 deterministic windows per slot at validation, 224 input, v3 soft labels with 0.5 masked, pos-weight cap 10, AdamW 1e-4 / weight decay 0.05, cosine, fp16, train batch 2 / validation batch 1, one loader worker and prefetch 1, 12 epochs. Fresh cached ImageNet initialization. Gold58 remains sealed. No public knee weights or OAI data used.

Primary comparison: epoch 12 against N3a epoch 12, paired patient-group bootstrap on fixed public and v3 labels. Secondary: fixed 0.80 existing N3a blend + 0.20 rank(N3b epoch 12), without a weight search. Promotion rule: standalone gain >=0.003 with CI lower bound >0, or blend gain >=0.002 with CI lower bound >0. Review medial and lateral meniscus separately. This tests the whole head package, not an isolated transformer effect. Best checkpoint is also retained using the existing v3-label selection rule; epoch 12 is the predeclared comparison.

Verification: src/check_n3b.py passed padding-NaN invariance, position sensitivity, encoder/projection/embedding/transformer gradients, invalid input rejection, frozen N3a random/even sampling parity and exact legacy attention logits. Real-cache CUDA preflight passed three finite optimizer updates with zero AMP skips; worst-six-slot batch peak allocated 3383.31 MiB, validation 1372.76 MiB. Smoke weights discarded. Source and input hashes plus snapshot are under outputs/n3b_context. Automated comparisons run after training. Estimate 2-3 hours is provisional, not a measured completion time.

## Why leaderboard models can be stronger

The reported 0.967 is not verified here as a specific team's result or recipe. Public primary-author evidence documents sequence-aware readers, higher native spatial detail and scan coverage, multiple training folds/full-data models, and complementary ensembles. Some recent stacks use external OAI knee data; our no-OAI constraint stays in force. Private leading teams' exact methods are unknown.

- https://www.kaggle.com/code/goodpjw2008/rsna-knee-stack-2-5d-convnext-mil-lb-0-944 : primary notebook distinguishes community stack and its reader.
- https://www.kaggle.com/datasets/dreaddevelopment/raptor-knee-arms : three diverse backbones scored 0.915 versus 0.914 for CoAtNet alone; author reports slice-corpus changes yielded 0.006-0.011. Evidence against blindly swapping backbones.
- https://github.com/TranBaDat2607/RSNA-Knee-Abnormality-Detection : author's submission ledger distinguishes independently trained models (~0.92), public-stack blends (~0.94), and external-data stacks (~0.95). Their evaluations on Gold58 are external evidence only; we did not run ours on Gold58. Their assertion about all high-scoring notebooks is not independently verified and must not be generalized to private teams.
- https://github.com/willmurray9/kaggle-rsna-knee-abnormality-detection : author clearly distinguishes own model (0.819) from unchanged public-stack reproduction (0.944), and reports one-tick public score variation.

Next decision: finish N3b before starting another GPU job. If it does not give a material gain, audit per-finding errors and prioritize a controlled native-resolution/physical-crop and slice-coverage experiment. Do not simply increase epochs or change backbone. Longer training requires a separate schedule experiment, because changing the cosine horizon alters all 12 epochs. Final full-nonexpert-data training follows recipe selection; use validation models for honest development comparisons.

Launch verification (14:32 Istanbul): epoch 1 reached step 100/1714, running mean loss 1.0209; stderr empty. GPU 55-56 C, about 4982 MiB total device usage (includes reservations/runtime, unlike preflight allocated memory), 94% utilization in sampled check. Dashboard verified HTTP 200 and N3b visible. Background launchers: training 20584, dashboard 15664; real training Python 20016 at verification. These PIDs are historical evidence, never safe targets without rechecking their command lines.
