# S01: fixed-window multi-series coverage

## Question and matched comparator

Test whether sampling two MRI series per plane helps when total encoder input stays at P00's four training and 16 validation windows per plane. The matched comparator is three-epoch P00, not six-epoch E01. E01 is the current artifact-verified development leader and a separate reference. S01 tests series breadth against within-series sampling density; a negative result does not show that other multi-series designs are useless.

Keep P00's ImageNet-pretrained ResNet18 2.5D, three planes, shared max pooling, 224-pixel input, hard/masked report-proxy targets, frozen grouped split, seed 0, AdamW 3e-4 with default weight decay, cosine schedule, batch sizes 2/1, three epochs, masked BCE, and macro-AUC checkpoint rule. No expert-label use or competition submission.

## Frozen series policy

For each study and plane, deduplicate by SeriesInstanceUID, rank by fluid-sensitive descending, fat-suppression descending, then UID ascending. Select the first two; rank one must equal P00's choice. With one series, allocate all four train and 16 validation slots to it. With two, allocate 2+2 train and 8+8 validation slots. Keep unused slots masked for short series; never redistribute them. Process and scale each series separately with the existing geometry, slice-order and percentile rules, then concatenate windows in rank order. Any failure in a selected series is a failed study and stops full training. Never cross a series boundary in a three-slice window.

## Metadata coverage audit

The local official `train_series.csv` SHA-256 is `573c1d80772bf41211c91b149c95677385a1c22d63f485c347f1b46c0177aef3`; frozen split SHA-256 is `dbe879ab5cb719e09c0f9aec612d5b6b53648d6f32dd8d405a075482339b3819`. The aggregate audit (`s01_series_coverage.json`) counts 10,371 one-series loads versus 17,175 selected loads in the 3,457 candidate-training studies, and 2,655 versus 4,383 in the 885 candidate-validation studies: about 65.6% and 65.1% more selected-series decodes, respectively. Every study has all three planes in this metadata, but second-series availability varies. These are metadata counts, not verified DICOM successes.

## Gates and reporting

1. Compile and test deterministic selection, duplicate handling, one-series parity with P00, two-series allocation, short-series masks, no cross-series windows, and failure propagation.
2. Run offline Kaggle CPU preflight on real selected series: hashes/exclusions, actual DICOM geometry and decode, tensor shapes, 3,428 eligible train and 863 eligible validation studies. Pilot must exercise series-rich studies and count decoded series and masked windows.
3. Run a private bounded T4 pilot of 80 training and 40 validation studies, at most one GPU-hour. Require zero failures, finite logits/gradients, verified hashes, and a conservative three-epoch runtime projection below the 7.5-hour stop limit. Record peak VRAM and whole-phase time. The fixed encoder window count does not imply fixed CPU decode or memory cost.
4. Only then launch a three-epoch full run under an eight-hour watchdog, subject to available Kaggle GPU budget. Audit all 12 per-target AUCs, macro-AUC, masked BCE, checkpoint hash, GPU time, VRAM, decoded-series and failed-study counts. Compare predictions on identical validation UIDs with a paired 839-group bootstrap against P00. Report separately against E01 without calling it a matched one-change comparison.

The 58 expert-labeled studies remain sealed. No final competition submission is authorized.

## Execution update — October 12, 2026

Claude CLI authored the initial S01 data module, standalone notebook builder and focused tests. Local review tightened the CPU preflight, runtime pilot sample and completion bookkeeping; an independent Astra review found no blocking data/model flaw. Local Python compilation passed. Local PyTorch/pytest was unavailable, so correctness was tested in the private offline Kaggle CPU preflight: 10 train plus 10 validation studies, 111 selected series, 600 valid windows, zero failed studies, finite model gradient, and verified source/weight/split/label hashes.

The [private S01 GPU pilot](https://www.kaggle.com/code/aymenslimani/rsna-knee-v2-s01-gpu-pilot) completed with `status: passed`: 80 train / 40 validation studies, 400 / 202 selected series, zero failures, 40 optimizer calls and 40 successful steps, 134.36 s training, 61.26 s validation, and 758 MiB peak reserved VRAM. The conservative full three-epoch projection including 20% headroom is 25,483.70 s (7.08 h), below the 27,000 s stop limit by 1,516 s. The uploaded pilot version did not include an explicit AMP-overflow predicate; 40 successful steps out of 40 expected calls verify zero skipped steps in this run. That predicate and counts are included in the local generator for future pilot versions.

The [private full S01 Kaggle notebook](https://www.kaggle.com/code/aymenslimani/rsna-knee-v2-s01-full-train) was launched and reported `RUNNING` at the initial status check. Its only change from the piloted training source is an engineering guard that preserves an already saved completed summary on a late SIGTERM, addressing the E01 status artifact. It does not change model outputs or training. Full results are pending; no expert labels or competition submission were used.
