# Astra scientific review after engineering pilots — 2026-10-09

**Decision: GO for bounded engineering and label-quality work; NO-GO for full M00 or M01 training.** The M00 ResNet18/shared-max implementation, 14 synthetic CPU tests, one synthetic Tesla T4 step probe and 20 real-DICOM preprocessing studies demonstrate executable code and sampled data flow. None measures image-label discrimination or a full-cohort epoch.

The current N01 English-only strict report extractor has candidate validation positive/negative support at or above 20/20 for just three targets: medial meniscus 86/78, lateral meniscus 50/153 and effusion 36/109. MCL, contusion, fracture and all OA targets have no training positives under the strict rules. Lowering support thresholds cannot create training signal or justify a 12-target architecture comparison. Keep ambiguous and silent report cells masked.

Astra required: (1) an explicit all-12 training and independent-validation support gate; (2) integrated 12-window training and 48-window inference runtime measurements; (3) near-report connected-component reconciliation and cross-fold edge count, plus expanded image-linkage screening; (4) slice gap, full orientation and spacing checks; (5) explicit handling of all-unobserved batches; and (6) versioned code/artifact hashes. The plan and M00 geometry code were updated accordingly. Four-worker data loading was measured after the review, but remains a 20-study estimate.

The expert 58 remain untouched for training, tuning and repeated architecture selection. A single final check remains reserved only after a candidate is frozen. No validation AUC, trained checkpoint or competition submission exists for V2.
