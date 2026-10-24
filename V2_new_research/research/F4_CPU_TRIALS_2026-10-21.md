# F4: two local screening experiments after F3

Live Kaggle CLI on October 21 verified submission 57451681 COMPLETE at public 0.957. F0 (57309738) remains 0.958; F1 is 0.954. No private scores are available. GPU quota displayed 26.11 hours used, 3.89 remaining, refreshing October 24. These trials consume zero Kaggle or local GPU time.

F3 only shrank four family weights toward equal weights. The aggregate leaderboard result does not identify which finding regressed or prove why. Do not reverse-engineer per-target weights from this one scalar result.

## Research and chosen hypotheses

- [Large-Scale Robust Deep AUC Maximization, ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/html/Yuan_Large-Scale_Robust_Deep_AUC_Maximization_A_New_Surrogate_Loss_and_ICCV_2021_paper.html) studies AUC-oriented objectives for medical image classification. F4 ranking is an inexpensive **pairwise logistic surrogate plus BCE**, not a reproduction of that paper's loss or optimizer. It weights supported findings equally and ignores unknown labels. Hypothesis: improving positive/negative ordering can help our macro AUC endpoint.
- [ModDrop++ (MICCAI 2022)](https://arxiv.org/abs/2203.04959) studies MRI segmentation with missing modalities. F4 borrows only the whole-series dropout idea: randomly remove each available series with probability 0.2 during training, retaining one if all were removed. It does not implement ModDrop++'s dynamic head or co-training. Hypothesis: reduce reliance on one series when acquisition varies. Applicability to this classification task is unproven.
- [Medical Slice Transformer](https://arxiv.org/abs/2411.15802) explores DINOv2 with slice aggregation, including knee MRI. This supports the existing feature-based screening setup; it does not establish novelty or a competition gain.

## Fixed protocol, recorded before results

Run ranking and series_dropout workers concurrently, each on two CPU threads, with one 900-second supervisor deadline. Reuse immutable completed F2 features and identical 3,427/863 study cohort. Verify source, feature hashes, cohort order, split and proxy-label hashes. Inherit F2's one documented faulty training exclusion and zero validation failures. No reports, expert labels, final test data or competition weights are needed.

Same B architecture, initialization construction, seed, optimizer, batch order, 12 epochs, schedule and highest macro AUC / lower BCE / earlier epoch selector. Ranking coefficient fixed 0.25. No hyperparameter sweep or blending trials. Dropout uses its own seeded generator, so it does not perturb Transformer dropout's random stream. Validation always receives all available original series. Each model checkpoint must reproduce saved validation probabilities exactly after reload.

Compare with saved F2 B (0.7988737822). Use 500 paired group bootstrap draws, with intervals explicitly exploratory after fold reuse and epoch selection. Neither a proxy gain nor decorrelation qualifies a model for addition to F0. The F0 parents lack an honest comparable holdout, and F1 already showed that adding a weaker local member can reduce public performance. Retain F0 as incumbent; no automated submission or final-slot changes.

## New files

- `notebooks/f4_cpu_trials/config.json`: fixed settings.
- `notebooks/f4_cpu_trials/run.py`: CPU worker, two-process supervisor, training, metrics, hashes and checkpoint parity.
- `tests/test_f4_cpu_trials.py`: ranking/masking and whole-series dropout tests.
- `notebooks/f4_cpu_trials/README.md`: execution and interpretation instructions.
- This research record.

Outputs are newly created under `outputs/f4_cpu_trials/20261021T151219Z/`, with separate logs, checkpoints, predictions, histories, provenance and results per arm. Existing F0/F2/F3 source and artifacts are unchanged.

## Completed results

| Arm | Proxy macro AUC | Change versus F2 B | Selected epoch | Worker seconds |
|---|---:|---:|---:|---:|
| Existing F2 B | 0.798874 | reference | 12 | historical |
| F4 ranking | 0.807809 | +0.008935 | 12 | 81.30 |
| F4 series dropout | 0.795851 | -0.003022 | 12 | 74.69 |

Workers ran concurrently. Three focused unit tests passed; both selected checkpoints reproduce all validation predictions exactly. No GPU was used and no Kaggle notebook was pushed or submitted.

Ranking improved 11/12 targets; MCL declined. Its exploratory paired group bootstrap interval for the delta is [0.006851, 0.011496] (500 draws). Dropout's interval is [-0.006301, -0.000093]. These intervals reflect validation sampling, not seed variability, epoch-selection bias, repeated-fold selection, or proxy-to-expert label shift. Both use two CPU threads rather than the historical baseline's four; the training algorithm is otherwise unchanged except for the specified hypothesis. No seed replication has been performed.

Decision: reject dropout for this configuration. Ranking loss is worth a matched multi-seed confirmation before further training expense. It is not an established improvement to F0, and no F0 blend is justified by these results alone. F0 remains the best verified public candidate at 0.958. Preserve the remaining 3.89 Kaggle hours for a candidate with stronger evidence or deployment needs; this run does not establish a new submission candidate.
