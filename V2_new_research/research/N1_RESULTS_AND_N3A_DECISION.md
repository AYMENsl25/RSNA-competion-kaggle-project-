# N1 result and next decision — October 15, 2026

The user requested verification of completed N1 and execution of the best next experiment.

## Verified N1 outcome

N1 completed all 12 epochs and its automatic paired comparisons in **1.4385 hours**. The best epoch on both frozen weak-label tables is **10**: public-label macro AUC **0.8877356**, v3-label AUC **0.8934210**. The existing `best.pt` corresponds to epoch 10. Final epoch 12 scores **0.8867645** public and **0.8924837** v3. These are development-validation scores, not a new leaderboard score.

The preregistered endpoint comparison, epoch 12 against E8 epoch 12, is **+0.0123156**, paired 1,000-draw group-bootstrap 95% interval **[+0.0070500, +0.0182671]**, on 885 studies / 843 known groups. Macro without Synovitis improves **+0.0129014**. Eleven findings improve; Fracture falls **0.01413**. Largest gains: Medial Meniscus **+0.04021**, ACL **+0.03655**, Contusion **+0.02486**.

The predeclared blend `0.80 Ens5 + 0.20 percentile_rank(N1 epoch12)` reaches **0.8893508**, versus Ens5 **0.8845611**: **+0.0047898 [0.0034707, 0.0061253]**. All 12 findings improve in that ensemble, including Fracture. These intervals condition on one split and training seed; private-test generalization and seed repeatability remain unmeasured.

## Decision: N3a before the label experiment

Attention now has a clear matched result. The most informative next use of the GPU is the **six-slot attention bridge**, the first stage of planned N3. The target is whether this head can exploit extra contrasts that the earlier six-slot max model used inconsistently. N2 consensus-label development remains deferred: it requires additional label tables and a train-only report audit, while this image-side test uses existing assets and fixed labels.

N3a changes the three-plane inputs into six plane × fat-suppression slots. It keeps the N1 architecture, ImageNet initialization source, v3 soft labels, clipped positive weights, random sampling rule, 224 resolution, mirroring, AdamW 1e-4 / weight decay 0.05, cosine schedule, seed 0, and 12 epochs. Train batch stays 2; validation batch is 1 to retain headroom for six-slot inputs. One worker per loader and prefetch 1 follow the successful N1 setup.

The old three learned plane biases are shared across each plane's contrast pair. **There are no new learned token embeddings, transformer layers or auxiliary losses.** Existing checkpoint shapes stay compatible. Five tests check exact three-slot parity against N1's frozen implementation, tied-plane behavior, missing-slot/NaN masking, gradients into extra slots and empty-study rejection.

This tests an **extra-series/extra-window package**: up to 24 training windows and 96 evaluation windows per study, versus N1's 12/48. It does not isolate sequence diversity from window count or equalize FLOPs. The same seeded sampling rule applies, but additional slots consume random draws and can change the sampled centers in existing slots. Report actual slot occupancy and costs; seed confirmation is still needed.

## Run and acceptance

- Entry point: `src/run_n3a.py`; output: `outputs/n3a_attn6/`.
- Real-cache preflight stresses two fullest training studies and the fullest validation study, checks cohorts/exclusions and cache indexes, and requires three finite fp16 updates. Smoke weights are discarded.
- Primary comparison: final epoch 12 versus N1 final epoch 12, using both fixed label tables. Secondary reference: E3 six-slot max at epoch 12.
- Predeclared ensemble diagnostic: `0.80 * (0.80 Ens5 + 0.20 N1_epoch12) + 0.20 rank(N3a_epoch12)`; compare against the existing 0.8893508 blend, not just Ens5.
- Promotion target: standalone gain ≥0.003 with positive paired lower bound, or ensemble gain ≥0.002 with positive lower bound. Examine both menisci and MCL explicitly. A losing standalone may remain an ensemble candidate only if its fixed-weight contribution is credible.
- Expected runtime **2.5–3.5 hours**, provisional until a full epoch completes. Planning gain versus N1 is uncertain: roughly 0 to +0.008 central expectation, with a meaningful downside risk given E3's lateral-meniscus regression.
- No early stopping based on a few noisy epochs. Twelve remains appropriate for the controlled comparison: N1 peaked at 10 and declined slightly at 11–12. More epochs alone are not the next priority.
- If justified afterward, N3b adds sequence/slice-position embeddings against this exact six-slot attention control. Do not silently merge that change into this run.

No Kaggle training or submission is launched by this decision. No expert-label evaluation or public knee-checkpoint download is performed. Preserve N1's best checkpoint and its source snapshots unchanged.
