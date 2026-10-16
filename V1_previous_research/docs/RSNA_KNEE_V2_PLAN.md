# RSNA Knee v2 plan

## Goal

Improve the clean group-safe gold benchmark while keeping the 58 official gold studies locked for model selection. The current reference points are report teacher AUC `0.76067`, canonical MRI baseline AUC `0.69936`, and six-slot attention AUC `0.64473`.

## Rules for every experiment

1. Use the frozen `cv_group_id` split.
2. Never use locked gold labels for architecture, loss, epoch, or ensemble selection.
3. Version every manifest, pseudo-label table, preprocessing configuration, and checkpoint.
4. Produce OOF predictions, per-target AUC/AP, macro AUC, prediction correlations, and a config/hash record.
5. Change one major factor at a time.

## Ordered strategy

### V2.0 reproducibility foundation

Convert the canonical ResNet18 SliceMIL pipeline from notebook-only execution into scripts. Add a manifest builder, split validator, metric runner, checkpoint metadata, and experiment ledger. This is the go/no-go gate for all later modeling.

### V2.1 preprocessing ablations

Run controlled comparisons against v1.2:

- deterministic laterality normalization;
- physical 130-mm crop versus current crop;
- dense deterministic windows instead of sparse anchors;
- contrast normalization and clipping variants;
- sequence-semantic slots instead of plane/FS-only slots;
- missing-slot masks and explicit quality features.

Expected highest-value change: semantic sequence grouping. `AX_NONFS` is too sparse and heterogeneous to be a reliable universal slot.

### V2.2 strong representation

Use a frozen DINOv2/DINOv3 image encoder with a small study-level head. Start frozen, then unfreeze only the final block if the frozen model improves the baseline. Add patch mean and top-k patch pooling for focal findings such as meniscus tears, fracture, contusion, and OA.

### V2.3 supervision robustness

Keep `PSEUDO_LABELS_v1_1` as the control. Test a separately versioned v2 teacher weighting scheme:

- preserve high-confidence rule positives/negatives;
- reduce weight for E5-abstain cases with low confidence;
- use teacher disagreement as uncertainty, not as a replacement label;
- compare macro BCE with a small auxiliary ranking loss only after the BCE control is stable.

### V2.4 OOF ensemble

Train five group-safe folds for the best preprocessing/model families. Produce per-target percentile ranks, measure family correlation, and select equal or OOF-validated rank blends. Keep B2B1 only as a diversity candidate, not as the primary model.

### V2.5 benchmark and final refit

Reproduce the audited public approximately `0.94` pipeline as a separate benchmark track. Only after all research choices are frozen may a submission-only refit include the 58 official gold labels. Its score must not be reported as clean validation.

## Proposed repository layout

```text
src/                 reusable data, supervision, models, training, ensemble code
configs/             versioned YAML/JSON experiment configurations
scripts/             reproducible entry points
notebooks/           exploration and visualization only
docs/                handoff, decisions, protocols, and plans
artifacts/           results, checkpoints, embeddings, and model files
```

## First implementation milestone

Create `src/`, `configs/`, and `scripts/` around the canonical v1.2 baseline. The first successful run must reproduce its stored gold macro AUC within a small numerical tolerance before any v2 model is trusted.
