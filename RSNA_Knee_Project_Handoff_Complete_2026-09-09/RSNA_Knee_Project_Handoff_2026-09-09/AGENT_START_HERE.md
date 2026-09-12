# AGENT START HERE

You are taking over an RSNA Knee Abnormality Detection Kaggle project.

Read these files in order:

1. `RSNA_PROJECT_MASTER_HANDOFF.md`
2. `PUBLIC_094_NOTEBOOK_AUDIT.md`
3. `EXPERIMENT_LEDGER.md`
4. `NEXT_EXPERIMENTS_AND_INNOVATION_PLAN.md`
5. `PROJECT_STATE.json`

## Do not repeat

Do not rerun:
- rule-v0.2 negation fixes;
- NLI-vs-E5 decision;
- v1.1/v1.2 loss diagnosis;
- six-slot plane×FS feasibility audit;
- ResNet18 six-slot target-attention B2B1.

Those questions are already answered.

## Frozen scientific contracts

- 58 gold studies remain locked for research selection.
- `cv_group_id` must be respected.
- `PSEUDO_LABELS_v1_1` is frozen unless a new supervision version is explicitly created.
- canonical loss = per-target weighted BCE normalized by target weight sum, then equal target average.
- public leaderboard blend constants are not research validation.

## Current bottleneck

The current bottleneck is not DICOM decoding.

It is the combination of:
- representation strength;
- sequence semantics;
- weak-supervision mismatch;
- insufficient model-family diversity.

## First task

Audit a new semantic slot plan:

- SAG_FLUID_FS
- COR_FLUID_FS
- AX_FLUID_FS
- SAG_FLUID_NOFS
- COR_T1
- SAG_T1

Use Phase 4B-1A `Fluid_Sensitive`, `Fat_Suppression`, series descriptions, and plane metadata.

Produce:
- coverage;
- missing-slot patterns;
- readable-slice counts;
- crop feasibility;
- deterministic series selection.

## Second task

Build a conservative strong-backbone baseline:
- DINOv3 ViT-S/16 or DINOv2 small;
- 130-mm crop;
- laterality normalization;
- semantic slots;
- frozen encoder first;
- small target head;
- same pseudo supervision;
- group-safe selection;
- gold only after selection.

## Third task

Create five folds and OOF rank predictions.

Measure:
- per-target AUC on pseudo OOF where meaningful;
- prediction rank correlation across folds/families;
- model diversity.

## Fourth task

Add one diverse family:
- RadImageNet ResNet50 or
- CoAtNet dense slice reader.

Use equal rank averaging first.

## Key hypothesis

B2B1's failure suggests that adding trainable capacity under weak report supervision can improve pseudo loss while degrading official-label ranking.

Therefore:
- prefer pretrained representation;
- freeze most backbone capacity initially;
- use robust / confidence-aware supervision;
- use diversity/ranking rather than one overly flexible student.

## Public 0.94 benchmark

The two supplied 0.94 notebooks are essentially the same source recipe.

Use them as:
- a reproducibility benchmark;
- an architecture inventory;
- an engineering reference.

Do not count them as two independent validation results.
