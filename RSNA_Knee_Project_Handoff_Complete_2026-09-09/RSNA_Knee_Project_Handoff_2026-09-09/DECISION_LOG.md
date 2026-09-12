# Decision Log and Rationale

This file records high-level project reasoning and decisions. It is intentionally a reproducible decision summary, not private chain-of-thought.

## Decision: preserve gold lock
Reason:
58 gold studies are too small to support repeated architecture tuning without overfitting.

## Decision: freeze rule teacher
Reason:
rule v0.2 achieved high covered balanced accuracy and the remaining system needed stable supervision.

## Decision: E5 continuous, not hard rescue
Reason:
hard rescue increased errors; continuous semantic scores preserve uncertainty.

## Decision: exclude NLI from final pseudo fusion
Reason:
NLI was globally weaker and often neutral on true radiology positives.

## Decision: macro-target loss
Reason:
the competition metric averages 12 target AUCs equally.

## Decision: keep v1.1 historical but use v1.2 as canonical
Reason:
v1.1 accidentally changed target gradient weighting.

## Decision: stop squeezing ResNet18 SliceMIL
Reason:
v1.2 remained right-censored, but architecture/representation changes had higher expected value.

## Decision: audit public DINO recipe before copying
Reason:
public leaderboard score combines architecture, folds, public weights, ranking and calibration; copying the final number as if it were one model would be misleading.

## Decision: Phase 4B-2A slot/crop audit first
Reason:
verify that 6-slot / 130-mm / multi-slice assumptions actually fit our dataset before expensive training.

## Decision: B2B1 used ResNet18 first
Reason:
isolate study-representation changes from backbone changes.

## B2B1 outcome
The representation trained mechanically but clean gold AUC fell to 0.6447.

Interpretation:
the experiment rejected the hypothesis that adding target-attention capacity to a ResNet18 student under current pseudo supervision is enough.

It did not reject:
- semantic slots;
- strong SSL backbones;
- dense windows;
- model-family diversity.

## New public 0.94 notebook decision
The two notebooks are not independent. Their main source prediction cells match exactly.

Therefore:
- count them as one recipe;
- use Bee's instrumentation as the auditable version;
- extract transferable ideas, not duplicated evidence.

## New strategic shift
The next innovation should focus on:
1. sequence semantics;
2. laterality;
3. strong pretrained representation;
4. weak-supervision robustness;
5. rank ensembles and diversity.
