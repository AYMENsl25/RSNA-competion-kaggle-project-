# Experiment Ledger

## Overall scores

| phase      | name                                          | type               |   macro_auc |   macro_ap | notes                                                                                                                               |
|:-----------|:----------------------------------------------|:-------------------|------------:|-----------:|:------------------------------------------------------------------------------------------------------------------------------------|
| 4A-1       | Rule teacher v0.2                             | report NLP         |    0.6871   | nan        | Strict AUC 0.6871; broad AUC 0.7080; strict coverage 28.59%; covered balanced accuracy 0.8496; 22 hard errors.                      |
| 4A         | Rule v0.2 + continuous multilingual E5        | report NLP teacher |    0.76067  | nan        | Best trustworthy report teacher. NLI diagnostic was not used in final fusion.                                                       |
| 4B-1B v1.1 | ResNet18 SliceMIL historical baseline         | MRI                |    0.70977  |   0.5334   | Historical baseline. Used unintended global weight normalization; selection best at epoch 4/4 and therefore right-censored.         |
| 4B-1B v1.2 | ResNet18 SliceMIL corrected macro-target loss | MRI                |    0.699358 |   0.528926 | Canonical clean loss baseline. Best epoch 8/8, still right-censored.                                                                |
| 4B-2B1     | Six-slot 2.5D ResNet18 + target attention     | MRI                |    0.644729 |   0.480957 | Selection had a real optimum at epoch 5; final 5-epoch refit completed. Lower pseudo loss did not translate to better gold ranking. |

## v1.2 vs B2B1 target AUC

| target           |   v1_2_auc |   b2b1_auc |   delta_b2b1_minus_v1_2 |
|:-----------------|-----------:|-----------:|------------------------:|
| ACL              |   0.843137 |   0.688725 |               -0.154412 |
| MCL              |   0.614512 |   0.503401 |               -0.111111 |
| Medial Meniscus  |   0.621394 |   0.663462 |                0.042068 |
| Lateral Meniscus |   0.631056 |   0.637267 |                0.006211 |
| Medial OA        |   0.717829 |   0.683721 |               -0.034108 |
| Lateral OA       |   0.775629 |   0.605416 |               -0.170213 |
| PF OA            |   0.746461 |   0.701416 |               -0.045045 |
| Effusion         |   0.814907 |   0.816149 |                0.001242 |
| Synovitis        |   0.709677 |   0.648746 |               -0.060931 |
| Baker's          |   0.655797 |   0.637681 |               -0.018116 |
| Contusion        |   0.622132 |   0.545209 |               -0.076923 |
| Fracture         |   0.640278 |   0.605556 |               -0.034722 |

## Phase 4B-2A slot coverage

| slot      |   n_studies_present |   fraction_studies_present |
|:----------|--------------------:|---------------------------:|
| SAG_FS    |                4150 |                   0.941684 |
| SAG_NONFS |                4266 |                   0.968005 |
| COR_FS    |                4248 |                   0.963921 |
| COR_NONFS |                3406 |                   0.772861 |
| AX_FS     |                4407 |                   1        |
| AX_NONFS  |                 857 |                   0.194463 |

## Phase 4B-2A 16-slice feasibility

| slot      |   n_slot_series |   median_readable |   median_band_count |   fraction_has_16_in_band |
|:----------|----------------:|------------------:|--------------------:|--------------------------:|
| AX_FS     |            4407 |                32 |                  26 |                  0.952122 |
| AX_NONFS  |             857 |                34 |                  28 |                  0.995333 |
| COR_FS    |            4248 |                30 |                  24 |                  0.89548  |
| COR_NONFS |            3406 |                30 |                  24 |                  0.869055 |
| SAG_FS    |            4150 |                30 |                  24 |                  0.93012  |
| SAG_NONFS |            4266 |                30 |                  24 |                  0.977262 |

## 130-mm crop

- selected slot-series: 21,334
- valid physical metadata: 1.000000
- crop fits: 0.997797
- failures: 47
