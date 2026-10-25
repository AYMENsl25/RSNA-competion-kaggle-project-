# RSNA Knee Abnormality Detection: our Kaggle project

Predict 12 knee MRI findings (ACL, MCL, menisci, OA in 3 compartments, effusion, synovitis, Baker's cyst, contusion, fracture) per study, scored by mean AUC. The competition ran on Kaggle and closed 2026-10-22.

## Final results

| What | Public LB | Private LB |
|---|---:|---:|
| **Our best own model: Ens4** (4 CNNs we trained, competition data only) | 0.857 | 0.837 |
| Fork of the public 0.958 stack (F0) | 0.958 | 0.942 |
| Top private score | | 0.960 |

No medal. The technical report examines supervision, image coverage, model selection and compute as possible contributors to the gap; the experiments do not isolate a single cause.

## Start here

| Document | What it covers |
|---|---|
| [Technical report](docs/technical-report/REPORT.md) · [PDF](docs/technical-report/rsna-knee-technical-report.pdf) | Architecture, controlled ablations, replication, final scores and reproducibility |
| [docs/RESULTS.md](docs/RESULTS.md) | Every submission with public and private scores, every development experiment, gold-58 checks |
| [docs/OUR_BEST_MODEL.md](docs/OUR_BEST_MODEL.md) | Deep dive on Ens4: preprocessing, sampling, model, labels, training, ensembling, per-finding AUC, why it fell short |
| [docs/LESSONS_FROM_TOP_SOLUTIONS.md](docs/LESSONS_FROM_TOP_SOLUTIONS.md) | Reusable playbook distilled from the 2nd, 5th, 7th and 8th place solutions |
| [V2_new_research/research/POSTMORTEM_21ST_22ND_2026-10-24.md](V2_new_research/research/POSTMORTEM_21ST_22ND_2026-10-24.md) | Comparison with the 21st and 22nd place solutions |

## Repository layout

```
docs/                      Final write-ups (start here)
V2_new_research/           The main study (Oct 9-24)
  research/                Plans, decision records, results notes, external evidence
  src/                     Preprocessing (canon.py), models, training (train_v3.py), audits, runners
  notebooks/               Kaggle kernels: cache builds, training, offline inference, F0-F7 submissions
  experiments/             registry.csv + completed-run histories with paired comparisons
  tests/                   Unit tests for data, models and submission builders
  scripts/                 F3-F7 submission builders and preview checks
V1_previous_research/      Archive of the first project attempt
```

## Timeline

| Dates | Phase | Outcome |
|---|---|---|
| Sep - Oct 8 | V1: first attempt | Archived in `V1_previous_research/` |
| Oct 9-12 | V2: audits, frozen split, baselines (M00, P00, E01) | First submission 0.769 |
| Oct 13-15 | V3: canonical cache, labels v3, ensembles | Ens4 **0.857** public / 0.837 private |
| Oct 15-17 | N-series: attention heads, 6 slots, native res, LR, seed replication | Weak-label gains; none submitted |
| Oct 17-22 | Public-weights track: F0 fork, K/T/S fine-tunes and students, F-variants | F0 0.958 / 0.942; no variant beat it |
| Oct 24-25 | Postmortem and top-solution review | `docs/` |

## Data and weights

Competition data, MRI caches, report text, label tables and checkpoints are **not** in Git, per the competition rules. They stay local or in private Kaggle datasets. `.kaggle/` holds local credentials and is ignored.
