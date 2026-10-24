# F4 CPU screening trials

From the workspace root:

```powershell
.\V2_new_research\.venv\Scripts\python.exe -m unittest discover -s V2_new_research/tests -p test_f4_cpu_trials.py -v
.\V2_new_research\.venv\Scripts\python.exe V2_new_research/notebooks/f4_cpu_trials/run.py
```

The second command starts two CPU workers in parallel, each using two threads. It creates a fresh UTC-stamped output directory, with a shared 15-minute hard deadline. No GPU, downloads, Kaggle push, test inference or submissions occur. Run only with the existing completed F2 cache pinned in `config.json`.

`ranking` adds a 0.25-weighted pairwise logistic ranking term to masked BCE. `series_dropout` keeps BCE and hides whole series independently with probability 0.2 during training; at least one originally valid series remains. Both use F2 B's architecture and starting initialization, frozen inputs and 12-epoch schedule.

Each output folder contains history, selected checkpoint, validation probabilities, provenance and `result.json`. The supervisor records both exit codes. Checkpoint reload must reproduce every saved probability exactly. Results use the reused weak-label fold, not expert labels or the public leaderboard. A positive local result does not establish incremental benefit when blended with F0.
