# N1 — random-window attention, laptop run

User authorized implementation and local execution on October 15, 2026.

The model reuses the existing finding-specific gated attention in `src/v3_model.py` and the shared `train_v3.py` training/inference components. The only modeling change from E8 is **max pooling -> attention**. No new model dependency or extra data source is needed.

- ConvNeXt-Tiny `fb_in22k_ft_in1k`, same cached ImageNet initialization as E8.
- 3 slots; **4 random adjacent-slice windows per slot for training**, 16 deterministic windows per slot for validation. No fixed slab.
- Resolution 224 from the existing canonical 256 cache; existing laterality mirroring.
- v3 soft targets, unknown cells masked, positive weights clipped at 10.
- AdamW LR 1e-4, weight decay 0.05, cosine schedule, fp16, seed 0.
- **12 epochs**, batch 2, validation batch 2, no augmentation or averaging.
- Laptop RAM adjustment only: 1 persistent worker per loader, prefetch 1, OMP/MKL threads 2. Progress and the existing thermal guard are checked every 100 steps. Historical trainer defaults stay unchanged.

Twelve epochs is the matched first test, not a claim of an optimal stopping point. E8 public AUC was 0.873944 / 0.874283 / 0.874449 at epochs 10/11/12. E3 similarly flattened. If attention is promising, compare both heads using a separately planned longer schedule; blindly extending a completed cosine schedule whose LR is already zero is not that test.

## Preflight actually completed

- Exactly 3,428 eligible training studies and 885 validation studies; no overlapping known groups and no excluded/expert study loaded.
- All five cache index hashes match their shard summaries; array dtypes, dimensions and declared byte counts match. The full 50-GiB pixel arrays were not rehashed for this launch.
- Random windows repeat for the same seed/epoch and change across epochs.
- Common encoder and classifier initialization match the max-head control exactly.
- Three actual-cache fp16 optimizer updates have finite encoder and attention gradients. One initial overflow was skipped by GradScaler while calibrating its scale.
- Two-study, 16-window-per-slot validation is finite. Arbitrary/NaN pixels in masked windows do not affect predictions.
- Peak allocated GPU memory: about **1,882 MiB training**, **1,362 MiB validation** in the smoke check. The smoke model was discarded; full training starts fresh.

Before launch the GPU was idle at 36 C, with 6,141 MiB total VRAM. The laptop had about 2.7 GiB available RAM and 467 GiB free disk, and was on external power. Existing dashboard processes were stopped and the dashboard was restarted; HTTP 200 was verified at http://127.0.0.1:8050.

## Files and completion

- Entry point: `src/run_n1.py` (offline ImageNet cache only).
- Live stdout: `outputs/n1_attn_random.log`; separate stderr next to it.
- Run directory: `outputs/n1_attn_random/`.
- `preflight.json`, `run_manifest.json`, source snapshots, and `resources_at_launch.json` record the check, configuration, sources, environment and process identity.
- `log.json` and per-epoch validation predictions follow the same format as existing experiments; the dashboard discovers the run automatically.
- `best.pt` follows the existing v3-label selection rule. Primary comparison uses **epoch 12**, not an epoch chosen after inspecting public-label AUC. `last.pt` includes its state and optimizer/scheduler state.
- After training the launcher automatically computes 1,000-draw paired group-bootstrap comparisons against E8 epoch 12 on both fixed public and v3 labels.
- A predeclared secondary diagnostic compares `0.80 * Ens5 + 0.20 * percentile_rank(N1)` against Ens5. This is not an authorized final submission or permission to tune weights.
- `run_status.json` says `running`, `failed` or `completed`; completion is written only after the full training and comparisons finish.

Resume only with `.venv/Scripts/python src/run_n1.py --resume`; the launcher refuses changed source/input/configuration identities. It reuses the trainer's existing epoch-boundary resume behavior, not a bitwise mid-epoch replay. Use the original two-thread environment when resuming.

Do not launch another GPU trainer concurrently. The last gold-58 look remains untouched. Expected runtime is provisional until full epoch timing is available; the earlier E8 took 1.62 GPU-hours for 12 epochs.
