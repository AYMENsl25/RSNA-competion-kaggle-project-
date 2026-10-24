# N4 early checkpoint decision and N5 handoff

User explicitly requested waiting for epoch3 or epoch4 results, stopping N4 if they are low, and starting N5. N4 was not stopped at request time (epoch3 still training).

One-time rule: compare completed epochs3/4 with N3a at the identical epoch. If BOTH fixed public and v3 macro AUC are more than 0.03 lower, stop N4 after its completed checkpoint is saved and launch N5. Otherwise wait from epoch3 to epoch4; if neither meets the rule, leave N4 running. N3a public references: epoch3 0.81990957, epoch4 0.85132002. N4 observed public epoch1 0.55069544 / epoch2 0.56231241. This is an operational futility rule requested during the run, not a preregistered final12 result or evidence that resolution itself fails. Root cause of unusually poor learning is not yet established.

The one-time local process src/watch_n4_epoch_gate.py is waiting; the previously paused 15-minute app automation remains paused. It identifies N4 process command lines and descendants before stopping, preserves all predictions and best/last checkpoints, and records the decision. No N5 GPU work runs concurrently with N4. The guard exits after either decision and does not create a recurring schedule.

N5 is two fresh seed0 runs using N1 attention (three slots, native256 cache / input224):
1. Uniform LR 1e-4, 16 epochs.
2. Encoder LR 3e-5 and all non-encoder parameters LR 1e-3, 16 epochs.
Both use identical initialization, v3 soft labels/0.5 masks, posweight cap10, batch, windows, cosine multiplier, weight decay and augmentation. No N4 weights or pixels used. No EMA/OneCycle/simultaneous augmentation change. Primary comparison final16 vs final16; uniform16 vs N1 final12 is only a secondary schedule readout. One fixed 80/20 addition to the accepted N3a ensemble is recorded separately. Promote on paired macro gain >=0.003 with positive CI or fixed ensemble gain >=0.002 with positive CI.

src/run_n5.py performs a real-cache CUDA smoke check after N4 stops, discards smoke weights, snapshots source and input hashes, runs both arms sequentially, and automatically compares fixed endpoints. Tests passed exact uniform optimizer update parity, complete/disjoint differential parameter groups, cosine LR-ratio preservation and gate wait/stop/continue cases. Outputs: outputs/n5_lr, n5_uniform16 and n5_differential16. Guard outputs: outputs/n4_epoch_gate.

Paused at user request for laptop shutdown: uniform16 completed public AUC0.8896348502; differential saved epoch1 public0.8123387357. Original checkpoints and optimizer/scheduler/AMP state verified on CPU. Local N5 processes stopped. Resume via src/resume_n5.py, which preserves original source/input hashes and skips the completed control. N6 remote run unaffected.

User requested continuing after shutdown. Both saved checkpoints, original source/input hashes, optimizer, cosine scheduler and AMP state verified again. Uniform16 skipped; differential16 resumed after epoch1 with its original16-epoch schedule. Hidden resume launcher11540, controller16032 at launch. Comparison-only helper was fixed to import NumPy; training sources/configurations unchanged. Dashboard restarted and verified HTTP200 with N5 visible. The 15-minute schedule remains paused. Launch record outputs/n5_lr/resume_launch.json.
