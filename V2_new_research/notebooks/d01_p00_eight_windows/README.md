# D01: denser training-window sampling

Controlled comparison against P00's three-epoch full run. The sole experimental change is eight rather than four randomly sampled valid three-slice windows per MRI plane during training. Validation remains 16 deterministic windows per plane. The pretrained ResNet18, shared max pooling, 224-pixel preprocessing, weak report labels, frozen group split, seed, optimizer, learning rate, three epochs, and selection rule are unchanged.

`src/research/build_d01.py` regenerates the source and its embedded SHA256 manifest. The private Kaggle CPU preflight is in `notebooks/d01_cpu_check`; the bounded one-hour real-data GPU pilot is in `notebooks/d01_gpu_pilot`. Run those gates before full training. The pilot must report zero failed studies, finite steps/logits, verified split/label hashes, and projected three-epoch runtime under 7.5 hours. Full training has an eight-hour watchdog and saves separate `d01_*` outputs.

Evaluate against P00 on the identical 863 eligible development studies, reporting all 12 target AUCs and a paired group-bootstrap interval. The 58 expert-labeled studies remain sealed. This notebook makes no competition submission.
