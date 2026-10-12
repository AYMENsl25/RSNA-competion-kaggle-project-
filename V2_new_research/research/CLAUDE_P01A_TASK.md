Implement P01A preparation for an RSNA Knee Kaggle experiment in this repository. Work only under V2_new_research. Do not launch Kaggle/GPU or read expert labels.

Goal: pretrained timm `coatnet_0_rw_224.sw_in1k` backbone, SAME 2.5D data path, SAME shared max pooling and linear 12-logit head, SAME training schedule/split/labels as M00. This is a backbone-only comparison; do not implement finding attention yet.

Read `src/m00_model.py`, `src/m00_data.py`, `notebooks/m00_full_train/train_template.py`, `src/research/build_standalone_notebooks.py`, `notebooks/m00_full_train/kernel-metadata.json`. Create `src/p01a_model.py` with offline weight loader from a fixed file path: use timm model with `pretrained=False,num_classes=0`, load exact official safetensors state dict after removing classifier weights appropriately; strict backbone key matching and SHA256 check of weight file, no network. Official model safetensors SHA256 is `efe01c8c139a4b0c360f508d10ae1217f7e3f3431fd85824438ed1110773366c`. Input is 3x224x224 per window, output feature dim 768. Validate exact API against installed timm where possible; if unavailable, write clear requirements/gates.

Create a P01A full-training Kaggle script from M00 template via a P01A template + builder, altering only model construction, pretrained weight path/hash, output prefixes, source hashes, and environment logging. Keep M00 data and all experimental constants identical. Avoid direct imports after source embedding. Add `notebooks/p01a_full_train/kernel-metadata.json` with placeholders for a private attached weight dataset, official competition data, frozen split, report labels, GPU on, Internet off; do not push.

Add a lightweight synthetic CPU self-check that checks strict weight loading (with a fake state dict fixture if official weights unavailable), forward/backward shapes and no padded-window impact. Do not access expert studies. Do not change M00/M01 source. Document exact remaining gates, including offline weight dataset, timm version and runtime.

Use minimal, auditable changes. Do not claim tests passed unless run them.
