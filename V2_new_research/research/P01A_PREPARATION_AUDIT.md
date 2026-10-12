# P01A CoAtNet backbone preparation — 2026-10-11

## Frozen comparison

P01A swaps M00's randomly initialized ResNet18 encoder for the official ImageNet-1k
`timm/coatnet_0_rw_224.sw_in1k` encoder. It inherits `M00.forward` verbatim: three MRI
planes, the same 3-slice windows, valid-window-only encoding, shared feature max and one
12-logit linear head. The 768-feature head replaces M00's 512-feature head by necessity.
M00 data, split, weak labels, masking, batch sizes, AdamW 3e-4, cosine schedule, three
epochs, AMP, and 8-hour hard cap are unchanged. No expert-label studies are accessed.

Official model card: https://huggingface.co/timm/coatnet_0_rw_224.sw_in1k . Model
license Apache-2.0; trained on ImageNet-1k. Its 224x224 fixed input matches the existing
M00 window geometry. The official 109,813,430-byte safetensors file is mounted from
private Kaggle dataset `aymenslimani/rsna-v2-official-coatnet0-imagenet1k` and checked
against SHA256 `efe01c8c139a4b0c360f508d10ae1217f7e3f3431fd85824438ed1110773366c`
before loading. The 1000-class classifier is removed; all remaining keys load strictly.
No network path exists in the model loader or kernel.

## Gates completed

- Trainer template diff against M00 was reviewed: only model and weight metadata,
  timm/safetensors imports, environment versions, and P01A output names changed.
- Generated trainer, CPU preflight, and GPU pilot passed Python syntax compilation.
- Private Kaggle CPU preflight v2 completed with Internet and GPU off. It verified
  checksum, timm creation, strict checkpoint loading, [1,12] finite output, backward,
  and padded-window invariance. Kaggle CPU image: torch 2.11.0+cpu, timm 1.0.29,
  safetensors 0.8.0. Result: `outputs/p01a_cpu_preflight/p01a_cpu_preflight.json`.
- A prior preflight v1 failed only when writing its JSON due to an undefined result
  metadata variable; the computation had completed. v2 fixed this and passed.

## Open gates

- The bounded P01A GPU pilot v1 launched at
  https://www.kaggle.com/code/aymenslimani/rsna-knee-v2-p01a-gpu-pilot . Training
  reached validation, then failed at 142 s because the copied P00 pilot treated
  `b["errs"]` (a list) as a dictionary. The P01A generator now counts `len(b["errs"])`.
  Kaggle refused the corrected v2 push while two account batch GPU sessions were
  occupied; retry when a slot frees. Its measured throughput, peak VRAM, AMP behavior,
  failed studies, and projected full runtime must be reviewed before any full run.
- A full P01A result, if runtime gates pass, must be compared with M00 on the identical
  863-study weak-label validation set. Keep the 58 expert cases sealed.
- A final Kaggle inference notebook is not prepared or submitted.
