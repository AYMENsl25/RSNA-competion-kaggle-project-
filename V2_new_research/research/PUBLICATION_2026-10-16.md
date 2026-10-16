# Completed-work publication: October 16, 2026

This snapshot publishes the accumulated research source, notebooks, tests, aggregate evidence, and decision notes. It was prepared in a separate local clone so the active research checkout and running jobs remain untouched.

## Boundaries

- N4's pilot bundle, resolution test, dedicated note, and registry row are excluded at the user's request. `src/canon.py` is taken from `outputs/n3a_attn6/source/canon.py`, preserving the fixed 256-pixel implementation before the local N4 extension. Shared N1/N3 code and historical proposals remain available.
- N3b code and its launch protocol are included. The source note records launch evidence, not a final result; no completion is inferred from publication.
- All 289 previously committed V1 paths are preserved under `V1_previous_research/`, with identical Git blobs in the relocation commit. This also retains older files missing from the live local archive. New historical plans and matched preprocessing notebooks follow separately.
- Credentials, Python environments, raw data, patient-level labels/review packets, downloaded caches, checkpoints, and generated run outputs are not newly uploaded. Previously committed V1 artifacts remain in history and in the relocated archive.
- `s01_series_coverage.json` and `NEXT_SIX_EVIDENCE_AUDIT.json` are aggregate audit evidence. Their source/input hashes connect the written decisions to the local assets; the underlying patient-level files stay local.
- `experiments/completed_runs/` preserves nine completed V3 run histories, N1/N3a paired comparison summaries, and a SHA-256 manifest. These are aggregate metrics and historical settings, without patient-level predictions.
- Downloaded public notebooks and reader code retain their authorship and source identifiers. See `notebooks/e12_blend/README.md` and the research source notes for provenance. Reference evidence is not a claim of an independently trained result.

## Reading order

Start with the study README, source guide, and notebook guide. Then read the dated baseline results, V3 plan/progress record, N1 result, N3a refresh, and N3b launch note. Earlier claims about pending results or sealed expert studies describe their historical state; the current README records the later authorized E01 diagnostic look.

## Verification

The publication copy passed 43 CPU regression tests. One optional test requiring the official CoAtNet weight file was skipped. An existing S01 test incorrectly expected a 10-slice series to fill 16 distinct three-slice windows; its expectation now checks the eight valid triplets and masked zero padding. The loader and historical training bundles are unchanged.

Standalone canonicalization, multilingual relabeling, and N3b CPU compatibility checks also passed. The N3b check covered padding invariance, positional sensitivity, gradients, rejection of invalid input, frozen N3a sampling, and exact legacy logits. The canonicalization check used the local pilot DICOM fixture; frozen N1/N3a source fixtures were available for parity checks. Those ignored local fixtures are not published.

Static checks parsed 85 Python files, 30 JSON files, and 42 notebook JSON documents across the initial publication set; no credential patterns were found. The archive relocation was checked against all 289 original Git blobs. New overview and guide links resolve, dedicated N4 files are absent, and the canonicalizer matches the frozen pre-N4 source. Completed-run JSON evidence was additionally parsed and checked for patient-level table keys before copying.

Verification runtime: Python 3.12.10, NumPy 2.5.2, PyTorch 2.11.0+cu128, torchvision 0.26.0+cu128, timm 1.0.30, pydicom 3.0.2, SciPy 1.18.1, and pytest 9.1.1. CUDA was disabled for checks. Pytest was installed only into the publication clone's ignored tools folder. The suite emitted pydicom deprecation warnings. GPU/cache training checks and Kaggle end-to-end runs were not repeated while another experiment was active.

No training, new competition submission, or N4 evaluation is part of publishing this snapshot.
