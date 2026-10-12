# Decision log

## 2026-10-09 — Gated independent study approved

- Start from official CSVs, competition rules and independently reviewed public methods.
- Audit geometry, report text, split integrity and offline runtime before full image training.
- Propose a compact whole-study 2.5D baseline, then one-change experiments in coverage, pooling, preprocessing, NLP supervision, learning rate, loss and deployment.
- Keep the 58 expert-labeled studies out of image-model fitting and reserve them for a frozen check. Once used to select a model, they become development data.
- Public repository scores are author-reported; Reddit provides no controlled evidence for a model decision.
- Astra returned a conditional go for audits and a compact baseline after gates, and no-go for the full sweep. Its review is in `ASTRA_REVIEW.md`. The user approved M00 then M01 only after integrity, label and runtime gates pass. The user requires separate approval before the final competition submission.
- Model and training code is drafted primarily through the authenticated Claude CLI at the user's request, with Sol reviewing and verifying it. Claude wrote the M00 modules and most engineering pilot code; no full model training has occurred.

## 2026-10-09 — Sampling split frozen, model training still blocked

- The first sampled image screen found no exact cross-study matches but flagged 272 perceptual candidates. A private CPU follow-up found zero high-similarity matches, 101 sampled-dissimilar candidates, 170 incompatible/unresolved candidates, and one expert-linked pair checked by headers only. Astra noted that 93/100 matched controls were also incompatible, so absence of high-similarity matches is weak evidence.
- Candidate split v3 conservatively joins all 171 unresolved edges with report/patient groups and excludes all expert-connected components: 3,457 train, 885 validation, 65 excluded; zero detected image/report edges cross partitions. Astra approved manifest SHA-256 `b5926bd9d82154debee602a1824cf874aed1d0f409386308a387ed6a2f49debc` **for blinded report sampling only**. This contains detected candidate links; it does not establish patient independence.
- Local reviewer packets A (60 training-side reports) and B (120 held-out reports) were generated with blank state/evidence columns, one current group per report, and private UID maps. The user can arrange qualified clinical and language reviewers. No reviewer judgment or independent reference exists yet.
- Both allowed GPU engineering pilots have been used. The second completed 70/20/10 real-DICOM timing studies with synthetic targets and no retained weights, failed studies, nonfinite values or truncation. It ran 123.12 s on one Tesla T4, peak reserved VRAM 758 MiB. It produced no AUC or submission. Full eight-epoch M00 runtime projects 9.87 h with headroom using notebook steady rates, or 15.54 h using whole-phase wall rates, exceeding the approved 8 h. No full training until both label and runtime gates pass.

## 2026-10-09 — CPU cache alternatives fail disk and parity gates

- A private CPU-only 100-study Kaggle audit compared compressed resized float32 volumes against two fixed quantized uint16 draws. Both built/read all 100 and deleted their scratch caches. This used no GPU pilot.
- Float32 volumes preserve dynamic window selection, but project 50.00 GiB mean for the 3,457 training studies and differ from direct preprocessing by at most 1.79e-7 in the 10-study parity check. It fails the 15.5 GiB usable-disk gate and exact parity.
- Fixed uint16 draws project 19.36 GiB mean, exceeding the same disk gate. Their 10-study maximum quantization error is 7.63e-6 with exact masks, but fixed draws alter M00's dynamic sampling. Read rates are provisional because OS page-cache eviction was best-effort.
- Do not launch full M00. Next evidence is independent multilingual clinical report review and a revised, measured runtime/storage design within the approved GPU budget. The final competition submission still requires explicit user approval.

## 2026-10-10 — AI review received; full label gate remains closed

- The user cannot arrange clinician review and supplied AI-generated, multilingual A/B annotations. Both returned packets are complete and structurally valid, with 124/2,160 cells disputed between the two stated AI passes and adjudicated. The AI audit is in `AI_REVIEW_V1_AUDIT.md`.
- Only medial meniscus and effusion meet the predeclared 20-positive/20-negative support threshold in both A and B. These are report assertions, not MRI expert labels; structural agreement does not establish accuracy. The pass-isolation provenance is incomplete and the prompt's below-threshold rule conflicts with reviewer instructions.
- The prior clinical-review assumption is infeasible. Any move to an AI-only evaluation contract must be explicit and retain its limitations. Do not silently declare the existing 12-target gate passed. Full M00 training remains blocked by label and runtime gates; the 58 expert studies remain unopened for V2 selection.

## 2026-10-10 — Pragmatic three-epoch M00 route specified

- The user explicitly directed us to ignore the sparse 12-target support risk in the small AI-review packets and continue. This revises the development evaluation contract; it does not turn AI report judgments into expert MRI labels.
- A fixed public report-label v1 file (SHA-256 `96444cd3c7d396bae4a71454317d9cbe0689759b5278b30939781e04d0774537`) supplies masked hard labels for 3,428 of 3,457 training studies and all twelve targets have at least 20 positive/negative report proxies in the frozen 885-study validation fold. The public labels rank the AI-B report assertions at 0.997 macro AUC, which reflects shared report content and is not an image-model result. No protected expert values were read for this audit.
- Astra conditionally accepts three no-cache M00 epochs at a projected 5.83 h with 1.5× headroom (versus 15.54 h for eight) under the existing eight-hour cap. This changes the baseline recipe; M01 must match it. Exact Kaggle split/label inputs and guarded trainer remain before launch. The full contract is `M00_THREE_EPOCH_CONTRACT.md`.
- Claude CLI was authenticated but stalled twice, including a minimal request. No new model/training code was written with a fallback author. The public source and rules allow public external data, but notebook input attachment/hash verification remains pending. No full model training or submission was launched.

## 2026-10-10 — AI annotation substitutes for unavailable clinical reviewers

- The user cannot recruit qualified clinical or language reviewers for this competition and approved **AI (LLM) annotation** of packets A and B instead. These labels are AI-generated, report-derived and **not clinician reviewed**. Every returned row carries an `AI-LLM` reviewer ID and an AI note, and must never be described as radiologist or doctor reviewed.
- Protocol: two independent passes per report (Claude Opus and Claude Sonnet), then an AI adjudicator for disagreements and non-literal spans, using the same instructions and rubric. Packet B annotators and adjudicator see only packet B and are isolated from packet A, rules, weak labels and model outputs. The orchestrating agent inspects packet B only through checker counts.
- Caveat: packet B now measures agreement with an LLM reference, not clinical truth, and its errors may correlate with any LLM-derived rules. The label gate passed on these labels is an **AI-reference gate**. Outputs are in `outputs/ai_review_v1/`; merge and adjudication use `src/research/ai_review_merge.py`.
- Result (2026-10-10): both returned CSVs pass the structural checker (`complete`, 0 errors). The two passes agreed on 95.4% of cells in A and 93.7% in B; 33 A cells and 91 B cells were adjudicated. Packet B's 20/20 AI-reference support gate passes for **3/12** targets only (Medial Meniscus, Lateral Meniscus, Effusion), the same three as N01. **The label gate fails**; more held-out review is needed. SHA-256 of the returned A CSV is `8e14e15d…2a2e02` and of B is `652ad441…022745`. Neither is frozen yet.
## 2026-10-11 — Select M00 after matched M00/M01 GPU comparison

- Both approved three-epoch Kaggle GPU runs completed within their separate eight-hour caps, with matching frozen split/public-label hashes, 3,428 training and 863 eligible validation studies per epoch, and zero failed studies. Checkpoint hashes were independently verified after download.
- M00 epoch 3 reached 0.739719 macro AUC and 0.541696 masked BCE on report-derived weak labels in 2.62 hours, peak 760 MiB reserved VRAM. M01 epoch 3 reached 0.690701 and 0.572107 in 2.76 hours, peak 758 MiB.
- M00 ranks higher on all twelve findings. A paired 839-group, 1,000-draw bootstrap for M01 minus M00 gave −0.049019 macro AUC, percentile 95% interval [−0.060970, −0.038909]. Astra independently reviewed the gates and comparison. This is one seed per model and one selected checkpoint per model; it does not establish general superiority beyond the tested recipe.
- Freeze **M00 epoch 3** (checkpoint SHA-256 `783743cf59b6acea67b1fd3cf1b2ed11a9b818c615a69af9b6d61f73f785c721`) as the development candidate. Do not use M01 or a blend for the predeclared expert check. Prepare offline inference and a frozen evaluation script before reading the 58 expert values. No competition submission is authorized yet.
