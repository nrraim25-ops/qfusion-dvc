# Build Prompt for Antigravity — Full-Fidelity Quantum-Fusion Dense Video Captioning (STRICT v2 — Paper-Ready)

Copy everything below this line into Antigravity as the project brief. This version supersedes both earlier drafts. It adds three things a Q1 reviewer will demand and the earlier version lacked: (1) a parameter-matched classical control condition, so "quantum helped" can't be dismissed as "more parameters helped," (2) multi-seed runs with a real statistical test, so results aren't a single lucky/unlucky run, and (3) an auto-generated paper draft with related work pre-cited. Read Section 1 (Operating Rules) first — it governs every other section.

---

## 0. Honest scope statement

This reproduces the full thesis architecture end-to-end on ActivityNet Captions and compares **three** fusion conditions — classical baseline, parameter-matched classical control, and quantum — across **three random seeds each**, for a statistically defensible result. That is **9 full training runs** of the main pipeline (Gate 13), each requiring the same GPU/time budget as a single run in the earlier draft (16GB+ VRAM, likely a day or more per run depending on hardware and dataset size actually obtained). Total project time is realistically **multiple weeks** of GPU time, not days. This is the honest cost of the rigor a Q1 submission needs. If compute is genuinely constrained, reduce seeds from 3 to 2 as a documented, explicit deviation in `BLOCKED_LOG.md` (Rule 1.3) — do not silently run 1 seed and report it as if it were 3.

---

## 1. Operating Rules (unchanged from the prior strict version — still governs everything below)

**1.1 — No inference of missing specifications.** Stop and log to `BLOCKED_LOG.md` rather than guessing.
**1.2 — No placeholder, mocked, or fabricated outputs, ever.** No fake results under any circumstance, including time pressure.
**1.3 — The BLOCKED protocol.** Stop, log, mark `GATE_STATUS.md`, don't let downstream gates claim completeness on top of a blocked dependency.
**1.4 — Definition of "implemented" and "complete."** Only real execution against real data/tools, with an actually-run acceptance test, counts.
**1.5 — Every gate requires evidence before proceeding.** Gate-by-gate, in order, no batching.
**1.6 — No ambiguous quantifiers for required components.** Every "e.g."/"typically"/"or equivalent" for a required component has been resolved to one fixed decision below. If you find one that wasn't, that's a spec bug — stop and ask.
**1.7 — Optional work is walled off in the final gate only.** Nothing required is ever "if time allows."
**1.8 — NEW: No cherry-picking seeds or runs.** All 3 seeds per condition must be run and reported, including any seed that produced a poor or embarrassing result. Selectively reporting only favorable seeds is a Rule 1.2 violation (fabrication by omission).
**1.9 — NEW: Report the true outcome, whatever it is.** If the parameter-matched classical control performs comparably to or better than the quantum condition, this must be reported plainly and is not a failure of the project — it is a valid, publishable scientific result, and the paper draft (Section 13) must represent it honestly rather than being spun toward a predetermined conclusion.

---

## 2. Non-negotiable architectural constraints

- Every component in Section 7 implemented exactly as specified. No shortcuts.
- **Three fusion conditions, controlled via `--fusion_type classical|classical_matched|quantum`.** All three share identical code, backbones, hyperparameters, and data split for everything except the fusion module.
- Quantum simulation only, on `default.qubit` (Section 2 of the prior version, unchanged).
- Plain, portable `.py` files only. No notebooks.
- **Three random seeds per condition: `42`, `123`, `2024`.** Fixed, not chosen at runtime, applied identically across all three conditions for each seed value (i.e., seed `42` uses the same data shuffling/init pattern whether it's the classical, classical_matched, or quantum run).

---

## 3. Dataset

Unchanged from the prior strict version: official ActivityNet Captions annotations, `yt-dlp`-sourced videos with an honest, logged missing-video rate, `train`/`val_1`/`val_2` splits exactly as released, `val_2` required (not optional), 2.0-second snippets, 25fps/native-fps frame extraction, 224x224 crops, 16kHz mono audio.

---

## 4. Hard Gates

Same evidence discipline as before: implement → acceptance test → evidence file → update `GATE_STATUS.md` → proceed. Gates 1-6 (repo/env, dataset acquisition, preprocessing, ViViT, AST, TSP pretraining) are **unchanged from the prior strict version** — implement them exactly as previously specified (Sections 7.1-7.2 below repeat them for completeness). The gates below are new or modified.

### Gate 7 — Classical fusion (baseline)
Unchanged: Section 7.3. Acceptance test as before (shape check, gradient check).

### Gate 8 — NEW: Parameter-matched classical control fusion
**Do:** implement Section 7.4 (below) — a classical module deliberately matched in parameter count and structural depth to the quantum module, containing no quantum circuit at all.
**Acceptance test:** confirm the module's total trainable parameter count is between 68 and 76 (matching the quantum module's 72, within a small tolerance since exact equality may require adjusting bias terms) — print and log the exact count. Confirm shape `[T,768]+[T,768] -> [T,512]` and gradient flow, same as Gate 7.
**Evidence:** `gate_evidence/gate8_matched_classical_test.txt` — exact parameter count with a line-by-line breakdown of where each parameter comes from, so the match to the quantum module's 72 is independently verifiable by inspection.
**Why this gate exists, stated explicitly for whoever reads this repo later:** without this condition, any observed difference between "classical fusion" and "quantum fusion" is confounded with the fact that the quantum module also happens to add extra depth/nonlinearity. This condition isolates that confound. A reviewer will ask for this; do not skip it.

### Gate 9 — Quantum fusion
Unchanged: Section 7.5 (renumbered from 7.4 in the prior version). Acceptance test as before, including printing the actual circuit diagram.

### Gates 10-12 — Encoder, localization, captioning
Unchanged from the prior strict version (previously Gates 9-11). Implement exactly as previously specified.

### Gate 13 — Full integration and training, all three conditions × three seeds
**Do:** train all **9 combinations** (3 fusion conditions × 3 seeds) to completion, per Section 8's schedule.
**Acceptance test:** for each of the 9 runs, training loss is finite at every logged step, the run reaches its stopping condition, and a checkpoint is saved. All 9 runs must complete and be reported — per Rule 1.8, do not stop early on a subset and extrapolate.
**Evidence:** `gate_evidence/gate13_training_report.txt` — one row per run (9 rows: condition × seed), final epoch, stopping reason, final train/val loss, checkpoint path.

### Gate 14 — Evaluation with statistical testing
**Do:** run Section 9's evaluation protocol on all 9 trained models, on both `val_1` and `val_2`.
**Statistical analysis (required, new):** for each metric (localization F1@0.5, METEOR, CIDEr, BLEU-4), compute the mean and standard deviation across the 3 seeds for each of the 3 conditions. Run a paired statistical test — **Wilcoxon signed-rank test** (appropriate for n=3 paired seed-matched comparisons; do not use a t-test, which assumes a sample size this small can approximate normality, which it cannot) — comparing quantum vs. classical, and quantum vs. classical_matched, on each metric. Report the exact p-value for each comparison. Do not describe a result as "significant" if p >= 0.05; describe it exactly as what it is (e.g., "not statistically significant at the standard 0.05 threshold with n=3 seeds").
**Evidence:** `gate_evidence/gate14_evaluation_report.txt` — full mean±std table per condition per metric, plus the complete statistical test results table (comparison, metric, p-value, direction of effect).

### Gate 15 — NEW: Paper draft generation
**Do:** auto-generate a manuscript-structured document from the actual results (Section 13, below).
**Acceptance test:** every numeric claim in the generated draft traces to a specific file in `gate_evidence/` or `results/` — no number in the paper draft may be typed by hand or estimated; the generation script must pull every number programmatically from the saved CSVs/JSONs.
**Evidence:** the paper draft itself (`paper_draft/manuscript.md`), plus `gate_evidence/gate15_traceability_check.txt` confirming every numeric claim's source file.

### Gate 16 — Optional extensions
Unchanged from the prior strict version's Gate 14 (SCST fine-tuning, beam-width sweep, quantum architecture ablation). Still fully optional, still walled off, still no fake results if attempted.

---

## 5. Repository structure

```
qfusion-dvc/
  README.md
  requirements.txt
  setup.sh
  GATE_STATUS.md
  BLOCKED_LOG.md
  gate_evidence/
  configs/
    tsp_pretrain_video.yaml
    tsp_pretrain_audio.yaml
    main_classical.yaml
    main_classical_matched.yaml
    main_quantum.yaml
  data/  (unchanged from prior version)
  backbones/  (unchanged)
  models/
    fusion_classical.py
    fusion_classical_matched.py     # NEW
    fusion_quantum.py
    encoder_deformable.py
    localization_head.py
    context_mask.py
    captioning_head.py
    dvc_model.py
  utils/
    device.py
    stats.py                        # NEW: Wilcoxon test implementation/wrapper
  train_main.py                     # now takes --fusion_type classical|classical_matched|quantum --seed 42|123|2024
  evaluate.py
  make_plots.py
  generate_paper_draft.py           # NEW
  scst_finetune.py
  results/
    classical/seed_42/  classical/seed_123/  classical/seed_2024/
    classical_matched/seed_42/  classical_matched/seed_123/  classical_matched/seed_2024/
    quantum/seed_42/  quantum/seed_123/  quantum/seed_2024/
    plots/
    RESULTS_SUMMARY.md
  paper_draft/
    manuscript.md                   # NEW
    references.bib                  # NEW
```

---

## 6. Overall forward pass

Unchanged from the prior version, applies identically to all three fusion conditions:

```
raw video, raw audio (per video)
   -> snippet chunking
   -> ViViT(video_snippets) -> video_feats [T, 768]      (TSP-pretrained)
   -> AST(audio_snippets)   -> audio_feats [T, 768]       (TSP-pretrained)
   -> FUSION module (Section 7.3 / 7.4 / 7.5, condition-dependent) -> fused_feats [T, 512]
   -> deformable + sparse-attention encoder                -> encoder_out [T, 512]
   -> event-query localization head                        -> 30 predicted events
   -> differentiable context mask, per event                -> soft-pooled context vector [512]
   -> captioning decoder, per event                          -> generated caption
```

---

## 7. Component specifications

### 7.1-7.2 Backbones, TSP pretraining
Unchanged from the prior strict version. Implement exactly as previously specified (Kinetics-pretrained ViViT `google/vivit-b-16x2-kinetics400`, AudioSet-pretrained AST `MIT/ast-finetuned-audioset-10-10-0.4593`, TSP foreground/background pretraining with the 50%-overlap threshold rule and the 20-epoch/patience-3 stopping rule).

### 7.3 Classical fusion (baseline) — `models/fusion_classical.py`
Unchanged: linear `768->512` per modality, concat, linear `1024->512`, LayerNorm. This has substantially more parameters than the quantum module (roughly `768*512*2 + 1024*512 ≈ 1.3M` parameters) — this asymmetry is expected and fine; this condition answers "how does quantum fusion compare to the thesis's actual original design," while Gate 8's condition answers the parameter-matched question. Both comparisons matter and are reported separately, never conflated.

### 7.4 NEW — Parameter-matched classical control — `models/fusion_classical_matched.py`
Designed to mirror the quantum module's exact shape (compress to 6 numbers total, "process" in 4 layers, expand) with an ordinary classical nonlinearity instead of a quantum circuit, at a matched parameter budget.
1. Classical compress: linear `video_feats [768] -> 3`, linear `audio_feats [768] -> 3` (identical to the quantum module's compression step, same parameter count in this part — this stage is *not* part of the matched budget, it's shared infrastructure common to both conditions).
2. Concatenate the two 3-vectors into one 6-vector (mirrors the quantum module having 6 total wires holding the encoded information).
3. "Processing," 4 layers, each layer: one `6x6` linear transform, **no bias term** (to match the quantum module's rotation gates, which have no separate bias), followed by `tanh` nonlinearity (mirrors the quantum module's rotation gates being bounded, nonlinear operations) — `4 layers x 6x6 = 144`... **this overshoots the 72 target by 2x; to hit the 68-76 parameter target exactly, use a `6x3` down-projection then `3x6` up-projection per layer instead of a full `6x6`** (`4 layers x (6*3 + 3*6) = 4 x 36 = 144`... still 2x over). **Correct construction to hit ~72 parameters:** use 4 layers of a single `6x3` linear (no bias) each, i.e., `4 x 18 = 72` parameters exactly, where each layer's 3-dim output is added residually back into a running 6-dim state via a fixed (non-trainable) broadcast — implement this precisely: each layer computes `delta = tanh(Linear_6to3(state))`, then updates `state[0:3] += delta` and `state[3:6] += delta` (a fixed, untrainable coupling between the two halves, deliberately analogous to the quantum module's fixed cross-modal CNOT wiring, which is also structural rather than trainable). This gives exactly `4 x (6x3) = 72` trainable parameters, matching the quantum module's 72 exactly, with an explicit, documented, structurally-analogous (not just numerically-matched) design — the strongest possible version of this control.
4. Classical expand: linear `6 -> 512`, LayerNorm (identical to the quantum module's expand step).
**Gate 8's acceptance test requires printing this exact parameter breakdown to confirm the count is exactly 72, matching the quantum module exactly rather than merely falling in a range.**

### 7.5 Quantum fusion — `models/fusion_quantum.py`
Unchanged from the prior strict version (6 qubits, 3 video/3 audio, 4 layers, 72 trainable quantum parameters, explicit cross-modal CNOT wiring `[2,3]` and `[5,0]`, `default.qubit` device).

### 7.6-7.9 Encoder, localization, context mask, captioning decoder
Unchanged from the prior strict version. Implement exactly as previously specified (deformable attention with 4 sampling points, sparse pruning to 50% after layer 2, 30 event queries with deep supervision and the exact Hungarian cost weights `5/2/1`, the differentiable sigmoid context mask, GloVe `glove.840B.300d` embeddings, beam search width 5).

---

## 8. Training configuration

Unchanged per-run configuration from the prior strict version (AdamW, two parameter groups at `1e-4`/`1e-5`, linear warmup + cosine decay, gradient clipping at 1.0, weight decay `1e-4`, batch size starting at 4 with logged auto-halving on OOM, 50-epoch cap with patience-5 early stopping) — **applied identically to all 9 runs** (3 conditions × 3 seeds `{42, 123, 2024}`). The only thing that varies between the 9 runs is the fusion module (Section 7.3/7.4/7.5) and the seed value, which controls data shuffling order and weight initialization identically across conditions for a given seed.

---

## 9. Evaluation — `evaluate.py`

Unchanged core protocol (official `densevid_eval` tool, tIoU thresholds `{0.3,0.5,0.7,0.9}`, BLEU/METEOR/CIDEr, `val_1` and `val_2` both required) — run on all 9 trained models. Statistical analysis per Gate 14 above (mean±std across 3 seeds per condition, Wilcoxon signed-rank test between conditions, exact p-values reported without significance-threshold spin).

---

## 10. Required plots and outputs — `make_plots.py`

All plots from the prior strict version, updated to show **three conditions with error bars (std across 3 seeds)** instead of two conditions with single-run values:
1. `tsp_boundary_sharpness.png` (unchanged, backbone-level, not condition-dependent).
2. `loss_curves.png` — mean training curve ± std band across seeds, one line+band per condition (3 total), train and val separately.
3. `localization_metrics_bar.png` — grouped bars with error bars (std across seeds), 3 conditions × 4 tIoU thresholds.
4. `captioning_metrics_bar.png` — grouped bars with error bars, 3 conditions × 3 metrics (BLEU-4/METEOR/CIDEr).
5. `val2_metrics_bar.png` — same as 3-4, for `val_2`.
6. `literature_comparison.png` — all 3 conditions' mean F1@0.5 and METEOR alongside literature baselines.
7. `fused_feature_space.png` — PCA scatter, **three** side-by-side subplots (classical / classical_matched / quantum), one representative seed.
8. `sparsification_keep_ratio.png` (unchanged, encoder-level).
9. `statistical_significance.png` — **NEW**: a table-style figure (rendered via matplotlib) showing every pairwise comparison's p-value and effect direction from Gate 14, color-coded (not spun) by whether it clears p<0.05.
10. `qualitative_examples.md` — extended to 3-way: ground truth vs. classical vs. classical_matched vs. quantum, same 15-video selection rule as before.

---

## 11. `RESULTS_SUMMARY.md`

Extended from the prior version to include: all 3 conditions' mean±std metrics (both val splits), the full statistical test table, and the "Deviations and BLOCKED Items" section pulled from `BLOCKED_LOG.md`/`GATE_STATUS.md`. The auto-templated summary paragraph (per Rule 1.9) must state the true outcome plainly, including if the matched-classical control performs comparably to or better than quantum — this is not framed as a negative result, it's framed as the actual finding.

---

## 12. Portability requirements

Unchanged from the prior strict version (Section 11.5 there) — plain `.py` files, no hardcoded paths, device-agnostic utility, `default.qubit` baseline, pinned `requirements.txt`, `setup.sh`, config-driven runs, README "run on a fresh machine" section.

---

## 13. NEW — Paper draft generation — `generate_paper_draft.py`, Gate 15

Auto-generate `paper_draft/manuscript.md`, a full manuscript-structured draft, with every number pulled programmatically from `results/` and `gate_evidence/` (Gate 15's acceptance test requires this traceability). Structure:

### 13.1 Title (template, fill in venue-appropriate wording, do not invent results-based claims in the title)
"[Working title] A Hybrid Quantum-Classical Fusion Layer for Multimodal Dense Video Captioning: An Empirical Study"

### 13.2 Abstract (auto-templated from real numbers)
Must state: the task, the architecture, the three-condition ablation design, and the actual headline result (whichever direction it went), plus the exact seed count and statistical test used. Do not let the abstract claim "improves performance" unless Gate 14's statistical test actually showed a significant improvement for that specific metric — if results are mixed, the abstract must say so.

### 13.3 Related Work (pre-populated citation list — write actual paragraphs synthesizing these, do not just list them)
Pull directly from this project's own literature review document, specifically:
- **Dense video captioning generally:** Krishna et al. (original DVC + ActivityNet Captions), PDVC (Wang et al., set prediction), SDVC (Mun et al., sequential event selection), SBS (Choi et al., explicit event counting).
- **Multimodal fusion in DVC:** MDVC/BMT (Iashin & Rahtu, late fusion), EVMF (Zhong et al., LLM-augmented fusion), the ACM Computing Surveys paper (Qasim et al.) as the field-level evidence that fusion depth is a named, unresolved gap.
- **Quantum-inspired (non-circuit) multimodal work — must cite and explicitly differentiate:** the Quantum Echo Imaging Framework (QEIF) — state plainly that QEIF uses a classical, metaphor-based "quantum-inspired" module (no actual qubits or circuit) for single-image captioning, whereas this work uses an actual variational quantum circuit (real qubits, real entangling gates, simulated via PennyLane) applied to video, with temporal localization, and includes audio as a second modality — this is the precise novelty delta and must be stated exactly this way, not vaguely.
- **Quantum ML more broadly:** quanvolutional networks (Henderson et al.) and quantum reservoir computing, as evidence that hybrid quantum-classical layers are an active, if still small-scale, research area, with an honest note that neither has previously been applied to video or to captioning/generation tasks.

### 13.4 Method (pulled from Section 7's component descriptions, rewritten in prose, with the exact architectural numbers — 6 qubits, 4 layers, 72 parameters, the exact entangling wiring — stated precisely, not approximately)

### 13.5 Experimental Setup (pulled from Sections 3, 8, 9 — dataset size actually obtained, per Gate 2's real numbers, not the official split's nominal size; training configuration; evaluation protocol; the three-condition, three-seed design explicitly justified as addressing the parameter-count confound and single-run unreliability)

### 13.6 Results (every number pulled from Gate 14's evidence files — full tables with mean±std, the statistical test table, and prose describing the actual pattern found, written to match Rule 1.9)

### 13.7 Limitations (auto-templated, must include at minimum): simulator-only quantum execution (no real hardware noise modeling); dataset size constrained by YouTube video availability (exact percentage from Gate 2); ActivityNet-specific results may not generalize to other domains (directly relevant given the earlier surveillance/satellite-video discussion in this project, if that direction is mentioned in the paper's motivation); n=3 seeds is a minimum for statistical testing, not a large sample, and effect sizes should be interpreted accordingly.

### 13.8 Conclusion (auto-templated from the real headline result, matching Rule 1.9 — no predetermined "quantum wins" framing baked into the template)

### 13.9 `references.bib` — auto-populated BibTeX entries for every work cited in 13.3, pulled from this project's own literature review document's citation details.

---

## 14. Final checklist

- [ ] `GATE_STATUS.md` shows all 15 required gates COMPLETE (Gate 16 optional items separately marked).
- [ ] `BLOCKED_LOG.md` accurately reflects every BLOCKED event.
- [ ] All 9 training runs (3 conditions × 3 seeds) completed — verify by counting checkpoint files, not by reading a summary claim.
- [ ] Gate 8's parameter count for the matched-classical control is confirmed exactly 72 (or the documented, justified alternate construction), not merely "close."
- [ ] Gate 14's statistical test was run with the correct test (Wilcoxon, not a t-test) and exact p-values are reported, not just "significant"/"not significant" labels.
- [ ] No cherry-picked seeds anywhere (Rule 1.8) — spot check by confirming exactly 3 seed-level result files exist per condition in `results/`.
- [ ] `RESULTS_SUMMARY.md` and `paper_draft/manuscript.md` represent the true outcome, including if quantum did not outperform the matched-classical control (Rule 1.9) — spot check the abstract and conclusion against the actual results table for consistency.
- [ ] Every numeric claim in `paper_draft/manuscript.md` is traceable to a specific evidence file (Gate 15's acceptance test, `gate_evidence/gate15_traceability_check.txt`).
- [ ] QEIF and the other Section 13.3 works are cited and the novelty delta from QEIF is stated precisely, not vaguely.
- [ ] Portability checks from the prior strict version's final checklist, unchanged, all still pass.
