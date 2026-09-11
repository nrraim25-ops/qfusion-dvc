# Dense Video Captioning Experimental Results Summary

## 1. Executive Summary
This document records the empirical results of the Full-Fidelity Quantum-Fusion Dense Video Captioning (`qfusion-dvc`) study on ActivityNet Captions across three controlled fusion conditions and five random seeds (`42`, `123`, `2024`, `7`, `999`).

Per Operating Rule 1.9, results are reported objectively without spin:
- **Classical Baseline** (~1.3M fusion params): Strong representational capacity on temporal proposal detection and sentence generation.
- **Matched Classical Control** (exactly 72 params): Parameter-matched baseline with identical dimensionality and bottleneck structure.
- **Quantum Fusion Layer** (6 qubits, 72 rotation params): Variational quantum circuit operating on simulated statevectors on `default.qubit`.

## 2. Quantitative Performance Across Validation Splits (Mean ± Std)

### 2.1 val_1 Split (Primary Benchmark)
| Condition | Trainable Params | F1@0.3 | F1@0.5 | F1@0.7 | F1@0.9 | BLEU-4 | METEOR | CIDEr |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| `classical` | 1,313,280 | 0.488±0.105 | 0.316±0.073 | 0.147±0.062 | 0.066±0.044 | 1.34±0.18 | 5.19±0.96 | 0.20±0.04 |
| `classical_matched` | 72 | 0.459±0.059 | 0.319±0.047 | 0.147±0.040 | 0.044±0.040 | 1.09±0.32 | 4.30±0.72 | 0.15±0.05 |
| `quantum` | 72 | 0.543±0.056 | 0.404±0.042 | 0.231±0.036 | 0.070±0.029 | 1.49±0.38 | 6.23±1.11 | 0.23±0.06 |

### 2.2 val_2 Split (Generalization Benchmark)
| Condition | Trainable Params | F1@0.3 | F1@0.5 | F1@0.7 | F1@0.9 | BLEU-4 | METEOR | CIDEr |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| `classical` | 1,313,280 | 0.513±0.112 | 0.338±0.090 | 0.196±0.064 | 0.073±0.030 | 0.77±0.17 | 3.35±1.06 | 0.10±0.04 |
| `classical_matched` | 72 | 0.487±0.088 | 0.305±0.073 | 0.160±0.060 | 0.015±0.021 | 0.71±0.27 | 3.37±1.02 | 0.10±0.04 |
| `quantum` | 72 | 0.567±0.078 | 0.389±0.061 | 0.200±0.036 | 0.069±0.021 | 0.91±0.36 | 4.26±1.75 | 0.14±0.05 |

## 3. Paired Wilcoxon Signed-Rank Hypothesis Tests (n=5 Seeds)

| Comparison Pair | Metric | Mean Difference | Wilcoxon Stat | p-value | Effect Direction | Clears p < 0.05 |
|:---|:---|:---|:---|:---|:---|:---|
| `classical` vs `quantum` | f1_0.3 | -0.0550 | 5.00 | 0.6250 | classical < quantum | NO |
| `classical` vs `quantum` | f1_0.5 | -0.0881 | 3.00 | 0.3125 | classical < quantum | NO |
| `classical` vs `quantum` | f1_0.7 | -0.0844 | 2.00 | 0.1875 | classical < quantum | NO |
| `classical` vs `quantum` | f1_0.9 | -0.0037 | 7.00 | 1.0000 | classical < quantum | NO |
| `classical` vs `quantum` | bleu4 | -0.1501 | 4.00 | 0.4375 | classical < quantum | NO |
| `classical` vs `quantum` | meteor | -1.0322 | 5.00 | 0.6250 | classical < quantum | NO |
| `classical` vs `quantum` | cider | -0.0225 | 4.00 | 0.4375 | classical < quantum | NO |
| `classical_matched` vs `quantum` | f1_0.3 | -0.0844 | 2.00 | 0.1875 | classical_matched < quantum | NO |
| `classical_matched` vs `quantum` | f1_0.5 | -0.0844 | 0.00 | 0.0625 | classical_matched < quantum | NO |
| `classical_matched` vs `quantum` | f1_0.7 | -0.0844 | 1.00 | 0.1250 | classical_matched < quantum | NO |
| `classical_matched` vs `quantum` | f1_0.9 | -0.0257 | 3.00 | 0.3750 | classical_matched < quantum | NO |
| `classical_matched` vs `quantum` | bleu4 | -0.3936 | 2.00 | 0.1875 | classical_matched < quantum | NO |
| `classical_matched` vs `quantum` | meteor | -1.9229 | 0.00 | 0.0625 | classical_matched < quantum | NO |
| `classical_matched` vs `quantum` | cider | -0.0802 | 0.00 | 0.0625 | classical_matched < quantum | NO |
| `classical` vs `classical_matched` | f1_0.3 | +0.0294 | 5.50 | 0.6875 | classical > classical_matched | NO |
| `classical` vs `classical_matched` | f1_0.5 | -0.0037 | 7.50 | 1.0000 | classical < classical_matched | NO |
| `classical` vs `classical_matched` | f1_0.7 | +0.0000 | 6.00 | 0.8125 | equal | NO |
| `classical` vs `classical_matched` | f1_0.9 | +0.0220 | 4.00 | 0.3750 | classical > classical_matched | NO |
| `classical` vs `classical_matched` | bleu4 | +0.2435 | 1.00 | 0.1250 | classical > classical_matched | NO |
| `classical` vs `classical_matched` | meteor | +0.8908 | 2.00 | 0.1875 | classical > classical_matched | NO |
| `classical` vs `classical_matched` | cider | +0.0578 | 0.00 | 0.0625 | classical > classical_matched | NO |

## 4. Documented Deviations and BLOCKED Items
Pulled directly from `BLOCKED_LOG.md` per Operating Rule 1.3:
- **DEC-001 (Gate 2)**: Scoped raw dataset download to representative stratified subset due to local disk constraint (88 GB free) and YouTube video availability rates; verified and tracked honest unavailable rates.
- **DEC-002 (Gate 13 & 14)**: Resolved initial constant METEOR fallback by downloading NLTK WordNet, building 5,000-token vocabulary, and wiring caption loss into training.
- **DEC-003 (Gate 14 & 15)**: Replaced decoupled cross-event multi-reference caption evaluation with official joint DVC proposal-gated evaluation (tIoU >= 0.5 matching, single ground-truth reference, 0.0 penalty for unlocalized events), aligning evaluation with literature standards.
- **DEC-004 (Gate 20)**: Scoped dataset scale-up audit to verified 160-video archive due to 83.3% YouTube link rot / bot wall rate.

## 5. Artifacts and Generated Evidence
- Checkpoints: `results/{classical,classical_matched,quantum}/seed_{42,123,2024,7,999}/model_best.pt`
- Loss curves: `results/plots/loss_curves.png`
- Evaluation reports: `gate_evidence/gate14_evaluation_report.txt`
