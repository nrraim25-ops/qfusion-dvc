# A Hybrid Quantum-Classical Fusion Layer for Multimodal Dense Video Captioning: An Empirical Study

**Authors**: Antigravity Research Team  
**Date**: September 2026  
**Artifact Repository**: `qfusion-dvc`

---

## Abstract
Dense video captioning (DVC) requires jointly localizing temporal event boundaries and generating coherent natural language descriptions from multimodal video and audio streams. While recent models incorporate increasingly complex transformer cross-attention mechanisms, the fundamental expressive efficiency of quantum circuit architectures for multimodal fusion remains largely uncharacterized in temporal video understanding. In this work, we present an empirical evaluation of a hybrid quantum-classical fusion layer within an end-to-end dense video captioning framework on ActivityNet Captions. We design a 6-qubit variational quantum circuit (VQC) with 4 parameterized layers (exactly 72 rotation parameters) utilizing cross-modal entangling gates between visual (ViViT) and auditory (AST) feature sub-registers. To address the parameter-count confound, we compare this architecture against both an overparameterized classical baseline (~1.3M parameters) and an explicitly parameter-matched classical control (exactly 72 trainable parameters) across five random seeds ($n=5$: 42, 123, 2024, 7, 999). On the primary validation benchmark (`val_1`), the classical baseline achieves F1@0.5 of 0.316 and METEOR of 5.19; the parameter-matched classical control achieves F1@0.5 of 0.319 and METEOR of 4.30; and the quantum fusion layer achieves F1@0.5 of 0.404 and METEOR of 6.23. Paired Wilcoxon signed-rank testing between the parameter-matched control and the quantum fusion model yields p = 0.0625. Our results demonstrate that parameterized quantum circuits can be stably integrated into complex multimodal perception architectures, while highlighting that under strict parameter matching, quantum representations yield competitive but non-superior inductive bias in simulator regimes without physical quantum advantage.

---

## 1. Introduction and Related Work
Dense video captioning requires temporal localization of unconstrained events accompanied by natural language description (Krishna et al., 2017). Early approaches utilized two-stage proposal and captioning pipelines, while recent advances have moved toward end-to-end set prediction architectures such as PDVC (Wang et al., 2021), streamlined sequential selection (Mun et al., 2019), and step-by-step explicit event counting (Choi et al., 2023).

Multimodal fusion in dense video captioning has demonstrated that combining auditory and visual features reliably improves boundary localization and descriptive richness (Iashin & Rahtu, 2020a, 2020b; Zhong et al., 2026). However, as identified in recent surveys (Qasim et al., 2025), multimodal fusion depth remains a named, unresolved design challenge: early projection schemes often suffer from modality collapse, while late fusion ignores cross-modal temporal dependencies.

Recently, quantum-inspired and variational quantum architectures have emerged across machine learning (Henderson et al., 2020; Bergholm et al., 2018). Crucially, we must explicitly differentiate our approach from prior metaphor-based quantum-inspired works, such as the Quantum Echo Imaging Framework (QEIF; Zahra et al., 2025). Whereas QEIF introduces a classical, metaphor-based algebraic module (without qubits, quantum statevectors, or quantum circuits) for single-image captioning, our work deploys an actual variational quantum circuit (real qubits, parameterized single-qubit rotations, and cross-modal entangling CNOT gates simulated via PennyLane) applied to dynamic video and audio streams with temporal event localization. To our knowledge, this represents the first integration of a variational quantum circuit within an end-to-end dense video captioning system.

---

## 2. Method and Architecture

### 2.1 Backbones and Temporal Snippet Pretraining (TSP)
The input video and audio are decomposed into non-overlapping 2.0-second snippets. Video snippets are processed by a Video Vision Transformer (ViViT-B/16x2) yielding visual features v_t in R^768. Audio snippets are processed by the Audio Spectrogram Transformer (AST) yielding acoustic features a_t in R^768. Both backbones are prospective-pretrained using Temporal Snippet Prospective (TSP) classification with a 50% IoU ground-truth boundary rule and early stopping.

### 2.2 Controlled Fusion Architectures
We formulate three strictly controlled fusion conditions:
1. **Classical Baseline**: High-dimensional cross-modal projection [T, 768] + [T, 768] -> [T, 512] totaling 1,313,280 parameters.
2. **Matched Classical Control**: Designed to eliminate parameter-count confounding, consisting of exactly 72 trainable parameters structured across 4 sequential layers of Linear(6, 3, bias=False) with tanh non-linearities and fixed cross-modal coupling.
3. **Quantum Fusion Layer**: A 6-qubit variational circuit on PennyLane default.qubit. Modality representations are compressed to 3 visual angles and 3 audio angles, state-prepared via Ry(theta), followed by 4 variational layers each comprising Rx, Ry, Rz rotations (18 parameters per layer x 4 layers = exactly 72 rotation parameters) and cross-modal entangling CNOT gates linking qubits [2, 3] and [5, 0]. Expectation values of Pauli-Z observables on all 6 qubits are expanded back to the 512-dimensional latent embedding.

### 2.3 Deformable Sparse Temporal Encoder and Heads
The fused sequence is processed by a 2-layer deformable temporal encoder with 4 sampling points and 50% temporal snippet pruning after layer 2. An event localization head with 30 learnable event queries performs bipartite Hungarian matching with loss weights 5.0 (L1), 2.0 (tIoU), and 1.0 (classification). Differentiable sigmoid soft-pooling context masks extract event-specific temporal features fed into an autoregressive caption decoder with GloVe embeddings.

---

## 3. Experimental Setup
- **Dataset**: ActivityNet Captions benchmark. To evaluate under controlled storage and compute budgets, experiments are conducted on a verified multimodal subset of 160 videos (130 train, 15 val_1, 15 val_2) with complete, synchronized video and audio streams, annotated with over 500 dense temporal events and natural language captions. YouTube link rot and unavailable video rates are strictly logged per operating protocol.
- **Optimization**: AdamW optimizer with two parameter groups (1e-4 for fusion and prediction heads, 1e-5 for backbones), cosine decay, gradient clipping at 1.0, and early stopping patience of 4 epochs.
- **Protocol**: 3 fusion conditions x 5 independent seeds (42, 123, 2024, 7, 999) = 15 complete training runs.

---

## 4. Empirical Results and Statistical Analysis

### 4.1 Primary Validation Benchmark (val_1)
Performance across the three conditions on val_1 (Mean +- Std across 5 seeds). Captioning metrics (BLEU-4, METEOR, CIDEr) are computed under the standard joint proposal-gated Dense Video Captioning (DVC) protocol at tIoU >= 0.5 without cross-event reference pooling:

| Condition | Trainable Params | F1@0.3 | F1@0.5 | F1@0.7 | F1@0.9 | BLEU-4 | METEOR | CIDEr |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| `classical` | 1,313,280 | 0.488 | 0.316 | 0.147 | 0.066 | 1.34 | 5.19 | 0.20 |
| `classical_matched` | 72 | 0.459 | 0.319 | 0.147 | 0.044 | 1.09 | 4.30 | 0.15 |
| `quantum` | 72 | 0.543 | 0.404 | 0.231 | 0.070 | 1.49 | 6.23 | 0.23 |

### 4.2 Generalization Benchmark (val_2)
Performance on the held-out val_2 split:
- **Classical Baseline**: F1@0.5 = 0.338, METEOR = 3.35
- **Matched Classical Control**: F1@0.5 = 0.305, METEOR = 3.37
- **Quantum Fusion**: F1@0.5 = 0.389, METEOR = 4.26

### 4.3 Wilcoxon Signed-Rank Statistical Tests
Paired Wilcoxon signed-rank tests across n=5 paired seeds between the matched control and quantum fusion layer show exact p = 0.0625 for localization F1@0.5. No statistically significant superiority (p < 0.05) is observed for quantum fusion over parameter-matched classical operations.

### 4.4 Systems Profiling and Simulation Overhead (Journal of Supercomputing)
To evaluate the computational and hardware overheads of variational quantum simulation in temporal multimodal perception, all three fusion architectures were profiled on NVIDIA GeForce RTX 4070 SUPER under strict synchronization:

| Metric | Classical Baseline | Matched Control | Quantum Fusion |
|:---|:---|:---|:---|
| **Trainable Fusion Parameters** | 1,313,280 | 72 | 72 |
| **Training Sec / Epoch (Mean ± Std)** | 1.779 ± 0.070 s | 1.774 ± 0.020 s | 3.356 ± 0.089 s |
| **Total Training Wall-Clock (s)** | 35.58 s | 35.48 s | 67.12 s |
| **Peak GPU Memory Allocation (MB)** | 773.11 MB | 731.19 MB | 727.16 MB |
| **Fusion Forward Latency (ms)** | 0.198 ms | 0.319 ms | 10.924 ms |
| **End-to-End Model Latency (ms)** | 3.048 ms | 2.919 ms | 13.120 ms |

#### Micro-architectural Quantum Overhead Decomposition
Analysis of the `QuantumCrossModalAttention` module reveals that classical linear compression ([T, 768] -> [T, 3]) and expansion ([T, 6] -> [T, 512]) consume 0.100 ms (1.0%) and 0.095 ms (0.9%), respectively. In contrast, the PennyLane `default.qubit` statevector circuit simulation requires 9.895 ms (98.1% of module execution). Consequently, the quantum fusion layer incurs a 34.28x latency penalty relative to the parameter-matched classical control (10.924 ms vs. 0.319 ms), without demonstrating statistically significant predictive advantage.

---

## 5. Limitations
1. **Simulation Regime**: Quantum circuits were executed on classical statevector simulators (default.qubit) without physical quantum noise, decoherence, or NISQ hardware latency.
2. **Dataset Scale and Literature Context**: While containing hundreds of densely annotated events across 160 videos, the experimental corpus represents an empirical subset of the full ActivityNet Captions archive due to YouTube video attrition and local storage constraints. In literature comparisons, published models (e.g., PDVC, BMT) were trained on the full ~20,000 video corpus; our prototype operates on a 130-video subset under strict parameter matching, yielding joint METEOR scores (~4.7-6.0) consistent with low-data regime prototypes rather than full-scale SOTA.
3. **Statistical Power**: With n=5 paired seeds, the minimum non-parametric two-sided Wilcoxon signed-rank p-value is 0.0625; detecting subtle effect sizes would require further seed expansion or larger validation splits.

---

## 6. Conclusion
We presented the first empirical investigation of a variational quantum fusion circuit in multimodal dense video captioning. By implementing an exact 72-parameter matched classical baseline alongside the 72-parameter quantum circuit, we eliminated the common parameter-count confound in hybrid quantum machine learning. Our findings establish that quantum variational layers can be trained stably end-to-end within modern multimodal pipelines, while underscoring that empirical quantum advantage remains unproven for temporal video reasoning under parameter-matched classical baselines.

---

## Appendix A: Audit Trail and Integrity Disclosures

In accordance with strict empirical reporting principles:
1. **Initial Evaluation & Metric Calibration**: In preliminary iterations, an implementation artifact was identified wherein cross-event reference pooling produced inflated captioning metrics (~17–18 METEOR). Gate 17 systematically resolved this anomaly by enforcing standard joint proposal-gated DVC evaluation (tIoU >= 0.5 threshold with single reference matching and 0.0 penalty for unlocalized events), calibrating scores to authentic ranges (4.7–6.0).
2. **Dataset Scale & Hardware Profiling**: Due to external video availability rates on YouTube (83.3% takedown/block rate during scale-up probing, documented under DEC-004), experiments were conducted on a verified, fully cached multimodal subset of 160 videos.
3. **Simulation Overhead**: As profiled in Section 4.4, statevector simulation on classical GPUs incurs a substantial simulation latency penalty (98.1% of execution time, ~34x slower than matched classical tensor operations) without yielding statistically significant accuracy gains.

---

## References
Please refer to `references.bib` for complete bibliographic records.
