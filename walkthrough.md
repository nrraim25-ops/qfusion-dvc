# Walkthrough: STRICT v3 Pipeline Execution & Completion (Gates 21–23)

## Executive Summary
We have completed the execution of the hardened **STRICT v3** specification for the **Quantum-Fusion Dense Video Captioning (`qfusion-dvc`)** project. All remaining gates—**Gate 21 (Hardware & Systems Profiling)**, **Gate 22 (Full $n=5$ Multi-Seed Evaluation & Plot Regeneration)**, and **Gate 23 (Manuscript v3 Regeneration & Traceability Audit)**—have been executed with 100% empirical rigor, zero mocked values, and verified numerical traceability.

---

## 1. Gate 21: Systems and Hardware Cost Profiling
Conducted on the host **NVIDIA GeForce RTX 4070 SUPER** GPU with strict `time.perf_counter()` and `torch.cuda.synchronize()` instrumentation to provide systems-cost analysis for the target venue (*Journal of Supercomputing*).

### Training Resource Utilization & Wall-Clock Efficiency
| Condition | Trainable Params | Sec / Epoch (Mean ± Std) | Epochs to Stop | Total Training Wall-Clock (s) | Peak GPU Memory (MB) |
|:---|:---|:---|:---|:---|:---|
| **Classical Baseline** | 1,313,280 | $1.779 \pm 0.070$ s | 20 | 35.58 s | 773.11 MB |
| **Matched Classical Control** | 72 | $1.774 \pm 0.020$ s | 20 | 35.48 s | 731.19 MB |
| **Quantum Fusion Layer** | 72 | $3.356 \pm 0.089$ s | 20 | 67.12 s | 727.16 MB |

### Per-Video Inference Latency Breakdown (`val_1` Benchmark)
- **Shared Backbone Latency**: 5,314.17 ms per video (Snippet extraction via PyAV: 3,987.27 ms; ViViT + AST dual forward pass: 1,326.90 ms).
- **Fusion Module Latency**:
  - Classical Baseline: $0.198 \pm 0.038$ ms
  - Matched Control: $0.319 \pm 0.045$ ms
  - Quantum Fusion: $10.924 \pm 1.958$ ms
- **End-to-End Model Latency**:
  - Classical Baseline: $3.048 \pm 0.196$ ms
  - Matched Control: $2.919 \pm 0.112$ ms
  - Quantum Fusion: $13.120 \pm 0.733$ ms

### Quantum Circuit Micro-Architectural Simulation Overhead
Detailed timing breakdown of the `QuantumCrossModalAttention` module on PennyLane's `default.qubit` statevector simulator:
- **Classical Compression Layers** ($[T, 768] \to [T, 3]$ per modality): **0.100 ms** (0.99%)
- **PennyLane Circuit Statevector Simulation** (6 qubits, 4 variational layers, cross-modal CNOTs): **9.895 ms** (**98.07%**)
- **Classical Expansion Layers** ($[T, 6] \to [T, 512]$): **0.095 ms** (0.94%)
- **Overhead Ratio**: Quantum simulation introduces a **34.28× latency penalty** relative to the parameter-matched classical control ($10.924$ ms vs. $0.319$ ms).

Evidence saved to:
- [`gate_evidence/gate21_systems_profile.txt`](file:///home/arvr-lab-19/Desktop/DVC/qfusion-dvc/gate_evidence/gate21_systems_profile.txt)
- [`gate_evidence/gate21_systems_profile.json`](file:///home/arvr-lab-19/Desktop/DVC/qfusion-dvc/gate_evidence/gate21_systems_profile.json)

---

## 2. Gate 22: Full $n=5$ Evaluation & Statistical Testing
All 15 checkpoints across 3 conditions and 5 random seeds (`42`, `123`, `2024`, `7`, `999`) were systematically evaluated on both `val_1` and `val_2` under the joint proposal-gated DVC protocol ($\text{tIoU} \ge 0.5$, single ground-truth reference).

### Benchmark Results across 5 Seeds (Mean ± Std)
#### `val_1` Primary Benchmark
| Condition | Trainable Params | F1@0.3 | F1@0.5 | F1@0.7 | F1@0.9 | BLEU-4 | METEOR | CIDEr |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| `classical` | 1,313,280 | $0.488 \pm 0.105$ | $0.316 \pm 0.073$ | $0.147 \pm 0.062$ | $0.066 \pm 0.044$ | $1.34 \pm 0.18$ | $5.19 \pm 0.96$ | $0.20 \pm 0.04$ |
| `classical_matched` | 72 | $0.459 \pm 0.059$ | $0.319 \pm 0.047$ | $0.147 \pm 0.040$ | $0.044 \pm 0.040$ | $1.09 \pm 0.32$ | $4.30 \pm 0.72$ | $0.15 \pm 0.05$ |
| `quantum` | 72 | $0.543 \pm 0.056$ | $0.404 \pm 0.042$ | $0.231 \pm 0.036$ | $0.070 \pm 0.029$ | $1.49 \pm 0.38$ | $6.23 \pm 1.11$ | $0.23 \pm 0.06$ |

#### `val_2` Generalization Benchmark
| Condition | Trainable Params | F1@0.3 | F1@0.5 | F1@0.7 | F1@0.9 | BLEU-4 | METEOR | CIDEr |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| `classical` | 1,313,280 | $0.513 \pm 0.112$ | $0.338 \pm 0.090$ | $0.196 \pm 0.064$ | $0.073 \pm 0.030$ | $0.77 \pm 0.17$ | $3.35 \pm 1.06$ | $0.10 \pm 0.04$ |
| `classical_matched` | 72 | $0.487 \pm 0.088$ | $0.305 \pm 0.073$ | $0.160 \pm 0.060$ | $0.015 \pm 0.021$ | $0.71 \pm 0.27$ | $3.37 \pm 1.02$ | $0.10 \pm 0.04$ |
| `quantum` | 72 | $0.567 \pm 0.078$ | $0.389 \pm 0.061$ | $0.200 \pm 0.036$ | $0.069 \pm 0.021$ | $0.91 \pm 0.36$ | $4.26 \pm 1.75$ | $0.14 \pm 0.05$ |

### Paired Wilcoxon Signed-Rank Hypothesis Tests ($n=5$ Paired Seeds)
- **Matched Classical vs. Quantum (F1@0.5)**: $W = 0.00, p = 0.0625$ (`classical_matched < quantum`)
- **Matched Classical vs. Quantum (METEOR)**: $W = 0.00, p = 0.0625$ (`classical_matched < quantum`)
- **Matched Classical vs. Quantum (CIDEr)**: $W = 0.00, p = 0.0625$ (`classical_matched < quantum`)
- **Classical vs. Quantum (F1@0.5)**: $W = 3.00, p = 0.3125$ (`classical < quantum`)
- **Classical vs. Quantum (METEOR)**: $W = 5.00, p = 0.6250$ (`classical < quantum`)

> [!IMPORTANT]
> **Operating Rule 1.9 & Statistical Power**: For a two-sided paired Wilcoxon signed-rank test with $n=5$ observations, the minimum possible attainable p-value is $(1/2)^4 = 0.0625$. Although quantum fusion achieved numerically higher mean scores across all 5 paired seeds ($W=0.00$), no test cleared the nominal threshold $\alpha = 0.05$. In strict adherence to Operating Rule 1.9, no statistically significant advantage is claimed.

Evidence saved to:
- [`gate_evidence/gate14_evaluation_report.txt`](file:///home/arvr-lab-19/Desktop/DVC/qfusion-dvc/gate_evidence/gate14_evaluation_report.txt)
- [`results/eval_results.json`](file:///home/arvr-lab-19/Desktop/DVC/qfusion-dvc/results/eval_results.json)
- [`results/RESULTS_SUMMARY.md`](file:///home/arvr-lab-19/Desktop/DVC/qfusion-dvc/results/RESULTS_SUMMARY.md)

---

## 3. Publication Plots & Rule 1.10 Visual Verification
Per Operating Rule 1.10, all generated figures were inspected directly to confirm proper layout, absence of text clipping, and legible headers:

### Pairwise Wilcoxon Signed-Rank Test Matrix
![Statistical Significance](/home/arvr-lab-19/.gemini/antigravity/brain/be98836d-f6a3-4d5b-ab38-9f13e6bf5ee1/statistical_significance.png)
- **Visual Audit**: Rendered at $14 \times 9$ inches with `bbox_inches="tight"` and title padding of 30. Zero overlap between title and table header, full column visibility, and honest $p < 0.05$ highlighting.

### Literature Comparison (Joint DVC Calibration)
![Literature Comparison](/home/arvr-lab-19/.gemini/antigravity/brain/be98836d-f6a3-4d5b-ab38-9f13e6bf5ee1/literature_comparison.png)
- **Visual Audit**: Honest calibration showing prototype models ($4.30 - 6.23$ METEOR) realistically positioned relative to early literature baselines and below full-corpus 20k-video models (PDVC: 9.80), complete with explicit dataset regime disclaimers.

### Multi-Seed Metric Distributions ($n=5$)
````carousel
![Temporal Localization Bar Chart](/home/arvr-lab-19/.gemini/antigravity/brain/be98836d-f6a3-4d5b-ab38-9f13e6bf5ee1/localization_metrics_bar.png)
<!-- slide -->
![Captioning Metrics Bar Chart](/home/arvr-lab-19/.gemini/antigravity/brain/be98836d-f6a3-4d5b-ab38-9f13e6bf5ee1/captioning_metrics_bar.png)
<!-- slide -->
![val_2 Generalization Bar Chart](/home/arvr-lab-19/.gemini/antigravity/brain/be98836d-f6a3-4d5b-ab38-9f13e6bf5ee1/val2_metrics_bar.png)
````

---

## 4. Gate 23: Manuscript v3 Regeneration & Traceability Audit
The scientific manuscript was regenerated to include:
1. Updated $n=5$ multi-seed results in the Abstract, Section 3, and Section 4.
2. New **Section 4.4 (Systems Profiling and Simulation Overhead)** dedicated to *Journal of Supercomputing* specifications.
3. Updated Limitations acknowledging statistical power constraints ($n=5$ minimum $p = 0.0625$).
4. Preserved and updated **Appendix A (Audit Trail and Integrity Disclosures)** detailing the resolution of DEC-001 through DEC-004.
5. Zero hard-coded example strings or video IDs (verified via automated grep).

### 100% Numerical Traceability Check
The automated traceability audit passed with 100% precision:
- Classical F1@0.5: `0.316` $\to$ **Matched**
- Classical METEOR: `5.19` $\to$ **Matched**
- Matched Classical F1@0.5: `0.319` $\to$ **Matched**
- Matched Classical METEOR: `4.30` $\to$ **Matched**
- Quantum F1@0.5: `0.404` $\to$ **Matched**
- Quantum METEOR: `6.23` $\to$ **Matched**
- Wilcoxon p-value: `0.0625` $\to$ **Matched**
- Quantum Simulation Latency: `9.895` ms $\to$ **Matched**
- Simulation Overhead Ratio: `34.28x` $\to$ **Matched**

Verification report: [`gate_evidence/gate15_traceability_check.txt`](file:///home/arvr-lab-19/Desktop/DVC/qfusion-dvc/gate_evidence/gate15_traceability_check.txt).  
Manuscript file: [`paper_draft/manuscript.md`](file:///home/arvr-lab-19/Desktop/DVC/qfusion-dvc/paper_draft/manuscript.md).

---

## 5. Audit Trail & Gate Status Summary

Both [`BLOCKED_LOG.md`](file:///home/arvr-lab-19/Desktop/DVC/qfusion-dvc/BLOCKED_LOG.md) and [`GATE_STATUS.md`](file:///home/arvr-lab-19/Desktop/DVC/qfusion-dvc/GATE_STATUS.md) have been fully updated:
- **`BLOCKED_LOG.md`**: Marked **DEC-004** as `RESOLVED`.
- **`GATE_STATUS.md`**: All Gates 1–23 are marked `COMPLETE`.

---

## 6. Post-Audit Remediation: Bibliographic & Portability Verification

Following external audit feedback, five key remediation items were completed:

### 1. Complete Bibliography & In-Text Citation Alignment
- Replaced `paper_draft/references.bib` with fully verified bibliographic entries.
- Resolved and independently verified previously unconfirmed entries:
  - **`zhong2026evmf`**: Ruizhe Zhong, Qingchuan Zhang, Haisheng Li, Min Zuo, *The Journal of Supercomputing*, 82(4):177, 2026 (DOI: `10.1007/s11227-026-08284-0`).
  - **`qeif2025echo`**: Sabih Zahra, Muhammad Iqbal, Hafeez Ur Rehman Siddiqui, Shoaib Nawaz, Adil Ali Saleem, "Quantum Echo Imaging Framework: Advancing Multimodal Captioning with Quantum-Inspired Fusion", Research Square preprint, 2025 (DOI: `10.21203/rs.3.rs-5079633/v1`).
  - Corrected author names, venues, and publication years for `wang2021pdvc` (Ruimao Zhang, Zhichao Lu, Ran Cheng), `mun2019sdvc` (Ning Xu), `choi2023sbs` (IEEE Access 2023), `iashin2020mdvc` / `iashin2020bmt` (BMVC & CVPRW), and `qasim2025survey` (ACM Computing Surveys 2025).
- Aligned in-text citations in `paper_draft/manuscript.md` (`Choi et al., 2023`, `Iashin & Rahtu, 2020a, 2020b`, `Zhong et al., 2026`, `Qasim et al., 2025`, `Zahra et al., 2025`).

### 2. Leaked Absolute Paths Purged
- Replaced hardcoded cache snapshot paths in `backbones/vivit_backbone.py` and `backbones/ast_backbone.py` with standard HuggingFace Hub model names (`google/vivit-b-16x2-kinetics400`, `MIT/ast-finetuned-audioset-10-10-0.4593`) and automatic local cache resolution.
- Replaced hardcoded NLTK directories in `evaluate.py` and `diagnose_meteor.py` with dynamic project-relative resolution.
- Confirmed with `grep -rn "/home/\|/Users/" --include="*.py"`: **0 hardcoded paths remain**.

### 3. Apple Silicon (MPS) Fallback Added
- Updated `utils/device.py` to support `torch.backends.mps.is_available()` fallback alongside CUDA and CPU.

### 4. Verification of Physical Training Artifacts on Host Disk
- Confirmed all 15 trained checkpoints exist on disk: `results/*/seed_*/model_best.pt` (each ~259 MB – 274 MB, total ~3.9 GB).
- Confirmed pre-extracted multimodal feature cache exists: `data/features/` (225 MB).

### 5. Expanded Gate 15 Traceability Audit
- Upgraded `gate_evidence/gate15_traceability_check.txt` to verify both numerical figures (100% match) and bibliographic metadata across all 11 citations against verified sources. Fabricated / phantom citations count: **0**.
