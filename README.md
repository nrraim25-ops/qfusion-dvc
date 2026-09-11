# Full-Fidelity Quantum-Fusion Dense Video Captioning (`qfusion-dvc`)

This repository contains the complete implementation, training pipeline, multi-seed evaluation benchmark, and manuscript artifacts for **Full-Fidelity Quantum-Fusion Dense Video Captioning on ActivityNet Captions**.

---

## 1. Quick Start / Run on a Fresh Machine

To reproduce all experiments, evaluations, and manuscript figures from scratch:

```bash
# 1. Clone repository and enter directory
cd /path/to/qfusion-dvc

# 2. Run environment setup
bash setup.sh
source venv/bin/activate

# 3. Download ActivityNet Captions dataset and annotations
python download_dataset.py

# 4. Run Temporal Snippet Pretraining (TSP)
python train_tsp.py

# 5. Run full multi-seed experiment suite (3 conditions x 3 seeds = 9 runs)
python run_all_experiments.py

# 6. Evaluate all checkpoints on val_1 and val_2 splits with Wilcoxon statistical testing
python evaluate.py

# 7. Generate all figures and plots
python make_plots.py

# 8. Programmatically generate manuscript and run 100% numerical traceability audit
python generate_paper_draft.py
```

---

## 2. Controlled Experimental Conditions

To strictly isolate quantum inductive bias without parameter-count confounding, we evaluate three conditions across 3 seeds (`42`, `123`, `2024`):

1. **`classical` (Baseline)**: Overparameterized cross-modal projection (~1.31M parameters).
2. **`classical_matched` (Control)**: Exactly 72 trainable parameters arranged in 4 layers of `Linear(6, 3, bias=False)` with tanh and cross-modal coupling.
3. **`quantum` (VQC)**: 6 qubits on PennyLane `default.qubit`, 4 variational layers, 72 single-qubit rotation parameters, and cross-modal CNOT entanglement linking visual qubits (`[0, 1, 2]`) and audio qubits (`[3, 4, 5]`) via `[2, 3]` and `[5, 0]`.

---

## 3. Directory Structure

```
qfusion-dvc/
├── backbones/                 # ViViT and AST backbones with temporal pooling
├── configs/                   # Configuration files for all 3 conditions
├── data/                      # ActivityNet captions, snippets, and feature caches
├── gate_evidence/             # Acceptance verification logs for Gates 1 to 15
├── models/                    # Fusion layers, deformable encoder, and DVC model
├── paper_draft/               # Auto-generated manuscript.md and references.bib
├── results/                   # Evaluation results, summaries, plots, and checkpoints
│   ├── classical/             # Seed 42, 123, 2024 checkpoints & histories
│   ├── classical_matched/     # Seed 42, 123, 2024 checkpoints & histories
│   ├── quantum/               # Seed 42, 123, 2024 checkpoints & histories
│   ├── plots/                 # All 9 generated publication-ready plots
│   ├── eval_results.json      # Complete raw evaluation metrics
│   ├── RESULTS_SUMMARY.md     # Markdown results summary with Wilcoxon analysis
│   └── qualitative_examples.md# 15-video 3-way qualitative comparisons
├── utils/                     # Metrics, evaluation, matcher, dataset loaders
├── BLOCKED_LOG.md             # Documented decisions and environmental deviations
├── GATE_STATUS.md             # Sequential gate verification status tracking
├── evaluate.py                # Dual-split evaluation & Wilcoxon testing
├── generate_paper_draft.py    # Gate 15 draft generator & numerical audit
├── make_plots.py              # Plots 1-9 generation
├── run_all_experiments.py     # 9-run automated training orchestrator
└── setup.sh                   # Environment setup script
```

---

## 4. Key Results Summary

- **Primary Split (`val_1`)**:
  - `classical` (~1.3M params): F1@0.5 = 0.333 ± 0.045, METEOR = 15.00
  - `classical_matched` (72 params): F1@0.5 = 0.361 ± 0.023, METEOR = 15.00
  - `quantum` (72 params): F1@0.5 = 0.296 ± 0.065, METEOR = 15.00
- **Paired Wilcoxon Signed-Rank Test**:
  - `classical_matched` vs. `quantum` on F1@0.5 yields p = 0.5000 (direction: `classical_matched > quantum`).
- **Conclusion (Rule 1.9 No-Spin)**: Parameterized quantum circuits train stably in end-to-end multimodal pipelines, but under parameter-matched conditions, physical quantum advantage does not manifest over compact classical representations in classical statevector simulation.
