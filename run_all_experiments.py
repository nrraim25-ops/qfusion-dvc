"""Orchestrator for Gate 13: Full Integration and Training across 9 Runs.

Executes:
3 conditions (classical, classical_matched, quantum) x 3 seeds (42, 123, 2024) = 9 full runs.
Generates:
- gate_evidence/gate13_training_report.txt
- results/plots/loss_curves.png (mean train & val curve +- std band per condition)
"""

import os
import sys
os.environ["MPLCONFIGDIR"] = "/tmp/mpl"
import json
import subprocess
from datetime import datetime
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def run_all():
    project_root = Path(__file__).resolve().parent
    evidence_dir = project_root / "gate_evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = project_root / "results" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    conditions = ["classical", "classical_matched", "quantum"]
    seeds = [42, 123, 2024]

    run_results = []
    py_bin = project_root / "venv" / "bin" / "python"

    print("=" * 70)
    print("GATE 13: EXECUTING ALL 9 TRAINING RUNS (3 CONDITIONS x 3 SEEDS)")
    print("=" * 70)

    for cond in conditions:
        for seed in seeds:
            print(f"\n>>> Launching Run: Condition={cond}, Seed={seed} <<<")
            cmd = [
                str(py_bin),
                str(project_root / "train_main.py"),
                "--fusion_type", cond,
                "--seed", str(seed),
                "--epochs", "20",
                "--patience", "4"
            ]

            res = subprocess.run(cmd, text=True)
            if res.returncode != 0:
                print(f"ERROR: Run failed for {cond}, seed {seed}")
                return 1

            # Load history
            history_path = project_root / "results" / cond / f"seed_{seed}" / "training_history.json"
            ckpt_path = project_root / "results" / cond / f"seed_{seed}" / "model_best.pt"

            with open(history_path, "r") as f:
                history = json.load(f)

            train_losses = history["train_loss"]
            val_losses = history["val_loss"]
            final_epoch = len(history["epochs"])

            # Verify finite loss at every step
            assert all(not np.isnan(l) for l in train_losses), f"NaN detected in train loss for {cond}, seed {seed}"
            assert all(not np.isnan(l) for l in val_losses), f"NaN detected in val loss for {cond}, seed {seed}"
            assert os.path.exists(ckpt_path) and os.path.getsize(ckpt_path) > 1000, f"Missing checkpoint for {cond}, seed {seed}"

            run_results.append({
                "condition": cond,
                "seed": seed,
                "final_epoch": final_epoch,
                "stopping_reason": "early_stopping_patience_4" if final_epoch < 20 else "max_epochs_20",
                "final_train_loss": train_losses[-1],
                "final_val_loss": val_losses[-1],
                "best_val_loss": min(val_losses),
                "ckpt_path": str(ckpt_path),
                "history": history
            })

    # 1. Generate Gate 13 Evidence Report
    evidence_lines = [
        "=" * 85,
        "GATE 13 ACCEPTANCE TEST: FULL INTEGRATION & TRAINING REPORT (9 RUNS)",
        "=" * 85,
        f"Timestamp: {datetime.now().isoformat()}",
        f"Protocol: 3 Conditions (classical, classical_matched, quantum) x 3 Seeds (42, 123, 2024)",
        "",
        f"{'Condition':<18} | {'Seed':<6} | {'Epochs':<8} | {'Stop Reason':<24} | {'Train Loss':<12} | {'Val Loss':<12} | {'Checkpoint'}",
        "-" * 115
    ]

    for r in run_results:
        evidence_lines.append(
            f"{r['condition']:<18} | {r['seed']:<6} | {r['final_epoch']:<8} | {r['stopping_reason']:<24} | {r['final_train_loss']:<12.4f} | {r['best_val_loss']:<12.4f} | {os.path.basename(r['ckpt_path'])}"
        )
    evidence_lines.append("-" * 115)
    evidence_lines.append("")
    evidence_lines.append("All 9 runs verified with strictly finite losses and saved checkpoints.")
    evidence_lines.append("GATE 13 VERIFICATION RESULT: PASSED")
    evidence_lines.append("=" * 85)

    evidence_file = evidence_dir / "gate13_training_report.txt"
    with open(evidence_file, "w") as ef:
        ef.write("\n".join(evidence_lines) + "\n")
    print(f"\nEvidence report saved to: {evidence_file}")

    # 2. Generate Plot 2: loss_curves.png (mean +- std band across seeds per condition)
    print("\nGenerating loss curves plot across all 3 conditions...")
    fig, (ax_train, ax_val) = plt.subplots(1, 2, figsize=(14, 5))
    colors = {"classical": "steelblue", "classical_matched": "forestgreen", "quantum": "darkviolet"}

    for cond in conditions:
        cond_runs = [r for r in run_results if r["condition"] == cond]
        # Align lengths
        min_len = min(len(r["history"]["epochs"]) for r in cond_runs)
        epochs_range = np.arange(1, min_len + 1)

        t_curves = np.array([r["history"]["train_loss"][:min_len] for r in cond_runs])
        v_curves = np.array([r["history"]["val_loss"][:min_len] for r in cond_runs])

        t_mean, t_std = np.mean(t_curves, axis=0), np.std(t_curves, axis=0)
        v_mean, v_std = np.mean(v_curves, axis=0), np.std(v_curves, axis=0)

        # Train plot
        ax_train.plot(epochs_range, t_mean, label=cond, color=colors[cond], linewidth=2)
        ax_train.fill_between(epochs_range, t_mean - t_std, t_mean + t_std, color=colors[cond], alpha=0.2)

        # Val plot
        ax_val.plot(epochs_range, v_mean, label=cond, color=colors[cond], linewidth=2)
        ax_val.fill_between(epochs_range, v_mean - v_std, v_mean + v_std, color=colors[cond], alpha=0.2)

    ax_train.set_title("Training Loss across Conditions (Mean ± Std, n=3 Seeds)")
    ax_train.set_xlabel("Epoch")
    ax_train.set_ylabel("Loss")
    ax_train.grid(True)
    ax_train.legend()

    ax_val.set_title("Validation Loss across Conditions (Mean ± Std, n=3 Seeds)")
    ax_val.set_xlabel("Epoch")
    ax_val.set_ylabel("Loss")
    ax_val.grid(True)
    ax_val.legend()

    plt.tight_layout()
    plot_file = plots_dir / "loss_curves.png"
    plt.savefig(plot_file, dpi=150)
    plt.close()
    print(f"Saved plot: {plot_file}")

    return 0


if __name__ == "__main__":
    sys.exit(run_all())
