"""Verification script for Gate 10: Deformable + Sparse-Attention Encoder."""

import os
import sys
from datetime import datetime
from pathlib import Path
import torch
import matplotlib.pyplot as plt

from models.encoder_deformable import DeformableSparseEncoder


def run_verification():
    project_root = Path(__file__).resolve().parent
    evidence_dir = project_root / "gate_evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = project_root / "results" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    report_lines = []
    def log(line=""):
        print(line)
        report_lines.append(line)

    log("=" * 60)
    log("GATE 10 ACCEPTANCE TEST: DEFORMABLE + SPARSE ENCODER VERIFICATION")
    log("=" * 60)
    log(f"Timestamp: {datetime.now().isoformat()}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Using device: {device}")

    success = True

    model = DeformableSparseEncoder(embed_dim=512, num_layers=4, num_heads=4, num_points=4, prune_ratio=0.5).to(device)

    # 1. Structural checks
    log("\n[1/4] Structural Specification Check:")
    log(f"  Embedding Dimension: {model.embed_dim}")
    log(f"  Attention Heads: {model.layer1.self_attn.num_heads}")
    log(f"  Sampling Points per Head: {model.layer1.self_attn.num_points}")
    log(f"  Pruning Ratio after Layer 2: {model.prune_ratio * 100:.1f}%")

    if model.layer1.self_attn.num_points == 4:
        log("  Sampling points check (K=4 points): PASSED")
    else:
        log(f"  ERROR: Expected 4 sampling points, got {model.layer1.self_attn.num_points}")
        success = False

    if model.prune_ratio == 0.5:
        log("  Sparsification keep ratio check (50% pruning after layer 2): PASSED")
    else:
        log(f"  ERROR: Expected 0.5 pruning ratio, got {model.prune_ratio}")
        success = False

    # 2. Shape Verification
    log("\n[2/4] Shape Verification:")
    T = 20
    x_in = torch.randn(T, 512, device=device, requires_grad=True)
    out, scores, keep_idx = model(x_in)

    log(f"  Input sequence shape: [T={T}, 512]")
    log(f"  Output sequence shape: {list(out.shape)}")
    log(f"  Saliency scores shape: {list(scores.shape)}")
    log(f"  Retained tokens count: {len(keep_idx)} (expected: {T // 2})")

    if out.shape == (T, 512) and len(keep_idx) == (T // 2):
        log("  Unbatched shape check [T, 512] -> [T, 512] with 50% pruning: PASSED")
    else:
        log(f"  ERROR: Expected shape ({T}, 512) and {T // 2} kept tokens, got {list(out.shape)} and {len(keep_idx)}")
        success = False

    # Batched check
    B = 4
    x_batch = torch.randn(B, T, 512, device=device)
    out_b, scores_b, keep_idx_b = model(x_batch)
    if out_b.shape == (B, T, 512) and keep_idx_b.shape == (B, T // 2):
        log(f"  Batched shape check [B={B}, T={T}, 512] -> [B={B}, T={T}, 512]: PASSED")
    else:
        log(f"  ERROR: Batched shape check failed: {list(out_b.shape)}")
        success = False

    # 3. Gradient Flow Verification
    log("\n[3/4] Gradient Flow Verification:")
    loss = out.sum() + scores.sum()
    loss.backward()

    all_grads_ok = True
    for name, p in model.named_parameters():
        if p.grad is None or torch.isnan(p.grad).any():
            log(f"  ERROR: Missing or invalid grad for {name}")
            all_grads_ok = False

    if all_grads_ok and x_in.grad is not None and not torch.isnan(x_in.grad).any():
        log("  Gradient flow back through all layers and input sequence: PASSED")
    else:
        log("  ERROR: Gradient check failed.")
        success = False

    # 4. Generate Plot 8: Sparsification Keep Ratio Plot
    log("\n[4/4] Generating Sparsification Keep Ratio Plot...")
    test_lengths = [10, 20, 30, 40, 50, 60, 80, 100]
    kept_counts = []
    ratios = []
    with torch.no_grad():
        for t_len in test_lengths:
            dummy_x = torch.randn(1, t_len, 512, device=device)
            _, _, k_idx = model(dummy_x)
            num_kept = k_idx.size(-1)
            kept_counts.append(num_kept)
            ratios.append(num_kept / t_len)

    plt.figure(figsize=(8, 5))
    plt.plot(test_lengths, ratios, marker="o", color="crimson", linewidth=2, label="Actual Keep Ratio")
    plt.axhline(0.5, linestyle="--", color="navy", label="Target 50% Keep Ratio")
    plt.xlabel("Input Sequence Length (T snippets)")
    plt.ylabel("Keep Ratio (Kept / T)")
    plt.title("Deformable Encoder Sparsification Keep Ratio across Sequence Lengths")
    plt.ylim(0.4, 0.6)
    plt.grid(True)
    plt.legend()
    plot_file = plots_dir / "sparsification_keep_ratio.png"
    plt.tight_layout()
    plt.savefig(plot_file, dpi=150)
    plt.close()
    log(f"  Saved plot: {plot_file}")

    log("\n" + "=" * 60)
    if success:
        log("GATE 10 VERIFICATION RESULT: PASSED")
    else:
        log("GATE 10 VERIFICATION RESULT: FAILED")
    log("=" * 60)

    evidence_file = evidence_dir / "gate10_encoder_test.txt"
    with open(evidence_file, "w") as ef:
        ef.write("\n".join(report_lines) + "\n")
    print(f"\nEvidence file saved to: {evidence_file}")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(run_verification())
