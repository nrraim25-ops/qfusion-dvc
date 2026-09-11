"""Verification script for Gate 7: Classical Fusion (Baseline)."""

import os
import sys
from datetime import datetime
from pathlib import Path
import torch

from models.fusion_classical import ClassicalFusion


def run_verification():
    project_root = Path(__file__).resolve().parent
    evidence_dir = project_root / "gate_evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)

    report_lines = []
    def log(line=""):
        print(line)
        report_lines.append(line)

    log("=" * 60)
    log("GATE 7 ACCEPTANCE TEST: CLASSICAL FUSION (BASELINE) VERIFICATION")
    log("=" * 60)
    log(f"Timestamp: {datetime.now().isoformat()}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Using device: {device}")

    success = True

    model = ClassicalFusion().to(device)

    # 1. Parameter Breakdown
    log("\n[1/3] Trainable Parameter Breakdown:")
    total_params = 0
    for name, p in model.named_parameters():
        if p.requires_grad:
            count = p.numel()
            total_params += count
            log(f"  {name:<30}: {count:>10,} params (shape {list(p.shape)})")
    log(f"  {'TOTAL TRAINABLE PARAMS':<30}: {total_params:>10,}")

    if total_params < 1_000_000 or total_params > 1_500_000:
        log(f"  WARNING: Total parameter count {total_params} differs from expected ~1.3M.")
        success = False
    else:
        log("  Parameter count scale check (~1.3M baseline): PASSED")

    # 2. Shape verification
    log("\n[2/3] Shape Verification:")
    T = 15
    v_in = torch.randn(T, 768, device=device, requires_grad=True)
    a_in = torch.randn(T, 768, device=device, requires_grad=True)

    out = model(v_in, a_in)
    log(f"  Input:  [T={T}, 768] + [T={T}, 768]")
    log(f"  Output: {list(out.shape)}")

    if out.shape == (T, 512):
        log("  Shape check [T, 768] + [T, 768] -> [T, 512]: PASSED")
    else:
        log(f"  ERROR: Expected ({T}, 512), got {list(out.shape)}")
        success = False

    # Also check batched shape [B, T, 512]
    B = 4
    v_batch = torch.randn(B, T, 768, device=device)
    a_batch = torch.randn(B, T, 768, device=device)
    out_batch = model(v_batch, a_batch)
    if out_batch.shape == (B, T, 512):
        log(f"  Batched shape check [B={B}, T={T}, 512]: PASSED")
    else:
        log(f"  ERROR: Expected ({B}, {T}, 512), got {list(out_batch.shape)}")
        success = False

    # 3. Gradient flow check
    log("\n[3/3] Gradient Flow Verification:")
    loss = out.sum()
    loss.backward()

    all_grads_ok = True
    for name, p in model.named_parameters():
        if p.grad is None or torch.isnan(p.grad).any():
            log(f"  ERROR: Invalid or missing gradient for {name}")
            all_grads_ok = False
        else:
            grad_norm = p.grad.norm().item()
            log(f"  {name:<30}: grad norm = {grad_norm:.6f} [OK]")

    if all_grads_ok and v_in.grad is not None and a_in.grad is not None:
        log("  Gradient flow back to inputs and parameters: PASSED")
    else:
        log("  ERROR: Gradient flow check failed.")
        success = False

    log("\n" + "=" * 60)
    if success:
        log("GATE 7 VERIFICATION RESULT: PASSED")
    else:
        log("GATE 7 VERIFICATION RESULT: FAILED")
    log("=" * 60)

    evidence_file = evidence_dir / "gate7_classical_fusion_test.txt"
    with open(evidence_file, "w") as ef:
        ef.write("\n".join(report_lines) + "\n")
    print(f"\nEvidence file saved to: {evidence_file}")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(run_verification())
