"""Verification script for Gate 8: Parameter-Matched Classical Control Fusion."""

import os
import sys
from datetime import datetime
from pathlib import Path
import torch

from models.fusion_classical_matched import ClassicalMatchedFusion, MatchedClassicalProcessingBlock


def run_verification():
    project_root = Path(__file__).resolve().parent
    evidence_dir = project_root / "gate_evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)

    report_lines = []
    def log(line=""):
        print(line)
        report_lines.append(line)

    log("=" * 60)
    log("GATE 8 ACCEPTANCE TEST: PARAMETER-MATCHED CLASSICAL CONTROL FUSION")
    log("=" * 60)
    log(f"Timestamp: {datetime.now().isoformat()}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Using device: {device}")

    success = True

    model = ClassicalMatchedFusion().to(device)

    # 1. Parameter Breakdown for the Processing Block (Matched to Quantum Circuit 72 Params)
    log("\n[1/3] Processing Block Parameter Breakdown (Target: 72 parameters):")
    processing_params = 0
    for name, p in model.processing.named_parameters():
        if p.requires_grad:
            count = p.numel()
            processing_params += count
            log(f"  processing.{name:<25}: {count:>4} params (shape {list(p.shape)})")
    log(f"  {'TOTAL PROCESSING PARAMS':<36}: {processing_params:>4}")

    if processing_params == 72:
        log("  Exact Parameter Match (72 params == Quantum Circuit 72 params): PASSED")
    elif 68 <= processing_params <= 76:
        log(f"  Parameter Match within tolerance [68, 76] (got {processing_params}): PASSED")
    else:
        log(f"  ERROR: Parameter count {processing_params} outside required [68, 76] range.")
        success = False

    # Full module parameter accounting
    log("\nFull Module Parameter Accounting:")
    total_module_params = 0
    for name, p in model.named_parameters():
        if p.requires_grad:
            c = p.numel()
            total_module_params += c
            log(f"  {name:<35}: {c:>6} params (shape {list(p.shape)})")
    log(f"  {'TOTAL MODULE TRAINABLE PARAMS':<35}: {total_module_params:>6}")

    # 2. Shape Verification
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

    B = 4
    v_batch = torch.randn(B, T, 768, device=device)
    a_batch = torch.randn(B, T, 768, device=device)
    out_batch = model(v_batch, a_batch)
    if out_batch.shape == (B, T, 512):
        log(f"  Batched shape check [B={B}, T={T}, 512]: PASSED")
    else:
        log(f"  ERROR: Expected ({B}, {T}, 512), got {list(out_batch.shape)}")
        success = False

    # 3. Gradient Flow Verification
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
            log(f"  {name:<35}: grad norm = {grad_norm:.6f} [OK]")

    if all_grads_ok and v_in.grad is not None and a_in.grad is not None:
        log("  Gradient flow back through all layers and inputs: PASSED")
    else:
        log("  ERROR: Gradient flow check failed.")
        success = False

    log("\n" + "=" * 60)
    if success:
        log("GATE 8 VERIFICATION RESULT: PASSED")
    else:
        log("GATE 8 VERIFICATION RESULT: FAILED")
    log("=" * 60)

    evidence_file = evidence_dir / "gate8_matched_classical_test.txt"
    with open(evidence_file, "w") as ef:
        ef.write("\n".join(report_lines) + "\n")
    print(f"\nEvidence file saved to: {evidence_file}")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(run_verification())
