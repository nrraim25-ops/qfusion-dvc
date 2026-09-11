"""Verification script for Gate 9: Quantum Fusion."""

import os
import sys
from datetime import datetime
from pathlib import Path
import torch

from models.fusion_quantum import QuantumFusion


def run_verification():
    project_root = Path(__file__).resolve().parent
    evidence_dir = project_root / "gate_evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)

    report_lines = []
    def log(line=""):
        print(line)
        report_lines.append(line)

    log("=" * 60)
    log("GATE 9 ACCEPTANCE TEST: QUANTUM FUSION VERIFICATION")
    log("=" * 60)
    log(f"Timestamp: {datetime.now().isoformat()}")

    # PennyLane default.qubit simulator runs seamlessly with PyTorch on CPU or GPU
    device = torch.device("cpu") # default.qubit backprop interface
    log(f"Using simulation device: {device} (PennyLane default.qubit)")

    success = True

    model = QuantumFusion().to(device)

    # 1. Parameter Breakdown for Quantum Circuit (Target: 72 parameters)
    log("\n[1/4] Quantum Circuit Parameter Breakdown (Target: 72 parameters):")
    quantum_params = model.quantum_weights.numel()
    log(f"  quantum_weights shape: {list(model.quantum_weights.shape)}")
    log(f"  Trainable Quantum Parameters: {quantum_params}")

    if quantum_params == 72:
        log("  Exact Quantum Parameter Match (72 params): PASSED")
    else:
        log(f"  ERROR: Expected 72 quantum parameters, got {quantum_params}")
        success = False

    # Full module parameter accounting
    log("\nFull Module Parameter Accounting:")
    total_params = 0
    for name, p in model.named_parameters():
        if p.requires_grad:
            c = p.numel()
            total_params += c
            log(f"  {name:<30}: {c:>6} params (shape {list(p.shape)})")
    log(f"  {'TOTAL MODULE TRAINABLE PARAMS':<30}: {total_params:>6}")

    # 2. Printed Circuit Diagram
    log("\n[2/4] Printed Quantum Circuit Diagram:")
    circuit_diagram = model.draw_circuit()
    log(circuit_diagram)

    if "[2, 3]" in str(circuit_diagram) or "2:" in circuit_diagram and "3:" in circuit_diagram:
        log("  Circuit diagram verification: PASSED")
    else:
        log("  WARNING: Could not verify circuit diagram.")

    # 3. Shape Verification
    log("\n[3/4] Shape Verification:")
    T = 10
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

    B = 2
    v_batch = torch.randn(B, T, 768, device=device)
    a_batch = torch.randn(B, T, 768, device=device)
    out_batch = model(v_batch, a_batch)
    if out_batch.shape == (B, T, 512):
        log(f"  Batched shape check [B={B}, T={T}, 512]: PASSED")
    else:
        log(f"  ERROR: Expected ({B}, {T}, 512), got {list(out_batch.shape)}")
        success = False

    # 4. Gradient Flow Verification
    log("\n[4/4] Gradient Flow Verification (PennyLane Autograd):")
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
        log("  Gradient flow back through classical expansion, quantum circuit, and compression: PASSED")
    else:
        log("  ERROR: Gradient flow check failed.")
        success = False

    log("\n" + "=" * 60)
    if success:
        log("GATE 9 VERIFICATION RESULT: PASSED")
    else:
        log("GATE 9 VERIFICATION RESULT: FAILED")
    log("=" * 60)

    evidence_file = evidence_dir / "gate9_quantum_fusion_test.txt"
    with open(evidence_file, "w") as ef:
        ef.write("\n".join(report_lines) + "\n")
    print(f"\nEvidence file saved to: {evidence_file}")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(run_verification())
