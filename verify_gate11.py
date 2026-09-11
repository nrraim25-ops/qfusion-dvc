"""Verification script for Gate 11: Event-Query Localization Head."""

import os
import sys
from datetime import datetime
from pathlib import Path
import torch

from models.localization_head import EventQueryLocalizationHead, HungarianMatcher1D


def run_verification():
    project_root = Path(__file__).resolve().parent
    evidence_dir = project_root / "gate_evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)

    report_lines = []
    def log(line=""):
        print(line)
        report_lines.append(line)

    log("=" * 60)
    log("GATE 11 ACCEPTANCE TEST: LOCALIZATION HEAD VERIFICATION")
    log("=" * 60)
    log(f"Timestamp: {datetime.now().isoformat()}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Using device: {device}")

    success = True

    model = EventQueryLocalizationHead(
        embed_dim=512,
        num_queries=30,
        num_decoder_layers=3,
        num_heads=8
    ).to(device)

    # 1. Structural Check
    log("\n[1/4] Structural Specification Check:")
    log(f"  Number of Event Queries: {model.num_queries} (expected: 30)")
    log(f"  Decoder Layers (Deep Supervision): {model.num_decoder_layers}")
    log(f"  Hungarian Cost Weights: L1={model.matcher.cost_l1}, tIoU={model.matcher.cost_tiou}, Class={model.matcher.cost_class}")

    if model.num_queries == 30:
        log("  Event queries count check (30 queries): PASSED")
    else:
        log(f"  ERROR: Expected 30 queries, got {model.num_queries}")
        success = False

    if model.matcher.cost_l1 == 5.0 and model.matcher.cost_tiou == 2.0 and model.matcher.cost_class == 1.0:
        log("  Hungarian Cost Weights (exact 5/2/1): PASSED")
    else:
        log("  ERROR: Hungarian cost weights do not match 5/2/1.")
        success = False

    # 2. Shape Verification
    log("\n[2/4] Shape Verification:")
    B, T = 2, 25
    enc_feats = torch.randn(B, T, 512, device=device, requires_grad=True)
    outputs = model(enc_feats)

    p_logits = outputs["pred_logits"]
    p_segs = outputs["pred_segments"]
    q_feats = outputs["query_features"]
    aux = outputs["aux_outputs"]

    log(f"  pred_logits shape:     {list(p_logits.shape)} (expected [{B}, 30, 1])")
    log(f"  pred_segments shape:   {list(p_segs.shape)} (expected [{B}, 30, 2])")
    log(f"  query_features shape:  {list(q_feats.shape)} (expected [{B}, 30, 512])")
    log(f"  Deep supervision layers: {len(aux)} auxiliary layers")

    if p_logits.shape == (B, 30, 1) and p_segs.shape == (B, 30, 2) and q_feats.shape == (B, 30, 512):
        log("  Output shapes verification: PASSED")
    else:
        log("  ERROR: Unexpected output shapes.")
        success = False

    if len(aux) == (model.num_decoder_layers - 1):
        log(f"  Deep supervision auxiliary outputs check ({len(aux)} layers): PASSED")
    else:
        log(f"  ERROR: Expected {model.num_decoder_layers - 1} aux layers, got {len(aux)}")
        success = False

    # 3. Hungarian Matching and Loss Computation
    log("\n[3/4] Hungarian Bipartite Matching and Loss Test:")
    # Synthetic ground-truth targets
    targets = [
        {
            "segments": torch.tensor([[0.1, 0.4], [0.5, 0.9]], device=device), # 2 events in video 1
            "labels": torch.tensor([1, 1], device=device)
        },
        {
            "segments": torch.tensor([[0.2, 0.6]], device=device),              # 1 event in video 2
            "labels": torch.tensor([1], device=device)
        }
    ]

    losses = model.compute_loss(outputs, targets)
    for k, v in losses.items():
        log(f"  {k:<20}: {v.item():.6f}")

    if not torch.isnan(losses["total_loss"]) and losses["total_loss"].item() > 0:
        log("  Hungarian Matching & Deep Supervision Loss: PASSED")
    else:
        log("  ERROR: Invalid loss value.")
        success = False

    # 4. Gradient Flow Verification
    log("\n[4/4] Gradient Flow Verification:")
    losses["total_loss"].backward()

    all_grads_ok = True
    for name, p in model.named_parameters():
        if p.grad is None or torch.isnan(p.grad).any():
            log(f"  ERROR: Missing or invalid grad for {name}")
            all_grads_ok = False

    if all_grads_ok and enc_feats.grad is not None and not torch.isnan(enc_feats.grad).any():
        log("  Gradient flow through localization head and encoder features: PASSED")
    else:
        log("  ERROR: Gradient flow check failed.")
        success = False

    log("\n" + "=" * 60)
    if success:
        log("GATE 11 VERIFICATION RESULT: PASSED")
    else:
        log("GATE 11 VERIFICATION RESULT: FAILED")
    log("=" * 60)

    evidence_file = evidence_dir / "gate11_localization_test.txt"
    with open(evidence_file, "w") as ef:
        ef.write("\n".join(report_lines) + "\n")
    print(f"\nEvidence file saved to: {evidence_file}")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(run_verification())
