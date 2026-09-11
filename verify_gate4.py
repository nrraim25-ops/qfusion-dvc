"""Verification script for Gate 4: ViViT Video Backbone."""

import os
import sys
import glob
from datetime import datetime
from pathlib import Path
import torch


def run_verification():
    project_root = Path(__file__).resolve().parent
    evidence_dir = project_root / "gate_evidence"
    os.makedirs(evidence_dir, exist_ok=True)

    report_lines = []
    def log(line=""):
        print(line)
        report_lines.append(line)

    log("=" * 60)
    log("GATE 4 ACCEPTANCE TEST: ViViT VIDEO BACKBONE VERIFICATION")
    log("=" * 60)
    log(f"Timestamp: {datetime.now().isoformat()}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Using device: {device}")

    success = True

    # 1. Load real video and extract snippets
    videos = glob.glob(str(project_root / "data" / "videos" / "**" / "*.mp4"), recursive=True)
    if not videos:
        log("ERROR: No downloaded videos found.")
        return 1

    sample_video = videos[0]
    log(f"Loading real video: {os.path.basename(sample_video)}")
    from data.preprocess import VideoAudioSnippetExtractor
    extractor = VideoAudioSnippetExtractor(frames_per_snippet=32, crop_size=224)
    v_snips, _, meta = extractor.extract_snippets(sample_video)
    T_snips = min(4, v_snips.shape[0])
    test_batch = v_snips[:T_snips].to(device)
    log(f"Input video snippet batch shape: {test_batch.shape}")

    # 2. Load ViViT backbone
    log("\nLoading ViViT backbone (google/vivit-b-16x2-kinetics400)...")
    from backbones.vivit_backbone import ViViTBackbone
    try:
        model = ViViTBackbone("google/vivit-b-16x2-kinetics400", pretrained=True).to(device)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log(f"ViViT Model Loaded: {total_params:,} total parameters ({trainable_params:,} trainable)")

        # Forward pass (feature extraction)
        log("Running forward feature extraction on batch...")
        with torch.no_grad():
            feats = model(test_batch)
        log(f"Extracted feature shape: {feats.shape}")

        if feats.shape == (T_snips, 768):
            log(f"Output shape verification [T, 768]: PASSED (got {feats.shape})")
        else:
            log(f"ERROR: Expected shape ({T_snips}, 768), got {feats.shape}")
            success = False

        # Gradient flow check
        log("Checking gradient flow through ViViT...")
        torch.cuda.empty_cache()
        single_snippet = test_batch[:1]
        grad_out = model(single_snippet)
        loss = grad_out.sum()
        loss.backward()
        has_grads = any(p.grad is not None and not torch.isnan(p.grad).any() for p in model.parameters())
        if has_grads:
            log("ViViT Gradient Flow Test: PASSED")
        else:
            log("ERROR: Gradient flow check failed.")
            success = False

    except Exception as e:
        log(f"ERROR verifying ViViT backbone: {e}")
        import traceback
        traceback.print_exc()
        success = False

    log("\n" + "=" * 60)
    if success:
        log("GATE 4 VERIFICATION RESULT: PASSED")
    else:
        log("GATE 4 VERIFICATION RESULT: FAILED")
    log("=" * 60)

    evidence_file = evidence_dir / "gate4_vivit_verification.txt"
    with open(evidence_file, "w") as ef:
        ef.write("\n".join(report_lines) + "\n")
    print(f"\nEvidence file saved to: {evidence_file}")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(run_verification())
