"""Verification script for Gate 5: AST Audio Backbone."""

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
    log("GATE 5 ACCEPTANCE TEST: AST AUDIO BACKBONE VERIFICATION")
    log("=" * 60)
    log(f"Timestamp: {datetime.now().isoformat()}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Using device: {device}")

    success = True

    # 1. Load real video and extract audio snippets
    videos = glob.glob(str(project_root / "data" / "videos" / "**" / "*.mp4"), recursive=True)
    if not videos:
        log("ERROR: No downloaded videos found.")
        return 1

    sample_video = videos[0]
    log(f"Loading real video audio: {os.path.basename(sample_video)}")
    from data.preprocess import VideoAudioSnippetExtractor
    extractor = VideoAudioSnippetExtractor(audio_sr=16000)
    _, a_snips, meta = extractor.extract_snippets(sample_video)
    T_snips = min(4, a_snips.shape[0])
    test_batch = a_snips[:T_snips].to(device)
    log(f"Input audio snippet batch shape: {test_batch.shape} (16kHz mono)")

    # 2. Load AST backbone
    log("\nLoading AST backbone (MIT/ast-finetuned-audioset-10-10-0.4593)...")
    from backbones.ast_backbone import ASTBackbone
    try:
        model = ASTBackbone("MIT/ast-finetuned-audioset-10-10-0.4593", pretrained=True).to(device)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log(f"AST Model Loaded: {total_params:,} total parameters ({trainable_params:,} trainable)")

        # Forward pass (feature extraction)
        log("Running forward feature extraction on audio batch...")
        with torch.no_grad():
            feats = model(test_batch)
        log(f"Extracted audio feature shape: {feats.shape}")

        if feats.shape == (T_snips, 768):
            log(f"Output shape verification [T, 768]: PASSED (got {feats.shape})")
        else:
            log(f"ERROR: Expected shape ({T_snips}, 768), got {feats.shape}")
            success = False

        # Gradient flow check
        log("Checking gradient flow through AST...")
        torch.cuda.empty_cache()
        single_snippet = test_batch[:1]
        # Directly pass tensor through feature extractor and model for autograd check
        inputs = model.feature_extractor(single_snippet.cpu().numpy(), sampling_rate=16000, return_tensors="pt")
        input_vals = inputs.input_values.to(device)
        out = model.ast(input_values=input_vals)
        loss = out.last_hidden_state.sum()
        loss.backward()

        has_grads = any(p.grad is not None and not torch.isnan(p.grad).any() for p in model.parameters())
        if has_grads:
            log("AST Gradient Flow Test: PASSED")
        else:
            log("ERROR: Gradient flow check failed.")
            success = False

    except Exception as e:
        log(f"ERROR verifying AST backbone: {e}")
        import traceback
        traceback.print_exc()
        success = False

    log("\n" + "=" * 60)
    if success:
        log("GATE 5 VERIFICATION RESULT: PASSED")
    else:
        log("GATE 5 VERIFICATION RESULT: FAILED")
    log("=" * 60)

    evidence_file = evidence_dir / "gate5_ast_verification.txt"
    with open(evidence_file, "w") as ef:
        ef.write("\n".join(report_lines) + "\n")
    print(f"\nEvidence file saved to: {evidence_file}")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(run_verification())
