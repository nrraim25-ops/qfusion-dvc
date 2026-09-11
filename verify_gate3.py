"""Verification script for Gate 3: Preprocessing Pipeline."""

import os
import sys
import glob
from datetime import datetime
from pathlib import Path


def run_verification():
    project_root = Path(__file__).resolve().parent
    evidence_dir = project_root / "gate_evidence"
    os.makedirs(evidence_dir, exist_ok=True)

    report_lines = []
    def log(line=""):
        print(line)
        report_lines.append(line)

    log("=" * 60)
    log("GATE 3 ACCEPTANCE TEST: PREPROCESSING PIPELINE VERIFICATION")
    log("=" * 60)
    log(f"Timestamp: {datetime.now().isoformat()}")

    success = True

    # 1. Check for real downloaded videos
    videos = glob.glob(str(project_root / "data" / "videos" / "**" / "*.mp4"), recursive=True)
    if not videos:
        log("ERROR: No downloaded videos found in data/videos/ to verify preprocessing.")
        return 1

    sample_video = videos[0]
    log(f"Found {len(videos)} downloaded videos.")
    log(f"Testing snippet extraction on real video: {os.path.basename(sample_video)}")

    # 2. Run extraction
    from data.preprocess import VideoAudioSnippetExtractor
    extractor = VideoAudioSnippetExtractor(
        snippet_duration=2.0,
        target_fps=25,
        frames_per_snippet=32,
        crop_size=224,
        audio_sr=16000
    )

    try:
        v_snips, a_snips, meta = extractor.extract_snippets(sample_video)
        log(f"  Extraction successful for duration: {meta['duration']:.2f}s")
        log(f"  Number of 2.0s snippets (T): {meta['num_snippets']}")
        log(f"  Video snippets tensor shape: {v_snips.shape}")
        log(f"  Audio snippets tensor shape: {a_snips.shape}")
        log(f"  Audio sample rate: {meta['audio_sr']} Hz (mono)")

        # Validate video snippet shape: [T, 32, 3, 224, 224]
        T_snips = meta['num_snippets']
        if v_snips.shape != (T_snips, 32, 3, 224, 224):
            log(f"  ERROR: Unexpected video snippet shape: {v_snips.shape}, expected ({T_snips}, 32, 3, 224, 224)")
            success = False
        else:
            log("  Video snippet dimensions check [T, 32, 3, 224, 224]: PASSED")

        # Validate audio snippet shape: [T, 32000]
        if a_snips.shape != (T_snips, 32000):
            log(f"  ERROR: Unexpected audio snippet shape: {a_snips.shape}, expected ({T_snips}, 32000)")
            success = False
        else:
            log("  Audio snippet dimensions check [T, 32000] (16kHz mono): PASSED")

        # Check values
        if v_snips.isnan().any() or a_snips.isnan().any():
            log("  ERROR: NaN detected in extracted snippet tensors.")
            success = False
        else:
            log("  Finite values check (no NaNs): PASSED")

    except Exception as e:
        log(f"  ERROR during snippet extraction: {e}")
        import traceback
        traceback.print_exc()
        success = False

    log("\n" + "=" * 60)
    if success:
        log("GATE 3 VERIFICATION RESULT: PASSED")
    else:
        log("GATE 3 VERIFICATION RESULT: FAILED")
    log("=" * 60)

    evidence_file = evidence_dir / "gate3_preprocessing_verification.txt"
    with open(evidence_file, "w") as ef:
        ef.write("\n".join(report_lines) + "\n")
    print(f"\nEvidence file saved to: {evidence_file}")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(run_verification())
