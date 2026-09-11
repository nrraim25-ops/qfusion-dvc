"""Verification script for Gate 1: Repository and Environment Setup."""

import sys
import os
import platform
from datetime import datetime


def run_verification():
    report_lines = []
    def log(line=""):
        print(line)
        report_lines.append(line)

    log("=" * 60)
    log("GATE 1 ACCEPTANCE TEST: ENVIRONMENT & REPO VERIFICATION")
    log("=" * 60)
    log(f"Timestamp: {datetime.now().isoformat()}")
    log(f"Platform: {platform.platform()}")
    log(f"Python Version: {sys.version}")

    success = True

    # 1. Check PyTorch and CUDA
    log("\n[1/5] Checking PyTorch and CUDA availability...")
    try:
        import torch
        log(f"  PyTorch Version: {torch.__version__}")
        cuda_avail = torch.cuda.is_available()
        log(f"  CUDA Available: {cuda_avail}")
        if cuda_avail:
            device_name = torch.cuda.get_device_name(0)
            total_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            free_mem = torch.cuda.mem_get_info()[0] / (1024 ** 3)
            log(f"  Device Name: {device_name}")
            log(f"  Total VRAM: {total_mem:.2f} GB")
            log(f"  Free VRAM: {free_mem:.2f} GB")
            # Quick tensor allocation on GPU
            x = torch.randn(10, 10, device="cuda")
            y = x @ x
            log(f"  CUDA Tensor Compute Test: PASSED (allocated and computed {y.shape})")
        else:
            log("  WARNING: CUDA is not available. GPU is required per spec.")
            success = False
    except Exception as e:
        log(f"  ERROR checking PyTorch/CUDA: {e}")
        success = False

    # 2. Check PennyLane & Quantum Device
    log("\n[2/5] Checking PennyLane quantum simulation...")
    try:
        import pennylane as qml
        log(f"  PennyLane Version: {qml.__version__}")
        dev = qml.device("default.qubit", wires=6)
        log(f"  PennyLane Device Initialized: {dev.name} (wires={len(dev.wires)})")

        @qml.qnode(dev, interface="torch")
        def circuit(inputs):
            for i in range(6):
                qml.RX(inputs[i], wires=i)
            qml.CNOT(wires=[2, 3])
            qml.CNOT(wires=[5, 0])
            return [qml.expval(qml.PauliZ(i)) for i in range(6)]

        test_inputs = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], requires_grad=True)
        res = circuit(test_inputs)
        loss = torch.sum(torch.stack(res))
        loss.backward()
        log(f"  Quantum Circuit Output Shape: {len(res)}")
        log(f"  Quantum Circuit Gradients: {test_inputs.grad}")
        if test_inputs.grad is not None and not torch.isnan(test_inputs.grad).any():
            log("  PennyLane Torch QNode Autograd Test: PASSED")
        else:
            log("  ERROR: PennyLane Torch QNode Autograd failed.")
            success = False
    except Exception as e:
        log(f"  ERROR checking PennyLane: {e}")
        success = False

    # 3. Check Transformers & Hugging Face
    log("\n[3/5] Checking Transformers and vision/audio backbones...")
    try:
        import transformers
        import timm
        log(f"  Transformers Version: {transformers.__version__}")
        log(f"  timm Version: {timm.__version__}")
        log("  Transformers & timm import: PASSED")
    except Exception as e:
        log(f"  ERROR checking transformers/timm: {e}")
        success = False

    # 4. Check Video/Audio Processing Tools
    log("\n[4/5] Checking media processing libraries & tools...")
    try:
        import yt_dlp
        log(f"  yt-dlp Version: {yt_dlp.version.__version__}")
        import soundfile
        log(f"  soundfile Version: {soundfile.__version__}")
        import scipy
        log(f"  scipy Version: {scipy.__version__}")
        import pandas
        log(f"  pandas Version: {pandas.__version__}")
    except Exception as e:
        log(f"  ERROR checking media libraries: {e}")
        success = False

    # Check ffmpeg binary
    import shutil
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        log(f"  ffmpeg found at: {ffmpeg_path}")
    else:
        # Check project bin
        local_ffmpeg = os.path.abspath(os.path.join(os.path.dirname(__file__), "bin", "ffmpeg"))
        if os.path.exists(local_ffmpeg):
            log(f"  ffmpeg found at: {local_ffmpeg}")
        else:
            log("  WARNING: ffmpeg not found in PATH or project bin.")
            success = False

    # 5. Check Directory Structure and Tracking Files
    log("\n[5/5] Checking repository directory structure...")
    required_dirs = ["gate_evidence", "configs", "data", "backbones", "models", "utils", "results", "paper_draft"]
    all_dirs_ok = True
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for d in required_dirs:
        dir_path = os.path.join(base_dir, d)
        exists = os.path.isdir(dir_path)
        log(f"  Directory '{d}': {'OK' if exists else 'MISSING'}")
        if not exists:
            all_dirs_ok = False
    if not all_dirs_ok:
        success = False

    required_files = ["GATE_STATUS.md", "BLOCKED_LOG.md", "requirements.txt", "setup.sh"]
    for f in required_files:
        f_path = os.path.join(base_dir, f)
        exists = os.path.isfile(f_path)
        log(f"  File '{f}': {'OK' if exists else 'MISSING'}")
        if not exists:
            success = False

    log("\n" + "=" * 60)
    if success:
        log("GATE 1 VERIFICATION RESULT: PASSED")
    else:
        log("GATE 1 VERIFICATION RESULT: FAILED")
    log("=" * 60)

    # Write evidence file
    evidence_dir = os.path.join(base_dir, "gate_evidence")
    os.makedirs(evidence_dir, exist_ok=True)
    evidence_file = os.path.join(evidence_dir, "gate1_setup_verification.txt")
    with open(evidence_file, "w") as ef:
        ef.write("\n".join(report_lines) + "\n")
    print(f"\nEvidence written to: {evidence_file}")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(run_verification())
