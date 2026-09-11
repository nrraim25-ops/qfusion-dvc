#!/usr/bin/env bash
set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo "=== Setting up Quantum-Fusion DVC Environment ==="

# 1. Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment at ${PROJECT_DIR}/venv..."
    python3 -m venv venv
fi

# 2. Activate venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel

# 3. Install PyTorch with CUDA 12 support
echo "Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. Install remaining dependencies
echo "Installing project dependencies from requirements.txt..."
pip install -r requirements.txt

# 5. Check / Install static ffmpeg if not on system PATH
if ! command -v ffmpeg &> /dev/null; then
    echo "ffmpeg not found in PATH. Checking static ffmpeg binary..."
    mkdir -p "${PROJECT_DIR}/bin"
    if [ ! -f "${PROJECT_DIR}/bin/ffmpeg" ]; then
        echo "Downloading static ffmpeg binary to ${PROJECT_DIR}/bin/..."
        curl -sSL https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz -o /tmp/ffmpeg-static.tar.xz
        tar -xJf /tmp/ffmpeg-static.tar.xz -C /tmp/
        mv /tmp/ffmpeg-*-amd64-static/ffmpeg "${PROJECT_DIR}/bin/"
        mv /tmp/ffmpeg-*-amd64-static/ffprobe "${PROJECT_DIR}/bin/"
        chmod +x "${PROJECT_DIR}/bin/ffmpeg" "${PROJECT_DIR}/bin/ffprobe"
        rm -rf /tmp/ffmpeg-* /tmp/ffmpeg-static.tar.xz
    fi
    export PATH="${PROJECT_DIR}/bin:$PATH"
fi

# 6. Download NLTK data for evaluation
python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('wordnet', quiet=True)"

echo "=== Setup Complete! ==="
