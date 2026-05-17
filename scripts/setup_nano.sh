#!/usr/bin/env bash
# One-shot Nano setup: apt deps + pip deps.  Idempotent.
#
# Assumes JetPack 4.6.1 is already flashed.  See docs/setup_nano.md for
# the manual walk-through with verification steps.

set -euo pipefail

echo "[setup] apt update + install system packages"
sudo apt update
sudo apt install -y \
    python3-pip python3-dev \
    build-essential cmake git \
    htop nano v4l-utils ffmpeg \
    python3-libnvinfer python3-libnvinfer-dev \
    python3-opencv \
    gstreamer1.0-tools gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly gstreamer1.0-libav

echo ""
echo "[setup] Verifying NVDEC plugin"
if gst-inspect-1.0 nvv4l2decoder >/dev/null 2>&1; then
    echo "[setup]   OK: nvv4l2decoder found"
else
    echo "[setup]   FAIL: nvv4l2decoder missing.  L4T multimedia API not installed." >&2
    exit 1
fi

echo ""
echo "[setup] Verifying OpenCV GStreamer support"
if python3 -c "import cv2; assert 'GStreamer' in cv2.getBuildInformation()" 2>/dev/null; then
    echo "[setup]   OK: OpenCV has GStreamer support"
else
    echo "[setup]   FAIL: OpenCV not built with GStreamer support" >&2
    exit 1
fi

echo ""
echo "[setup] Ensuring CUDA env vars (for pycuda build)"
# Append to ~/.bashrc so future interactive shells have it.
grep -q 'CPATH.*cuda' ~/.bashrc 2>/dev/null || {
    echo 'export CPATH=$CPATH:/usr/local/cuda/include'             >> ~/.bashrc
    echo 'export LIBRARY_PATH=$LIBRARY_PATH:/usr/local/cuda/lib64' >> ~/.bashrc
    echo 'export PATH=$PATH:/usr/local/cuda/bin'                   >> ~/.bashrc
}
# Also export in this shell so the pycuda build below can find cuda.h.
# (Non-interactive SSH sessions don't source ~/.bashrc, so this is the
# variable that actually matters for an unattended setup run.)
export CPATH="/usr/local/cuda/include:${CPATH:-}"
export LIBRARY_PATH="/usr/local/cuda/lib64:${LIBRARY_PATH:-}"
export PATH="/usr/local/cuda/bin:$PATH"

echo ""
echo "[setup] Upgrading pip"
python3 -m pip install --user --upgrade "pip<22"

echo ""
echo "[setup] Installing pycuda (this can take 5-10 minutes -- compiling C++)"
# --no-build-isolation:  use the system python3-numpy from apt instead of pip
#   trying to compile ancient numpy 1.12.1 from source (fails on aarch64
#   because xlocale.h was removed from modern glibc).
# --no-use-pep517:       pip 21+ on Python 3.6 picks the new PEP 517 build
#   backend, but JetPack's old setuptools doesn't have build_meta.__legacy__.
#   The old setup.py path works fine.
python3 -m pip install --user --no-build-isolation --no-use-pep517 pycuda==2021.1

echo ""
echo "[setup] Installing Python requirements"
python3 -m pip install --user -r requirements.txt

echo ""
echo "[setup] Maximising performance"
sudo nvpmodel -m 0 || true
sudo jetson_clocks || true

echo ""
echo "[setup] Done."
echo "        Next: copy yolov8n.onnx onto the Nano and run scripts/build_engine.sh"
