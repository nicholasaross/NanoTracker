# NanoTracker -- Jetson Nano (original, 4GB) Setup

Target platform: Jetson Nano (original) with JetPack 4.6.1
(L4T 32.7.1, Ubuntu 18.04, Python 3.6.9, CUDA 10.2, TensorRT 8.2.1).

This document is the install order for everything NanoTracker needs.
Run these on the **Nano** itself unless flagged `(DEV BOX)`.

## 0. Sanity check JetPack components

```bash
# Confirm L4T version (should print 32.7.1 or similar):
head -1 /etc/nv_tegra_release

# Confirm Python:
python3 --version          # expect Python 3.6.9

# Confirm CUDA:
nvcc --version             # expect 10.2

# Confirm TensorRT:
dpkg -l | grep -i tensorrt # expect 8.2.x
```

If any of these are missing, re-flash JetPack 4.6.1 before going further.
The original Nano has no JetPack 5.x / 6.x support -- 4.6.1 is the end of
the line.

## 1. System packages

Most of these will already be installed if you ran the apt command from
the previous session.  Re-running is harmless.

```bash
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
```

Verify NVDEC plugin is available (this is the lowest-CPU decode path):

```bash
gst-inspect-1.0 nvv4l2decoder
```

If this prints plugin info -- good.  If it prints "No such element" then
the L4T multimedia API is missing; re-flash JetPack.

Verify OpenCV has GStreamer support:

```bash
python3 -c "import cv2; print(cv2.getBuildInformation())" | grep -i gstreamer
```

Should print `GStreamer:  YES`.

## 2. Python packages

```bash
# Upgrade pip first (the system pip on JetPack 4.6 is ancient).
python3 -m pip install --user --upgrade "pip<22"

# pycuda is the Python wrapper for CUDA used by trt_engine.py.
# Version 2021.1 is known-good on JetPack 4.6.1.
python3 -m pip install --user pycuda==2021.1

# Project deps.
python3 -m pip install --user -r requirements.txt
```

If `pycuda` build fails complaining about missing `cuda.h`, fix the
include path:

```bash
echo 'export CPATH=$CPATH:/usr/local/cuda/include'              >> ~/.bashrc
echo 'export LIBRARY_PATH=$LIBRARY_PATH:/usr/local/cuda/lib64'  >> ~/.bashrc
echo 'export PATH=$PATH:/usr/local/cuda/bin'                    >> ~/.bashrc
source ~/.bashrc
```

Then retry the `pip install pycuda==2021.1` step.

## 3. Get the YOLOv8n ONNX file

The ONNX file is built on a normal dev box (any Python 3.9+, x86_64 or ARM
Mac), because ultralytics + modern PyTorch won't install on JetPack 4.6.

**(DEV BOX)** -- one-time:

```bash
pip install "ultralytics>=8.0,<9.0" onnx onnxsim
python scripts/export_onnx.py             # produces yolov8n.onnx
```

Copy the resulting `yolov8n.onnx` to the Nano:

```bash
scp yolov8n.onnx <nano-user>@<nano-ip>:~/NanoTracker/
```

## 4. Build the TRT engine on the Nano

This **must** run on the Nano -- TRT engines are not portable across GPU
architectures (an engine built on a 3080 won't load on Maxwell).

```bash
cd ~/NanoTracker
./scripts/build_engine.sh yolov8n.onnx
```

Expected output: `yolov8n_fp16.engine` (~7-9 MB).  First build is slow
(several minutes) because TRT searches kernels; subsequent loads are fast.

If you see out-of-memory errors during the build, drop the workspace size:

```bash
WORKSPACE_MB=512 ./scripts/build_engine.sh yolov8n.onnx
```

## 5. Configure the camera

```bash
cp camera_config.example.json camera_config.json
nano camera_config.json
```

Set `camera.ip`, `camera.password` (or use `$REOLINK_PASSWORD`).  The
default `nano.preferred_stream` is `"sub"` which gives the smaller H.264
stream -- right for a Nano because the 4096x2784 main stream would starve
the 4GB shared memory.  Switch to `"main"` only if you want to stress-test.

## 6. Maximise performance

Before running perf tests, lock the Nano into max-clock mode:

```bash
sudo nvpmodel -m 0          # 10W mode (highest power budget)
sudo jetson_clocks          # lock GPU/CPU to max frequencies
```

To revert later: `sudo jetson_clocks --restore`.

## 7. Run

```bash
cd ~/NanoTracker

# 60-second perf measurement, JSON+HTML on exit:
python3 nano_tracker.py --config camera_config.json --duration 60

# Run until Ctrl+C:
python3 nano_tracker.py --config camera_config.json
```

While it's running, watch system load in another shell:

```bash
sudo jtop      # from jetson-stats: live GPU/NVDEC/CPU/RAM utilisation
```

Outputs land in `output/session_<timestamp>/`:

  - `session_<timestamp>_events.jsonl` -- append-on-finalize event log (one
    JSON object per line, `fsync`ed)
  - `session_<timestamp>_summary.html` -- live dashboard (sortable table,
    thumbnails, auto-refresh every 15s)
  - `session_<timestamp>_data.json` -- consolidated snapshot, written at
    shutdown only
  - `vehicle_<id>_first.jpg` / `vehicle_<id>_last.jpg` -- thumbnail crops
  - `index.html` -- redirects to the current summary

NanoTracker also runs an HTTP server on port 8080 (bound to all
interfaces).  From any browser on the LAN:

  **http://&lt;nano-ip&gt;:8080/**

Find the Nano's IP with `hostname -I` on the Nano itself.

## Troubleshooting

**`Failed to open GStreamer RTSP pipeline`** -- check the URL is reachable
(`ping`, `ffprobe rtsp://...`), and that the codec in `camera_config.json`
matches the stream (h264 vs h265).  Reolink cameras often expose H.265 on
the path `/h264Preview_01_main` -- the path name is misleading.

**`Failed to deserialise engine`** -- the engine was built against a
different TRT version.  Rebuild on the Nano with `scripts/build_engine.sh`.

**Out of memory during inference** -- swap the sub stream in
`camera_config.json`, or drop `inference.input_size` from 640 to 416.

**FPS lower than expected** -- check `sudo jtop`:
  - GPU at ~100% during inference -> you're GPU-bound, this is the limit
  - GPU low + NVDEC high -> decode is the bottleneck (uncommon on sub stream)
  - GPU low + CPU pegged -> NMS / numpy postprocessing on CPU is bottleneck;
    consider lowering `input_size` or `conf_threshold`
