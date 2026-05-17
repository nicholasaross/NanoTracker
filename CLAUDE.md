# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project overview

NanoTracker is the Jetson Nano (original, JetPack 4.6.1) deployment of the
VehicleTracker pipeline.  Same goal (detect, track, classify moving
vehicles from a video source; emit a sortable HTML + JSON summary),
different runtime constraints:

  - Python 3.6.9 (no dataclasses, no f-string `=`, no `list[X]`, no walrus)
  - PyTorch / Ultralytics are NOT installed -- inference runs through
    TensorRT 8.2 via pycuda
  - Input is a live RTSP stream (Reolink, H.264 sub by default) decoded
    by OpenCV's bundled FFmpeg backend, **not** GStreamer.  MP4 file
    inputs use GStreamer + NVDEC.  See "GStreamer / OpenCV / NVDEC pitfalls" below.
  - Output is metadata-only (JSON + HTML + thumbnails); no annotated video

## Critical compatibility rules

  - **Python 3.6 syntax only.**  `from typing import List, Dict, Optional`
    everywhere; never use `list[int]` / `dict[str, X]` / `Optional[X] | None`.
    `dataclasses` is unavailable -- use `NamedTuple` or `__slots__` classes.
  - **Never import torch / torchvision / ultralytics.**  They won't install
    on the Nano.  Inference is pure TRT + numpy.
  - **TRT engines are not portable** across GPU architectures.  Always
    build engines ON the Nano via `scripts/build_engine.sh`.  The ONNX
    file produced by `scripts/export_onnx.py` IS portable.

## Architecture

```
nano_tracker.py              -- entry; tracker + attributes + output + HTTP dashboard
  ├── trt_engine.py:TRTYolo  -- engine load, infer, YOLOv8 decode, NMS
  └── gst_source.py
      ├── GstRtspSource  -- live RTSP via cv2.VideoCapture(url, CAP_FFMPEG)
      │                    (class name is historical; no GStreamer involved)
      └── GstFileSource  -- MP4 via GStreamer NVDEC (nvv4l2decoder)
```

The runtime loop is single-threaded by design -- on a 4GB Nano, async
producer/consumer queues tend to cause OOM more often than they help.
If decode-latency hides inference time, add an explicit thread later.

## Dev environment & deployment

You'll typically be editing on the **Windows dev box** (`D:\Projects\NanoTracker\...`)
and SSHing to the **Nano** to test.  The Nano is a separate machine -- none of the
runtime (TRT, pycuda, aarch64 OpenCV, GStreamer NVDEC) works on Windows, so do
not try to `python nano_tracker.py` on the dev box.

  - **Nano host:** `nano` (resolved via mDNS / hosts file -- works regardless
    of DHCP).  Reolink camera is at `192.168.1.72`.  Do NOT hardcode the
    Nano's LAN IP anywhere: it's DHCP-assigned and has moved at least once
    (was `.181`, now `.119`); the hostname is the stable handle.  To get
    the current LAN IPv4 when you need one (e.g. for the dashboard URL):
    `ssh -i ~/.ssh/nanotracker_claude claude@nano "hostname -I | tr ' ' '\n' | grep '^192.168'"`.
  - **SSH:** `ssh -i ~/.ssh/nanotracker_claude claude@nano`.  The working
    account is `claude`; the private key is named exactly `nanotracker_claude`
    (no extension) in the dev box's `~/.ssh/`.  No `Host nano` alias exists in
    `~/.ssh/config` by default -- pass `-i` explicitly or add one.  When the
    Nano's IP changes you may get `Host key verification failed` if you SSH
    by the new IP (the hostname entry in `known_hosts` is fine; new IPs are
    new entries) -- use `-o StrictHostKeyChecking=accept-new` for the first
    connection by IP.
  - **Working dir on the Nano:** `/home/claude/NanoTracker/`.  This is **NOT a
    git repository** -- files were copied
    in directly.  Deploy changes via `scp` of individual files; do not attempt
    `git pull` on the Nano.  (Future cleanup: `git init` + add origin so this
    becomes a normal pull workflow.)
  - **Camera password lives in `camera_config.json`** on the Nano (mode 600).
    `nano_tracker.py` reads it directly -- no `REOLINK_PASSWORD` env var needed
    when launching from the Nano.  Never copy `camera_config.json` from dev to
    Nano (it's gitignored on dev and only the `.example` template is checked
    in -- the dev copy has no real password).
  - **Dashboard:** `http://<nano-lan-ip>:8080/` while a session is running
    (see above for getting the current IP).  The tracker prints a dashboard
    URL on startup but it's often `127.0.1.1` from `gethostbyname` on the
    Nano -- that's loopback, useless from outside; use the LAN IP.  Index
    page auto-redirects to the latest summary HTML.

### Sync code changes from dev box to Nano

```bash
# from the dev box, in your worktree
scp -i ~/.ssh/nanotracker_claude nano_tracker.py claude@nano:/home/claude/NanoTracker/
ssh -i ~/.ssh/nanotracker_claude claude@nano "wc -c ~/NanoTracker/nano_tracker.py"
```

Gitignored on the dev side and Nano-only: `camera_config.json`, `*.engine`,
`*.onnx`, `output/`.  TRT engines are not portable across GPU architectures --
always build on the Nano.

### Launch / stop over SSH

```bash
# Launch nohup'd.  `< /dev/null` is essential -- without it ssh keeps the
# channel open even though python is daemonised, and your terminal hangs.
ssh claude@nano "cd ~/NanoTracker && nohup python3 -u nano_tracker.py \
    --config camera_config.json > nano_live.log 2>&1 < /dev/null & echo PID=\$!"

# That PID is the *bash wrapper*, not python.  To find the actual python
# process (needed for a graceful SIGTERM that flushes final JSON + HTML):
ssh claude@nano "ps -ef | awk '\$8 ~ /python3/ && /nano_tracker/'"
ssh claude@nano "kill <python_pid>"
```

## Common tasks

  - **Run on Nano (local shell):** `python3 nano_tracker.py --config camera_config.json --duration 60`
  - **Run from dev box:** see "Launch / stop over SSH" above.
  - **Rebuild engine:** `./scripts/build_engine.sh yolov8n.onnx`
  - **Verify NVDEC:** `gst-inspect-1.0 nvv4l2decoder`
  - **Verify OpenCV+GStreamer:** `python3 -c "import cv2; print(cv2.getBuildInformation())" | grep -i gstreamer`
  - **Watch live load:** `sudo jtop` (jetson-stats)

## Reolink path quirk

Many Reolink models expose H.265 at `/h264Preview_01_main` (the URL path
does not reflect the codec).  `camera_config.example.json` documents both
codecs; the runtime takes the codec from the chosen stream entry.

## GStreamer / OpenCV / NVDEC pitfalls on JetPack 4.6.1

Learned the hard way during phase 1 bring-up.  Read this before adding any
GStreamer-backed code paths.

### Two OpenCV installs; default sys.path picks the wrong one

JetPack ships **both**:

| Install | Path | GStreamer | Notes |
|---|---|---|---|
| Ubuntu Universe `python3-opencv` 3.2.0 | `/usr/lib/python3/dist-packages/cv2.cpython-36m-aarch64-linux-gnu.so` | **NO** | Wins default sys.path |
| NVIDIA L4T `libopencv-python` 4.1.1 | `/usr/lib/python3.6/dist-packages/cv2/` | **YES (1.14.5)** + NVDEC | What we want |

`nano_tracker.py` reorders `sys.path` at module top.  **Critical detail:**
insert the L4T path *immediately before* `/usr/lib/python3/dist-packages`,
NOT at position 0.  NVIDIA's cv2 bootstrap does `sys.path.insert(1, ...)`
to expose its real `.so`; a position-0 entry pointing at the parent dir
shadows that `.so` with the package directory and raises
`ImportError: recursion is detected` from `cv2/__init__.py`.

Quick check: `python3 -c "import nano_tracker; import cv2; print(cv2.__version__, [l for l in cv2.getBuildInformation().splitlines() if 'GStreamer' in l])"`
should print `4.1.1 ['GStreamer: YES (1.14.5)']`.

### Live RTSP: cv2.VideoCapture + GStreamer is broken for Reolink

Against this Reolink sub-stream over WiFi, **both** GStreamer decoder
paths fail through `cv2.VideoCapture(..., CAP_GSTREAMER)`:

- `nvv4l2decoder` (NVDEC): takes the first frame, then drops every
  subsequent one with `Stream format not found, dropping the frame`
  until the next IDR -- which Reolink keyframes long enough apart that
  the pipeline appears frozen.
- `avdec_h264` (software): never emits a single frame past PAUSED state.

Both pipelines work in standalone `gst-launch-1.0`.  The bug is in the
cv2 ↔ GStreamer ↔ Reolink interaction (suspect SPS/PPS handling +
rtpjitterbuffer + WiFi packet timing).  ReolinkDemo on the dev box hit
the same problem and works around it with an FFmpeg subprocess.

**Working path:** `cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)` with
`OPENCV_FFMPEG_CAPTURE_OPTIONS=rtsp_transport;tcp` exported before the
call.  Decode is software, but TRT inference (47ms/frame) dominates so
the difference vs NVDEC is <2 FPS.  This is what `GstRtspSource.open()`
now uses (the class name is now misleading -- rename to `RtspSource`
in phase 2).

### MP4 file input: GStreamer NVDEC works fine

`filesrc ! qtdemux ! h264parse ! nvv4l2decoder ! ...` via
`cv2.VideoCapture(..., CAP_GSTREAMER)` is stable for local MP4 files.
The bad case is live RTSP, not GStreamer in general.  `GstFileSource`
in `gst_source.py` uses NVDEC and is the right way to do perf testing
against recorded clips.

### Python and shell gotchas

- **stdout block-buffers ~8 KiB when redirected to a file** (`nohup ... > log`),
  so log files look empty for minutes even though the process is fine.
  `nano_tracker.py` line-buffers stdout/stderr at module top.  When
  running anything else, use `python3 -u`.
- **`pkill -f nano_tracker` SIGKILLs its own bash shell** because the
  literal string "nano_tracker" appears in the pkill command line and
  `-f` matches the whole command line.  Use a more specific pattern:
  `pkill -f "python3 -u nano_tracker"`.
- **`pgrep -af 'python3 -u nano_tracker'` matches the bash wrapper** that
  launched it -- because nohup'd shells keep the full python invocation
  visible in their own argv.  Recipes that actually return just the python
  PID: `ps -ef | awk '$8 ~ /python3/ && /nano_tracker/'` (filters on
  argv[0]==python3), or `pgrep -f 'nano_tracker.py' -P 1` (orphaned by
  nohup, so parent is init).
- **`pycuda 2021.1` install** on JetPack 4.6 needs both build-isolation
  off and PEP 517 off, plus CUDA headers on CPATH for the C++ build:
  `CPATH=/usr/local/cuda/include pip install --user --no-build-isolation --no-use-pep517 pycuda==2021.1`.
  `scripts/setup_nano.sh` does this.

### Confirmed perf baseline (Tegra X1 sm_53, YOLOv8n FP16, 640x640 input)

| Path | Pipe FPS | Inference ms | Source |
|---|---|---|---|
| Pure inference (trtexec benchmark) | 21.5 ceiling | 46.5 | n/a |
| MP4 via GStreamer + NVDEC | 13.1 | 47.5 | 1080p H.264 @ 20fps file |
| MP4 via FFmpeg (sw decode fallback) | 11.7 | 47.5 | 1080p H.264 @ 20fps file |
| Live RTSP via FFmpeg backend | 13.3 | 47.5 | Reolink sub-stream 896×512 H.264 @ 20fps |

Inference dominates (~76% of frame budget).  Decode path matters by ~2 FPS.
To push above 13 FPS the real lever is `input_size` 640→512 (~+5 FPS) or
640→416 (~+10 FPS) at some accuracy cost, not the decode pipeline.
