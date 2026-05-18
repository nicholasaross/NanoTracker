# NanoTracker

Live vehicle tracker for the Jetson Nano (original, 4GB, JetPack 4.6.1),
sized to fit Maxwell + CUDA 10.2 + Python 3.6 + TensorRT 8.2.  Forked
in spirit from [VehicleTracker](../VehicleTracker) -- same outputs (JSON +
HTML summary + thumbnails), different runtime.

## Why a separate project?

VehicleTracker's stack (Python 3.12, PyTorch 2.10, Ultralytics 8.4,
YOLO26m) is incompatible with the original Jetson Nano end-to-end.
Rather than degrade VehicleTracker, NanoTracker is a thin Nano-native
runtime sharing only the output format.

| | VehicleTracker | NanoTracker |
|---|---|---|
| Target | RTX 3080 / Apple Silicon | Jetson Nano |
| Python | 3.12 | 3.6.9 |
| Inference | Ultralytics YOLO26m | TensorRT YOLOv8n FP16 |
| Tracker | BoTSORT / ByteTrack | Custom IoU |
| Input | local video file | live RTSP H.265 via NVDEC |
| Output | annotated mp4 + JSON + HTML | append-on-finalize JSONL + thumbnails + idle-regenerated HTML |

## Quickstart

See [docs/setup_nano.md](docs/setup_nano.md) for the full Nano install.
Short version:

```bash
# DEV BOX (Windows/Mac/Linux, Python 3.9+)
pip install "ultralytics>=8.0,<9.0" onnx onnxsim
python scripts/export_onnx.py
scp yolov8n.onnx <user>@<nano>:~/NanoTracker/

# NANO (one-time)
./scripts/setup_nano.sh
./scripts/build_engine.sh yolov8n.onnx

# NANO (each run)
cp camera_config.example.json camera_config.json
# edit IP + password
python3 nano_tracker.py --config camera_config.json --duration 60
```

## Layout

```
NanoTracker/
├── nano_tracker.py            # main entry; tracker, attributes, output
├── trt_engine.py              # TensorRT inference + YOLOv8 decode + NMS
├── gst_source.py              # GStreamer NVDEC RTSP source
├── camera_config.example.json # config schema (mirrors ReolinkDemo)
├── requirements.txt           # Nano-side pip deps (Py3.6 compatible)
├── pyproject.toml             # project metadata + optional dev-box deps
├── scripts/
│   ├── export_onnx.py         # DEV BOX: YOLOv8n.pt -> .onnx
│   ├── build_engine.sh        # NANO: .onnx -> .engine via trtexec
│   ├── setup_nano.sh          # NANO: apt + pip one-shot
│   ├── pull_session.py        # DEV BOX: scp latest session back for analysis
│   ├── recolor_session.py     # DEV BOX: re-vote colors on a closed session
│   └── debug_color.py         # DEV BOX: tune HSV ranges interactively
└── docs/
    └── setup_nano.md          # detailed setup with verification
```

## Live dashboard

NanoTracker runs a built-in HTTP server on port 8080 by default.  From any
device on your LAN: **http://&lt;nano-ip&gt;:8080/**.  Disable with `--no-http`
or `http.enabled: false` in the config; change port with `--http-port 9000`
or `http.port` in the config.

The page **never reloads itself** -- it polls `vehicles.json` every 15s
and re-renders rows in place, so your sort, scroll position, and active
tab survive auto-updates.  Two tabs split the table by class (Cars /
People); the active tab is reflected in the URL hash (`#tab=cars`) and
restored on manual reload.

## Output durability

A live RTSP run can be hours or days long, so writing only at shutdown
would risk losing whole sessions to power loss or a crash.  Instead:

  - `{session}_events.jsonl` is appended (with `fsync`) the moment a track
    is finalized.  Crash loss is bounded to tracks still active in memory
    at crash time -- never tracks already written.
  - `{session}_summary.html` is regenerated **only during idle periods**:
    no active tracks AND `html_idle_seconds` (default 10s) have elapsed
    since the last finalization.  Also regenerated on graceful shutdown.
    Served live via the built-in HTTP server.
  - `{session}_data.json` is the consolidated final snapshot, written
    only at shutdown.  For incremental consumers, prefer the JSONL.
  - Per-track image assets are written at finalize-time alongside the
    JSONL append, with a class-aware prefix (`vehicle_` or `person_`):
    - `<prefix>_<id>.jpg` -- mid-journey crop, used as the dashboard tile.
    - `<prefix>_<id>_hq.jpg` -- best-of-track sub-stream crop (area *
      sharpness, non-edge); the click-through for the tile.
    - `<prefix>_<id>_main_<N>.jpg` -- up to 3 full-resolution Reolink
      main-stream snapshots per close-pass vehicle, fetched via async
      HTTP at increasing area thresholds (approach / peak / departure).
  - `vehicles.json` is rewritten alongside every `summary.html` regen
    and is what the dashboard polls; same shape as the embedded data.
  - `index.html` is a one-line redirect to the latest summary, so
    `http://<nano>:8080/` always lands on the current session.

The default class filter is `[2]` (car only) in the example config; the
deployed Nano config typically runs `[0, 2]` (person + car).  Set
`inference.vehicle_classes` in your config to add motorcycle (3), bus (5),
truck (7), or other COCO classes -- the `CLASS_NAMES` map in `trt_engine.py`
covers the common ones so the dashboard's Type column labels correctly.
See [camera_config.example.json](camera_config.example.json).

## Pulling captures for post-processing

`scripts/pull_session.py` brings a session directory back to the dev box
over scp.  Defaults match the documented SSH setup (host=nano,
user=claude, key=~/.ssh/nanotracker_claude); files land at
`./output/<session>/` mirroring the Nano layout, so the bundled
`summary.html` resolves all image links locally.

```bash
# from the dev box, in your worktree
python scripts/pull_session.py                  # latest session, full bundle
python scripts/pull_session.py --only-main      # just 4K snaps + JSON (for ALPR)
python scripts/pull_session.py --dry-run        # inventory only, no transfer
python scripts/pull_session.py --session session_20260518_141443
python scripts/pull_session.py --target D:\Captures
```

The script issues one SSH round-trip to report total size + file counts
(`du`, `find`) before invoking scp, so you can gauge cost mid-session.
Re-running is safe -- scp merges into the existing local copy.

For analysis ergonomics, the consolidated post-processing inputs are:

  - `{session}_data.json` -- the full vehicle record list (close to the
    JSONL with metadata-only fields).
  - `{session}_hourly.json` -- per-hour rollup with class / color /
    direction / lane breakdowns and IR-period gaps.
  - `{prefix}_{id}_main_{N}.jpg` -- the high-resolution 4K stills for
    ALPR / make-model classification / re-color (see schema notes in
    [CLAUDE.md](CLAUDE.md)).

## Perf notes

Initial expectation on Jetson Nano with YOLOv8n FP16 at 640x640:

  - **Inference:** 60-90ms per frame (~11-17 FPS GPU-bound)
  - **NVDEC:** negligible CPU; sub stream is 640x480 H.264 ~25fps
  - **Pipeline overhead:** ~10ms (letterbox + NMS in numpy)
  - **Effective end-to-end:** **8-14 FPS** depending on traffic density

If sub stream is unavailable and main stream (4096x2784 H.265) is the only
option, expect significant memory pressure on the 4GB Nano -- consider
GStreamer-side downscale via `nvvidconv` before appsink.
