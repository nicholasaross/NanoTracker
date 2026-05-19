# StreetTracker

Detect, track, and summarize moving vehicles and persons from video files or
a live RTSP stream. Emits a sortable HTML dashboard plus JSON / JSONL for
downstream analysis.

Successor to `VehicleTracker` (file-input only, dev-box-only) and
`NanoTracker` (live RTSP on Jetson Nano original / JetPack 4.6.1 / Python
3.6). Unifies both onto a single modern stack: Python 3.12, Ultralytics +
BotSORT, TensorRT, uv-managed.

## Hardware targets

| Role | Hardware | Status |
|---|---|---|
| Primary device | Jetson Orin Nano 8GB Super (67 TOPS, JetPack 6.x / 7.x) | active |
| Dev box | Linux / macOS / Windows (CUDA optional) | active |
| Original Jetson Nano | JetPack 4.6.1 | optional archive/web-host role only |

## Quick start (dev box)

```bash
uv sync
uv run streettracker batch path/to/video.mp4
```

## Quick start (Orin Nano)

```bash
scripts/setup_orin.sh        # idempotent JetPack + uv + Python 3.12 install
uv sync
uv run streettracker run --config camera_config.json
```

## CLI

- `streettracker run` — live RTSP capture (Orin-only)
- `streettracker batch <video>` — file input
- `streettracker pull` — rsync session from device
- `streettracker recolor <session>` — rerun color heuristic on a closed session
- `streettracker export-engine` — `.pt` → `.engine` via Ultralytics

See `CLAUDE.md` for architecture and operational notes.
