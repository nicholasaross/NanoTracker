# CLAUDE.md

Guidance for Claude Code when working in StreetTracker.

## Project overview

StreetTracker unifies VehicleTracker (dev-box, file input) and NanoTracker
(Jetson Nano original, live RTSP) onto a single Python 3.12 + Ultralytics
+ TensorRT stack, targeting Jetson Orin Nano 8GB Super as the primary
device.

Pipeline:

```
RTSP H.264/H.265   ┐
   or              ├─▶ FrameSource ─▶ Ultralytics YOLO(.engine).track()
MP4 (NVDEC on Orin)┘                  (BotSORT integrated)
                                          │
                                          ▼
                          attribute compute (direction/speed/color/lane)
                                          │
                                          ▼
                          per-track finalize ─▶ EventLog (jsonl, fsync)
                                          │
                                          ▼
                          idle ▶ regenerate summary HTML + hourly rollup
                                 ▶ on demand ▶ Reolink 4K HTTP snapshot
```

## Compatibility rules

- **Python 3.12.** Pin via `.python-version`. uv manages the install.
- **No Python 3.6 hacks.** sys.path reorder, NamedTuple-for-dataclass,
  `# type:` comments — all gone. Use `@dataclass(slots=True)` and PEP-604
  unions (`X | None`).
- **TRT engines are not portable** across GPU architectures. Always
  build engines ON the target device (Orin or dev-box-with-matching-GPU).

## Architecture

```
src/streettracker/
├── common/                 # shared across runtime + analysis
│   ├── schema.py           # TrackRecord, SessionMeta @dataclass
│   ├── color.py            # COLOR_RANGES + vote_color()
│   ├── summary.py          # HTML dashboard generation
│   ├── hourly.py           # build_hourly_rollup()
│   └── output.py           # EventLog, save_json, file-path helpers
├── inference/              # YOLO + BotSORT via Ultralytics
├── sources/                # RTSP (FFmpeg), file (NVDEC on Orin)
├── device/                 # Orin-only: live runtime, snapshotter, dashboard, IR
├── analysis/               # off-device: ALPR, recolor, make/model, re-id
└── cli/                    # `streettracker` entry + subcommands
```

Single import root: `from streettracker.common.schema import TrackRecord`.

## Device runtime notes (Orin Nano 8GB Super)

- JetPack 6.x ships Ubuntu 22.04 / Python 3.10 → install Python 3.12 via
  `uv python install 3.12`. JetPack 7.x ships Ubuntu 24.04 / Python 3.12
  natively. uv handles both transparently.
- Ultralytics' built-in TRT path (`YOLO('best.engine')`) replaces
  NanoTracker's hand-rolled `trt_engine.py` (manual YOLOv8 decode + numpy
  NMS) and bespoke IoU tracker.
- Live RTSP from Reolink: same FFmpeg-backend workaround as NanoTracker
  (`cv2.CAP_FFMPEG` + `OPENCV_FFMPEG_CAPTURE_OPTIONS=rtsp_transport;tcp`).
  Don't try GStreamer for live RTSP — it stalls on Reolink keyframes.
- MP4 input on Orin uses GStreamer + `nvv4l2decoder` (NVDEC). Works fine
  for file input; only the live-RTSP case is broken with cv2-GStreamer.

## Output schema (preserved from NanoTracker)

Per finalized track:

| File | Quality | Use |
|---|---|---|
| `{prefix}_{id}.jpg` | q=85, ~80px | dashboard tile |
| `{prefix}_{id}_hq.jpg` | q=95, ~250px | quick color/silhouette |
| `{prefix}_{id}_main_{N}.jpg` | 4K Reolink HTTP | ALPR / make-model |

`{prefix}` is `vehicle` or `person`. `N` is 1..max_snaps_per_track.

Session files:
- `{session}_events.jsonl` — appended line-per-track (crash-safe)
- `{session}_data.json` — array of records, written at session end
- `{session}_meta.json` — session-level metadata + IR periods
- `{session}_hourly.json` — per-hour rollup
- `{session}_summary.html` — dashboard
- `index.html` — auto-redirect to latest summary

JSON record fields: see `common/schema.py` (`TrackRecord`).

## Common tasks

- Run tests: `uv run pytest`
- Lint: `uv run ruff check src/ tests/`
- Format: `uv run ruff format src/ tests/`
- Type check: `uv run mypy src/`
- Batch on dev box: `uv run streettracker batch sample.mp4`
- Build TRT engine on device: `uv run streettracker export-engine yolov8m.pt`

## Migration status

This repo is a clean-slate replacement for VehicleTracker + NanoTracker.
Source-of-truth scaffolding is currently developed in NanoTracker's
`claude/nano-orin-setup-plan-CCWUD` branch under `streettracker/` and
mirrored here phase by phase via `cp -a`. See NanoTracker's `CLAUDE.md`
"Active migration: StreetTracker" section for the recipe.

| Phase | Scope | Status |
|---|---|---|
| 0 | repo init + pyproject + CI + configs | **done** |
| 1 | `common/`: schema, color, output, hourly, summary | **done** |
| 2 | `inference/` (Ultralytics runner) + `sources/` (RTSP, file) | **done** |
| 4a | `analysis/`: recolor + debug-color | **done** |
| 4b | `analysis/alpr/` wholesale port | pending |
| 5 | CLI: `pull`, `export-engine`, `setup_orin.sh`, systemd | pending |
| 3 | `device/`: live runtime, snapshotter, dashboard, IR | pending (Orin) |
| 6 | (opt) original Nano archive role | not started |
| 7 | cutover: archive both old repos | not started |

Tests at HEAD: **76 passing, ruff clean.**

Verify locally:

```bash
uv sync
uv run pytest
uv run ruff check src/ tests/
```

CLI smoke:

```bash
uv run streettracker --help
uv run streettracker --version
uv run streettracker recolor --help
uv run streettracker debug-color --help
```

`run` / `batch` / `pull` / `export-engine` print
"not yet implemented" until their phases land.

Until cutover (phase 7), VehicleTracker + NanoTracker remain
authoritative for their current targets.
