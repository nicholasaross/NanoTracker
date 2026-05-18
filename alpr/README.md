# alpr/ — post-processing ALPR comparison

Runs an ALPR (Automatic License Plate Recognition) pass over pulled NanoTracker
session `_main_*.jpg` snaps and compares accuracy between two end-to-end pipelines.

**Dev-box only.**  Do not install on the Nano (Python 3.6, no torch).

## Pipelines

| Name | Detector | OCR |
|---|---|---|
| `bespoke` | `license_plate_detector.pt` (Ultralytics YOLOv8) | EasyOCR |
| `preferred` | open-image-models (ONNX) | fast-plate-ocr (ONNX) |
| `ablation_bespokedet_fastocr` (optional) | `license_plate_detector.pt` | fast-plate-ocr |

The ablation isolates whether the bespoke detector or the EasyOCR head is the weak
link in the bespoke pipeline.

## Install

```bash
pip install -e .[alpr]
```

`easyocr` pulls torch (~1 GB) on first install.  ONNX models for the preferred
pipeline download lazily on first call (a few MB each).

## Place the bespoke model

The bespoke `.pt` lives at `D:\Projects\NanoTracker\license_plate_detector.pt`
(outside the worktree, untracked).  Copy it into `alpr/models/`:

```powershell
Copy-Item ..\..\..\license_plate_detector.pt alpr\models\
```

`*.pt` is globally gitignored, so this stays untracked.

## Workflow

```bash
# 1. Pull a session from the Nano (skip if you already have one).
python scripts/pull_session.py --only-main

# 2. Run both pipelines.  Add --ablation for the 3rd config.
python -m alpr.cli.run output/session_20260518_120000

# 3. Render the side-by-side comparison HTML.
python -m alpr.cli.report output/session_20260518_120000
# → open output/session_*/session_*_alpr_report.html

# 4. Label a sample (default filter is `disagreement` — most informative).
python -m alpr.cli.label output/session_20260518_120000

# 5. Score against labels.
python -m alpr.cli.score output/session_20260518_120000
# → output/session_*/session_*_alpr_scores.json
```

## Files written into the session directory

All paths sidecar to the existing session files; nothing in
`<session>_data.json` / `<session>_summary.html` is modified.

```
session_YYYYMMDD_HHMMSS/
├── <session>_alpr.json            # one record per (snap, pipeline)
├── <session>_alpr_by_track.json   # per-track best-of-N rollup
├── <session>_alpr_labels.json     # human labels, written by alpr.cli.label
├── <session>_alpr_report.html     # interactive review report
├── <session>_alpr_scores.json     # accuracy summary, written by alpr.cli.score
└── alpr_crops/
    ├── bespoke/<vehicle_<id>_main_<n>>.jpg
    └── preferred/<vehicle_<id>_main_<n>>.jpg
```

## Operational gotchas

These are inherited from how `nano_tracker.py` produces snaps (see [CLAUDE.md:227-302](../CLAUDE.md)):

- **Main and sub-stream FOV differ.**  Sub-stream bboxes do not map cleanly onto
  the 4K snap, so each pipeline re-detects on the full 4K frame.
- **Snap timing skews ~200-500 ms** behind the sub-stream frame that triggered
  the HTTP fetch.  At 30 mph that's ~5 m of motion — the plate is still in frame
  but at a different position than the trigger bbox.
- **Track IDs reset per session.**  Aggregate across sessions by `(session_label,
  track_id)` or `time_start_unix + class_name`, not `track_id` alone.
- **Plate readability varies a lot per snap.**  `_main_N.jpg` files for N=1..3
  are not ordered by quality — each is just a sample at a different bbox area.

## Out of scope (v1)

- Merging best plate back into `<session>_data.json` / patching
  `<session>_summary.html` (the recolor_session.py pattern).  Sidecar files are
  enough for the comparison question.
- Cross-session aggregation / plate-frequency analysis.
- Region-specific OCR models — fast-plate-ocr's global model is the default.
  Override via `--ocr-model NAME`.
- GPU acceleration.  Both pipelines default to CPU; flip with `--gpu` (EasyOCR only).
