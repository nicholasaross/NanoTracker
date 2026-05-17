#!/usr/bin/env python3
"""NanoTracker -- live RTSP vehicle tracker for Jetson Nano (JetPack 4.6.1).

Pipeline:

    RTSP H.265 -> GStreamer (NVDEC nvv4l2decoder) -> BGR numpy frame
        -> TensorRT YOLOv8n FP16 inference -> detections
        -> simple IoU tracker -> live track state
        -> on track finalize -> append JSONL + save thumbnails
        -> when system idle -> regenerate HTML summary

Output durability model: ``events.jsonl`` is appended (with flush+fsync) the
instant a track finalizes, so a crash loses only tracks still active in
memory at crash time -- never tracks already written.  The HTML summary is
regenerated during idle periods (no active tracks for N seconds) and on
graceful shutdown.

This is the Nano-optimised cousin of VehicleTracker.  It is intentionally
simpler than the upstream project:

  - Live, real-time (no two-pass render).
  - No BoTSORT (depends on ultralytics).  Custom lightweight IoU tracker.
  - No annotated video output (would compete with NVDEC for memory bw).
  - No fragment merging / dedup post-processing (those make sense for batch).

Run on the Nano:

    python3 nano_tracker.py --config camera_config.json
    python3 nano_tracker.py --config camera_config.json --duration 60   # 60s perf test
"""

import argparse
import base64
import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

# Line-buffer stdout/stderr when redirected to a file (as in `nohup ... > log`).
# Without this, Python block-buffers ~8 KiB of [main] / per-frame lines and the
# log appears empty for minutes -- making it look like nothing is happening when
# in fact the pipeline is processing frames.  No-op when stdout is already a tty.
try:
    if not sys.stdout.isatty():
        sys.stdout = os.fdopen(sys.stdout.fileno(), "w", buffering=1)
    if not sys.stderr.isatty():
        sys.stderr = os.fdopen(sys.stderr.fileno(), "w", buffering=1)
except Exception:
    pass  # best-effort; some sandboxes wrap stdio with non-fd objects

# --- Prefer NVIDIA's L4T OpenCV (4.1.1, with GStreamer + NVDEC) over the
# Ubuntu Universe python3-opencv (3.2.0, no GStreamer).  Both are installed
# on JetPack 4.6.1 by default; Python's default sys.path order picks the
# Ubuntu one first.  Inserting the L4T path makes cv2.VideoCapture(...,
# CAP_GSTREAMER) actually work, which is required for NVDEC.
_L4T_CV2_PATH = "/usr/lib/python3.6/dist-packages"
_UBUNTU_CV2_PATH = "/usr/lib/python3/dist-packages"
if sys.platform.startswith("linux"):
    try:
        if Path(_L4T_CV2_PATH, "cv2").is_dir() and _UBUNTU_CV2_PATH in sys.path:
            # Reorder sys.path so the L4T cv2/ package (4.1.1, GStreamer YES)
            # is found BEFORE Ubuntu's cv2.so (3.2.0, GStreamer NO).  Insert
            # right before the Ubuntu entry rather than at position 0 -- the
            # L4T cv2 bootstrap does `sys.path.insert(1, ...)` to expose its
            # .so, and a position-0 entry pointing at the parent dir would
            # shadow that .so with the package directory and cause a
            # recursion error during cv2 import.
            while _L4T_CV2_PATH in sys.path:
                sys.path.remove(_L4T_CV2_PATH)
            sys.path.insert(sys.path.index(_UBUNTU_CV2_PATH), _L4T_CV2_PATH)
    except Exception:
        pass  # not Jetson, or unexpected layout -- just use default cv2

import numpy as np

from trt_engine import CLASS_NAMES, VEHICLE_CLASSES, Detection, TRTYolo
from gst_source import GstRtspSource, GstFileSource


# ----------------------------------------------------------------------
# Track data model
# ----------------------------------------------------------------------

class TrackPoint(NamedTuple):
    frame: int
    t: float       # seconds since session start
    cx: float
    cy: float
    x1: float
    y1: float
    x2: float
    y2: float
    score: float


_CROP_BUFFER_MAX = 12   # halve to ~6 when exceeded; bounds per-track memory
_CROP_PAD_FRAC = 0.2    # pad bbox by 20% per side -- context for ALPR / OCR


class CropSample(NamedTuple):
    t: float            # seconds since session start
    score: float        # YOLO detection confidence (used to pick color reference)
    crop: np.ndarray    # padded BGR crop


class Track(object):
    """Mutable per-vehicle state."""
    __slots__ = ("id", "class_id", "points", "misses", "crops", "finalized")

    def __init__(self, track_id, class_id):
        self.id = track_id
        self.class_id = class_id
        self.points = []          # type: List[TrackPoint]
        self.misses = 0
        # Padded crop samples spread across the track lifetime.  Capped at
        # _CROP_BUFFER_MAX; on overflow we halve via [::2] so survivors still
        # bracket the full duration.  Used at finalize-time to pick the
        # mid-journey thumbnail, the highest-confidence HQ image (for ALPR),
        # and color-vote samples.
        self.crops = []           # type: List[CropSample]
        self.finalized = False


# ----------------------------------------------------------------------
# IoU-based tracker (SORT-lite, no Kalman -- adequate for perf assessment)
# ----------------------------------------------------------------------

def iou(box_a, box_b):
    # type: (Tuple[float, float, float, float], Tuple[float, float, float, float]) -> float
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter == 0.0:
        return 0.0
    aa = (ax2 - ax1) * (ay2 - ay1)
    bb = (bx2 - bx1) * (by2 - by1)
    return inter / (aa + bb - inter)


class IoUTracker(object):
    """Greedy IoU matching, per-class.  No Kalman filter, no re-id.

    Tracks survive ``max_misses`` consecutive frames without a match, then
    are returned as finalized so the caller can compute attributes & write
    out their JSON entry.
    """

    def __init__(self, iou_threshold=0.30, max_misses=15):
        self.iou_threshold = iou_threshold
        self.max_misses = max_misses
        self._next_id = 1
        self._active = {}   # type: Dict[int, Track]

    def update(self, detections, frame, t, raw_frame):
        # type: (List[Detection], int, float, np.ndarray) -> List[Track]
        """Match detections to active tracks; return tracks that just expired."""
        # Group by class -- match within class only.
        det_by_class = {}  # type: Dict[int, List[int]]
        for i, d in enumerate(detections):
            det_by_class.setdefault(d.class_id, []).append(i)

        active_by_class = {}  # type: Dict[int, List[int]]
        for tid, tr in self._active.items():
            active_by_class.setdefault(tr.class_id, []).append(tid)

        matched_tids = set()
        matched_dets = set()

        for cls, det_idxs in det_by_class.items():
            track_ids = active_by_class.get(cls, [])
            if not track_ids:
                continue
            # Build IoU matrix (len(tracks), len(dets)).
            pairs = []
            for tid in track_ids:
                tr = self._active[tid]
                last = tr.points[-1]
                tbox = (last.x1, last.y1, last.x2, last.y2)
                for di in det_idxs:
                    d = detections[di]
                    dbox = (d.x1, d.y1, d.x2, d.y2)
                    score = iou(tbox, dbox)
                    if score >= self.iou_threshold:
                        pairs.append((score, tid, di))
            # Greedy: highest IoU first.
            pairs.sort(reverse=True)
            for score, tid, di in pairs:
                if tid in matched_tids or di in matched_dets:
                    continue
                matched_tids.add(tid)
                matched_dets.add(di)
                self._append_point(self._active[tid], detections[di], frame, t, raw_frame)

        # Unmatched detections -> new tracks.
        for i, d in enumerate(detections):
            if i in matched_dets:
                continue
            tr = Track(self._next_id, d.class_id)
            self._next_id += 1
            self._append_point(tr, d, frame, t, raw_frame)
            self._active[tr.id] = tr

        # Tick misses on unmatched tracks, finalize expired.
        expired = []  # type: List[Track]
        for tid in list(self._active.keys()):
            tr = self._active[tid]
            if tid not in matched_tids:
                tr.misses += 1
                if tr.misses > self.max_misses:
                    tr.finalized = True
                    expired.append(tr)
                    del self._active[tid]
            else:
                tr.misses = 0
        return expired

    def flush(self):
        """Finalize all remaining active tracks (called on shutdown)."""
        out = list(self._active.values())
        for tr in out:
            tr.finalized = True
        self._active.clear()
        return out

    def _append_point(self, tr, d, frame, t, raw_frame):
        # type: (Track, Detection, int, float, np.ndarray) -> None
        cx = (d.x1 + d.x2) / 2.0
        cy = (d.y1 + d.y2) / 2.0
        tr.points.append(TrackPoint(frame, t, cx, cy, d.x1, d.y1, d.x2, d.y2, d.score))
        h, w = raw_frame.shape[:2]
        crop = _safe_crop(raw_frame, d, w, h, pad_frac=_CROP_PAD_FRAC)
        if crop is not None:
            tr.crops.append(CropSample(t, d.score, crop))
            if len(tr.crops) > _CROP_BUFFER_MAX:
                tr.crops = tr.crops[::2]


# ----------------------------------------------------------------------
# IR / night-mode detection
#
# Reolink switches to IR LEDs at night, producing a monochrome image (R=G=B).
# YOLO trained on color images detects much less reliably in IR, and color
# voting becomes meaningless.  Rather than waste inference budget and pollute
# the JSON with low-quality IR-period entries, we detect this state per frame
# and skip yolo.infer() entirely while in IR mode.  Decode keeps running so
# RTSP buffer drains and we can detect the day-mode transition.
# ----------------------------------------------------------------------

_IR_CHANNEL_DIFF_THR = 8        # max per-pixel |R-G| / |G-B| for "monochrome"
_IR_SAMPLE_STRIDE = 16          # downsample factor for the cheap check
_IR_HYSTERESIS_FRAMES = 30      # consecutive readings before flipping state


def is_ir_frame(frame, channel_diff_thr=_IR_CHANNEL_DIFF_THR, stride=_IR_SAMPLE_STRIDE):
    # type: (np.ndarray, int, int) -> bool
    """True if the frame looks like monochrome IR (R≈G≈B everywhere).

    Stride-samples the frame for speed -- whole check is sub-millisecond on
    the Nano even at full 1080p.  Threshold may need calibration if your
    camera outputs tinted IR rather than pure mono (some Reolinks have a
    faint greenish cast); raise the threshold to be more tolerant."""
    s = frame[::stride, ::stride].astype(np.int16)
    diff_rg = int(np.abs(s[:, :, 2] - s[:, :, 1]).max())
    diff_gb = int(np.abs(s[:, :, 1] - s[:, :, 0]).max())
    return diff_rg <= channel_diff_thr and diff_gb <= channel_diff_thr


def _safe_crop(frame, d, w, h, pad_frac=0.0):
    # type: (np.ndarray, Detection, int, int, float) -> Optional[np.ndarray]
    """Crop the detection's bbox, optionally padded by pad_frac of its size on
    each side (clamped to frame bounds for vehicles near edges)."""
    px = (d.x2 - d.x1) * pad_frac
    py = (d.y2 - d.y1) * pad_frac
    x1 = max(0, int(d.x1 - px)); y1 = max(0, int(d.y1 - py))
    x2 = min(w, int(d.x2 + px)); y2 = min(h, int(d.y2 + py))
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2].copy()


# ----------------------------------------------------------------------
# Attribute computation (mirrors VehicleTracker/main.py:compute_attributes)
# ----------------------------------------------------------------------

# HSV color ranges, ordered low->high inclusive.
# Tuned for sub-stream (896x512) H.264-compressed traffic footage where
# chromatic colors lose saturation (e.g. burgundy red sits around S=80, V=60).
# `black` is intentionally restricted to chromaticity-free dark (S<40 AND V<40)
# so dark windows / wheel arches on a chromatic car don't outvote the paint.
COLOR_RANGES = [
    ((0, 0, 200),   (180, 30, 255),  "white"),
    ((0, 0, 0),     (180, 40, 40),   "black"),    # tightened: was V<50 any S
    ((0, 0, 40),    (180, 40, 200),  "grey"),     # extended down: was V>=51
    ((0, 60, 50),   (10, 255, 255),  "red"),      # Sat floor 60: was 100
    ((170, 60, 50), (180, 255, 255), "red"),
    ((100, 80, 50), (130, 255, 255), "blue"),
    ((36, 80, 50),  (85, 255, 255),  "green"),
    ((20, 50, 180), (30, 150, 255),  "silver"),
    ((20, 80, 80),  (35, 255, 255),  "yellow"),
]

_ACHROMATIC = frozenset(("white", "black", "grey", "silver"))
_CHROMATIC_PREFER_FRAC = 0.15  # when grey is the achromatic plurality, a
                               # chromatic >= this fraction of voted pixels
                               # wins (catches "real color buried under
                               # road-grey background").
_COLOR_MIN_INNER_PIXELS = 2000 # below this, the bbox is too tiny for the
                               # color vote to be reliable (a handful of
                               # sky-reflection pixels swings the result) --
                               # return "unknown" honestly.


def vote_color(crop, pad_frac=_CROP_PAD_FRAC):
    # type: (Optional[np.ndarray], float) -> str
    """Pick a vehicle color from a *padded* crop.

    Heuristics on top of plain HSV-range voting:

    1. Strip the padding before counting -- otherwise grey road/curb pixels
       (~30-40% of a padded crop) drown out the paint.
    2. Return "unknown" if the inner crop is below _COLOR_MIN_INNER_PIXELS --
       sub-2k-pixel bboxes can't be voted reliably (a few sky reflections in
       a 1000-pixel crop can swing the result; see distant-lane tracks).
    3. **white** or **black** plurality wins outright.  These are strong-
       signal categories (V>=200 S<30 / V<40 S<40) -- a real white or black
       car produces many pixels matching, and we should trust that over any
       reflection/shadow chromatic noise (the previous rule of "any
       chromatic >=15% wins" mis-categorised obvious white cars as blue
       because window+sky reflections add up to ~20%).
    4. Otherwise the achromatic plurality is grey/silver -- weak achromatic
       signal that often masks a desaturated chromatic body.  Then a
       chromatic with >= _CHROMATIC_PREFER_FRAC of the vote wins (this is
       the case the rule was actually designed for: burgundy red car whose
       body reads as mostly grey but has a clear chromatic minority).
    """
    import cv2  # type: ignore
    if crop is None or crop.size == 0:
        return "unknown"
    # The crop was made with `_safe_crop(..., pad_frac=p)`, which grew the
    # bbox by p on each side.  Padded extent is (1 + 2p) x bbox.  Original
    # bbox sits centered with `p / (1 + 2p)` inset on each side.
    h, w = crop.shape[:2]
    inset_x = int(w * pad_frac / (1.0 + 2.0 * pad_frac))
    inset_y = int(h * pad_frac / (1.0 + 2.0 * pad_frac))
    inner = crop[inset_y:max(inset_y + 1, h - inset_y),
                 inset_x:max(inset_x + 1, w - inset_x)]
    if inner.size == 0 or (inner.shape[0] * inner.shape[1]) < _COLOR_MIN_INNER_PIXELS:
        return "unknown"

    hsv = cv2.cvtColor(inner, cv2.COLOR_BGR2HSV)
    counts = {}  # type: Dict[str, int]
    for low, high, name in COLOR_RANGES:
        m = cv2.inRange(hsv, np.array(low), np.array(high))
        counts[name] = counts.get(name, 0) + int(cv2.countNonZero(m))
    total = sum(counts.values())
    if total == 0:
        return "unknown"

    chromatic = {k: v for k, v in counts.items() if k not in _ACHROMATIC}
    achromatic = {k: v for k, v in counts.items() if k in _ACHROMATIC}
    best_chrom_count = max(chromatic.values()) if chromatic else 0

    # White/black plurality wins outright over any chromatic.
    if achromatic:
        best_ach_name = max(achromatic, key=lambda k: achromatic[k])
        if best_ach_name in ("white", "black") and achromatic[best_ach_name] > best_chrom_count:
            return best_ach_name

    # Grey/silver plurality: defer to dominant chromatic if substantive.
    if chromatic:
        best_chrom_name = max(chromatic, key=lambda k: chromatic[k])
        if chromatic[best_chrom_name] >= _CHROMATIC_PREFER_FRAC * total:
            return best_chrom_name
    return max(counts, key=lambda k: counts[k])


def total_displacement(points):
    # type: (List[TrackPoint]) -> float
    tot = 0.0
    for i in range(1, len(points)):
        dx = points[i].cx - points[i - 1].cx
        dy = points[i].cy - points[i - 1].cy
        tot += (dx * dx + dy * dy) ** 0.5
    return tot


def format_wall(unix_ts):
    # type: (float) -> str
    """Local-time ISO 8601 with timezone offset, e.g. '2026-05-17T14:32:05+01:00'.

    Used for time_start / time_end in vehicle records so a daylong analysis
    can see when in real time each capture happened, rather than seconds
    since session start.  Python 3.6 has datetime.astimezone() with no
    argument (defaults to local tz)."""
    import datetime
    return datetime.datetime.fromtimestamp(unix_ts).astimezone().isoformat(timespec="seconds")


def compute_attributes(tr, frame_h, min_duration_s, parked_disp_px, color, t_start_wall):
    # type: (Track, int, float, float, str, float) -> Optional[dict]
    if len(tr.points) < 2:
        return None
    duration = tr.points[-1].t - tr.points[0].t
    if duration < min_duration_s:
        return None

    p0, pN = tr.points[0], tr.points[-1]
    net_disp = ((pN.cx - p0.cx) ** 2 + (pN.cy - p0.cy) ** 2) ** 0.5
    if net_disp < parked_disp_px:
        return None  # parked: not logged

    disp = total_displacement(tr.points)
    speed_px_s = net_disp / duration if duration > 0 else 0.0
    direction = "left to right" if pN.cx > p0.cx else "right to left"

    avg_y = sum(p.cy for p in tr.points) / len(tr.points)
    third = frame_h / 3.0
    lane = "top" if avg_y < third else ("middle" if avg_y < 2 * third else "bottom")

    avg_conf = sum(p.score for p in tr.points) / len(tr.points)
    return {
        "track_id": tr.id,
        "class_id": tr.class_id,
        "class_name": CLASS_NAMES.get(tr.class_id, "unknown"),
        "time_start": format_wall(t_start_wall + p0.t),
        "time_end": format_wall(t_start_wall + pN.t),
        "time_start_unix": round(t_start_wall + p0.t, 2),
        "time_end_unix": round(t_start_wall + pN.t, 2),
        "time_start_s": round(p0.t, 2),
        "time_end_s": round(pN.t, 2),
        "duration_visible": round(duration, 2),
        "direction": direction,
        "speed_px_s": round(speed_px_s, 1),
        "color": color,
        "lane": lane,
        "avg_confidence": round(avg_conf, 3),
        "displacement_px": round(disp, 1),
        "net_displacement_px": round(net_disp, 1),
        "num_detections": len(tr.points),
    }


# ----------------------------------------------------------------------
# Output: JSON, thumbnails, HTML summary
# ----------------------------------------------------------------------

def save_thumbnail(crop, path, quality=85):
    # type: (np.ndarray, Path, int) -> bool
    try:
        import cv2  # type: ignore
        return bool(cv2.imwrite(str(path), crop, [cv2.IMWRITE_JPEG_QUALITY, quality]))
    except Exception as e:  # pragma: no cover
        print("[output] thumbnail save failed for {}: {}".format(path, e))
        return False


def generate_html(attrs_list, output_dir, html_path, session_label, meta, refresh_seconds=15):
    # type: (List[dict], Path, Path, str, dict, int) -> None
    """Render the session dashboard.

    Always uses a virtualized renderer (server-emits JSON; browser renders
    only the visible window).  This handles small sessions and 24h+ sessions
    with thousands of rows equally well -- the DOM never grows past the
    visible viewport's worth of rows."""
    import html as html_mod

    class_counts = {}  # type: Dict[str, int]
    for v in attrs_list:
        class_counts[v["class_name"]] = class_counts.get(v["class_name"], 0) + 1
    parts = ["{} {}{}".format(c, n, "s" if c != 1 else "") for n, c in sorted(class_counts.items())]
    summary_text = "{} vehicle{}: {}".format(
        len(attrs_list), "s" if len(attrs_list) != 1 else "", ", ".join(parts)
    ) if attrs_list else "No vehicles detected"

    meta_kv = " · ".join("{}: {}".format(k, v) for k, v in sorted(meta.items()))
    refresh_tag = '<meta http-equiv="refresh" content="{}">'.format(int(refresh_seconds)) if refresh_seconds > 0 else ''

    # Serialize vehicle data to embed in the page.  Keep only the fields the
    # row renderer needs to minimise page size.
    data_json = json.dumps([
        {
            "track_id": v["track_id"],
            "class_name": v["class_name"],
            "color": v["color"],
            "time_start": v["time_start"],
            "time_start_unix": v["time_start_unix"],
            "duration_visible": v["duration_visible"],
            "direction": v["direction"],
            "speed_px_s": v["speed_px_s"],
            "lane": v["lane"],
            "avg_confidence": v["avg_confidence"],
        }
        for v in attrs_list
    ], separators=(",", ":"))

    # Pre-format styles + script using string concatenation (NOT .format) to
    # avoid CSS/JS brace collisions with .format() placeholders.
    page = (
        '<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">'
        + refresh_tag
        + '<title>NanoTracker -- ' + html_mod.escape(session_label) + '</title>'
        + '<style>'
        + '*{margin:0;padding:0;box-sizing:border-box}'
        + 'body{background:#1a1a1a;color:#e0e0e0;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,monospace;padding:24px}'
        + 'h1{font-size:1.4rem;margin-bottom:8px}'
        + '.summary{font-size:0.9rem;color:#aaa;margin-bottom:8px}'
        + '.meta{font-size:0.75rem;color:#777;margin-bottom:20px;font-family:monospace}'
        # 10-column grid: thumb 88, ID 56, Type 76, Color 76, Time 100, Duration 80, Direction 110, Speed 96, Lane 76, Conf 80
        + '.thead,.row{display:grid;grid-template-columns:88px 56px 76px 76px 100px 80px 110px 96px 76px 80px;gap:10px;padding:8px 12px;align-items:center;font-size:0.85rem}'
        + '.thead{background:#252525;color:#ccc;font-weight:600;border-bottom:1px solid #444;position:sticky;top:0;z-index:2}'
        + '.thead span{cursor:pointer;user-select:none}'
        + '.thead span:hover{color:#fff}'
        + '.thead span.sk-asc::after{content:" \\25B2"}'
        + '.thead span.sk-desc::after{content:" \\25BC"}'
        + '.vp{height:85vh;overflow-y:auto;border:1px solid #333;position:relative}'
        + '.spacer{position:relative}'
        + '.row{position:absolute;left:0;right:0;border-bottom:1px solid #2a2a2a}'
        + '.row:hover{background:#252525}'
        + '.row img{max-width:80px;max-height:54px;border-radius:4px;display:block}'
        + '.no-img{width:80px;height:54px;background:#333;border-radius:4px;display:flex;align-items:center;justify-content:center;color:#666;font-size:0.7rem}'
        + '</style></head><body>'
        + '<h1>NanoTracker Summary</h1>'
        + '<div class="summary">Session: ' + html_mod.escape(session_label) + ' &mdash; ' + html_mod.escape(summary_text) + '</div>'
        + '<div class="meta">' + html_mod.escape(meta_kv) + '</div>'
        + '<div class="vp" id="vp">'
        + '  <div class="thead">'
        + '    <span>Thumbnail</span>'
        + '    <span data-sk="track_id">ID</span>'
        + '    <span data-sk="class_name">Type</span>'
        + '    <span data-sk="color">Color</span>'
        + '    <span data-sk="time_start_unix" class="sk-desc">Time</span>'
        + '    <span data-sk="duration_visible">Duration</span>'
        + '    <span data-sk="direction">Direction</span>'
        + '    <span data-sk="speed_px_s">Speed</span>'
        + '    <span data-sk="lane">Lane</span>'
        + '    <span data-sk="avg_confidence">Conf</span>'
        + '  </div>'
        + '  <div class="spacer" id="spacer"><div id="rows"></div></div>'
        + '</div>'
        + '<script id="vehicles-data" type="application/json">' + data_json + '</script>'
        + '<script>'
        + 'const DATA=JSON.parse(document.getElementById("vehicles-data").textContent);'
        + 'const ROW_H=72;'
        + 'let sortKey="time_start_unix",sortAsc=false;'
        + 'let sorted=DATA.slice();'
        + 'const VP=document.getElementById("vp"),SPACER=document.getElementById("spacer"),ROWS=document.getElementById("rows");'
        + 'function esc(s){return String(s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",\'"\':"&quot;"}[c]))}'
        + 'function sortData(){sorted.sort((a,b)=>{let va=a[sortKey],vb=b[sortKey];if(typeof va==="number")return sortAsc?va-vb:vb-va;va=String(va);vb=String(vb);return sortAsc?va.localeCompare(vb):vb.localeCompare(va)})}'
        + 'function rowHtml(v,i){'
        +   'const t=(v.time_start||"").substr(11,8);'
        +   'const thumb="vehicle_"+v.track_id+".jpg";'
        +   'const hq="vehicle_"+v.track_id+"_hq.jpg";'
        +   'return `<div class="row" style="top:${i*ROW_H}px">'
        +     '<a href="${hq}" target="_blank" title="open full-quality crop"><img src="${thumb}" loading="lazy"></a>'
        +     '<span>#${v.track_id}</span>'
        +     '<span>${esc(v.class_name)}</span>'
        +     '<span>${esc(v.color)}</span>'
        +     '<span title="${esc(v.time_start)}">${t}</span>'
        +     '<span>${v.duration_visible}s</span>'
        +     '<span>${esc(v.direction)}</span>'
        +     '<span>${v.speed_px_s} px/s</span>'
        +     '<span>${esc(v.lane)}</span>'
        +     '<span>${v.avg_confidence.toFixed(3)}</span>'
        +   '</div>`'
        + '}'
        + 'function render(){'
        +   'SPACER.style.height=(sorted.length*ROW_H)+"px";'
        +   'const top=VP.scrollTop,visH=VP.clientHeight;'
        +   'const start=Math.max(0,Math.floor(top/ROW_H)-5);'
        +   'const end=Math.min(sorted.length,Math.ceil((top+visH)/ROW_H)+5);'
        +   'let h="";for(let i=start;i<end;i++)h+=rowHtml(sorted[i],i);'
        +   'ROWS.innerHTML=h;'
        + '}'
        + 'document.querySelectorAll(".thead span[data-sk]").forEach(s=>{'
        +   's.addEventListener("click",()=>{'
        +     'const k=s.dataset.sk;'
        +     'if(sortKey===k)sortAsc=!sortAsc;else{sortKey=k;sortAsc=true}'
        +     'document.querySelectorAll(".thead span").forEach(x=>x.classList.remove("sk-asc","sk-desc"));'
        +     's.classList.add(sortAsc?"sk-asc":"sk-desc");'
        +     'sortData();render();'
        +   '});'
        + '});'
        + 'VP.addEventListener("scroll",render);'
        + 'window.addEventListener("resize",render);'
        + 'sortData();render();'
        + '</script>'
        + '</body></html>'
    )
    html_path.write_text(page, encoding="utf-8")

    # Tiny index.html so http://<nano>:<port>/ lands on the latest summary
    # without needing to know the timestamped filename.
    index_path = output_dir / "index.html"
    if not index_path.exists() or index_path.stat().st_mtime < html_path.stat().st_mtime - 1:
        index_path.write_text(
            '<!DOCTYPE html><meta http-equiv="refresh" content="0; url={}">'.format(html_path.name),
            encoding="utf-8",
        )


def build_hourly_rollup(attrs_list, ir_periods):
    # type: (List[dict], List[dict]) -> dict
    """Bucket vehicles by wall-clock hour and summarise per-hour counts.

    Returns: {"hours": [{...per hour...}], "ir_periods": [...]}
    Each hour entry has hour ISO key, total count, and breakdowns by
    class / color / direction / lane.  IR periods are included alongside
    so 'no cars seen this hour' can be told apart from 'we were asleep'.
    """
    import datetime
    by_hour = {}  # type: Dict[str, dict]
    for v in attrs_list:
        unix_ts = v.get("time_start_unix")
        if unix_ts is None:
            continue
        hour_unix = int(unix_ts // 3600) * 3600
        hour_key = datetime.datetime.fromtimestamp(hour_unix).astimezone().isoformat(timespec="hours")
        bucket = by_hour.get(hour_key)
        if bucket is None:
            bucket = {
                "hour": hour_key,
                "count": 0,
                "by_class": {},
                "by_color": {},
                "by_direction": {},
                "by_lane": {},
            }
            by_hour[hour_key] = bucket
        bucket["count"] += 1
        for field, key in (("by_class", "class_name"), ("by_color", "color"),
                           ("by_direction", "direction"), ("by_lane", "lane")):
            val = v.get(key, "unknown")
            bucket[field][val] = bucket[field].get(val, 0) + 1
    return {
        "hours": sorted(by_hour.values(), key=lambda b: b["hour"]),
        "ir_periods": ir_periods,
    }


def save_json(attrs_list, meta, data_path, meta_path):
    # type: (List[dict], dict, Path, Path) -> None
    """Write the vehicles array as a bare top-level array (for jq / pandas /
    SQL ingestion), with session metadata in a sibling file."""
    data_path.write_text(json.dumps(attrs_list, indent=2), encoding="utf-8")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("[output] JSON data: {}".format(data_path))
    print("[output] JSON meta: {}".format(meta_path))


class EventLog(object):
    """Append-on-finalize JSONL event writer.

    Each finalized track is written as a single JSON line and immediately
    flushed + fsynced.  A crash loses only tracks that were still active
    in memory at crash time -- never tracks that were already finalized.
    """
    def __init__(self, path):
        # type: (Path) -> None
        self.path = path
        self._fh = open(str(path), "a", encoding="utf-8")
        self.count = 0

    def append(self, attrs):
        # type: (dict) -> None
        self._fh.write(json.dumps(attrs, separators=(",", ":")) + "\n")
        self._fh.flush()
        try:
            os.fsync(self._fh.fileno())
        except OSError:
            pass  # fsync can fail on some filesystems (tmpfs, etc.) -- not fatal
        self.count += 1

    def close(self):
        try:
            self._fh.close()
        except Exception:
            pass


# ----------------------------------------------------------------------
# Built-in HTTP server (serves the session output directory)
# ----------------------------------------------------------------------

def start_http_server(directory, host="0.0.0.0", port=8080):
    # type: (Path, str, int) -> Optional[object]
    """Start a daemon-thread HTTP server serving ``directory``.

    Returns the server object (so caller can .shutdown()) or None on failure.
    Python 3.6's SimpleHTTPRequestHandler doesn't accept a directory arg
    (that's 3.7+), so we subclass and override translate_path.
    """
    import os as _os
    import posixpath
    import threading
    import urllib.parse
    from http.server import HTTPServer, SimpleHTTPRequestHandler

    serve_dir = str(directory)

    class _Handler(SimpleHTTPRequestHandler):
        def translate_path(self, path):
            # Mirror SimpleHTTPRequestHandler.translate_path but root at serve_dir.
            path = path.split("?", 1)[0].split("#", 1)[0]
            trailing = path.rstrip().endswith("/")
            try:
                path = urllib.parse.unquote(path, errors="surrogatepass")
            except (UnicodeDecodeError, TypeError):
                path = urllib.parse.unquote(path)
            path = posixpath.normpath(path)
            words = [w for w in path.split("/") if w]
            full = serve_dir
            for word in words:
                if _os.path.dirname(word) or word in (_os.curdir, _os.pardir):
                    continue
                full = _os.path.join(full, word)
            if trailing:
                full += "/"
            return full

        def log_message(self, format, *args):
            return  # suppress per-request stdout spam

    try:
        server = HTTPServer((host, port), _Handler)
    except OSError as e:
        print("[http] Failed to bind {}:{} -- {}.  Dashboard disabled.".format(host, port, e))
        return None

    thread = threading.Thread(target=server.serve_forever, name="http-server", daemon=True)
    thread.start()
    return server


# ----------------------------------------------------------------------
# Config loading + main loop
# ----------------------------------------------------------------------

def load_config(path):
    # type: (str) -> dict
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_rtsp_url(cfg, password_override=None):
    # type: (dict, Optional[str]) -> Tuple[str, str]
    """Returns (url, codec) for the configured preferred stream."""
    cam = cfg["camera"]
    user = cam.get("username", "admin")
    password = password_override or os.environ.get("REOLINK_PASSWORD") or cam.get("password")
    if not password:
        sys.exit("[fatal] No password.  Set --password, $REOLINK_PASSWORD, or camera.password in config.")

    preferred = cfg.get("nano", {}).get("preferred_stream", "sub")
    streams = cfg.get("streams", [])
    chosen = next((s for s in streams if s.get("quality") == preferred), None)
    if chosen is None:
        sys.exit("[fatal] No stream with quality='{}' in config".format(preferred))

    ip = cam["ip"]
    port = cfg.get("ports", {}).get("rtsp", 554)
    url = "rtsp://{u}:{p}@{ip}:{port}{path}".format(
        u=user, p=password, ip=ip, port=port, path=chosen["path"],
    )
    codec = chosen.get("codec", "h265")
    return url, codec


def run(args):
    cfg = load_config(args.config)
    inf_cfg = cfg.get("inference", {})
    trk_cfg = cfg.get("tracking", {})
    out_cfg = cfg.get("output", {})
    nano_cfg = cfg.get("nano", {})

    # Source: --video <path> overrides RTSP for perf testing on recorded MP4.
    if args.video:
        codec = args.video_codec
        print("[main] Source: file {}  codec={}".format(args.video, codec))
    else:
        rtsp_url, codec = build_rtsp_url(cfg, args.password)
        safe_url = rtsp_url.replace(cfg["camera"].get("password") or "", "***") if cfg["camera"].get("password") else rtsp_url
        print("[main] RTSP: {}  codec={}".format(safe_url, codec))

    # Output dir
    session_label = "{}_{}".format(
        out_cfg.get("session_label_prefix", "session"),
        time.strftime("%Y%m%d_%H%M%S"),
    )
    output_dir = Path(out_cfg.get("dir", "output")) / session_label
    output_dir.mkdir(parents=True, exist_ok=True)
    print("[main] Output: {}".format(output_dir))

    # Inference engine
    engine_path = args.engine or inf_cfg.get("engine_path", "yolov8n_fp16.engine")
    if not Path(engine_path).exists():
        sys.exit("[fatal] Engine not found: {}.  Build it first via scripts/build_engine.sh".format(engine_path))
    print("[main] Loading TRT engine: {}".format(engine_path))
    yolo = TRTYolo(
        engine_path=engine_path,
        input_size=int(inf_cfg.get("input_size", 640)),
        conf_threshold=float(inf_cfg.get("conf_threshold", 0.30)),
        iou_threshold=float(inf_cfg.get("iou_threshold", 0.45)),
        class_filter=tuple(inf_cfg.get("vehicle_classes", VEHICLE_CLASSES)),
    )
    print("[main] Engine input shape: {}".format(yolo._input_shape))

    # GStreamer source -- file or RTSP.
    if args.video:
        src = GstFileSource(file_path=args.video, codec=codec)
    else:
        src = GstRtspSource(
            rtsp_url=rtsp_url,
            codec=codec,
            latency_ms=int(nano_cfg.get("latency_ms", 200)),
            transport=nano_cfg.get("rtsp_transport", "tcp"),
        )
    src.open()

    # Tracker
    tracker = IoUTracker(
        iou_threshold=float(trk_cfg.get("iou_match_threshold", 0.30)),
        max_misses=int(trk_cfg.get("max_misses", 15)),
    )

    # Output file paths (stable across the session).
    events_path = output_dir / "{}_events.jsonl".format(session_label)
    html_path = output_dir / "{}_summary.html".format(session_label)
    final_json_path = output_dir / "{}_data.json".format(session_label)
    meta_json_path = output_dir / "{}_meta.json".format(session_label)
    event_log = EventLog(events_path)
    print("[main] Event log:    {}".format(events_path))
    print("[main] HTML summary: {} (regenerated on idle)".format(html_path))

    # Write a placeholder index.html immediately so the HTTP dashboard shows
    # something useful from the very first request, rather than a directory
    # listing.  Overwritten by the real summary HTML on first idle regen.
    # Plain string concatenation (NOT .format) to avoid CSS-brace collisions.
    placeholder = (
        '<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">'
        '<meta http-equiv="refresh" content="10">'
        '<title>NanoTracker - waiting</title>'
        '<style>body{background:#1a1a1a;color:#e0e0e0;'
        'font-family:-apple-system,"Segoe UI",monospace;padding:40px;}'
        'h1{font-size:1.4rem;margin-bottom:12px;}p{color:#aaa;}</style>'
        '</head><body>'
        '<h1>NanoTracker - waiting for first track</h1>'
        '<p>The dashboard appears here after the first track finalizes '
        '(track-end + idle period).</p>'
        '<p>This page auto-refreshes every 10s.</p>'
        '<p>Session: ' + session_label + '</p>'
        '</body></html>'
    )
    (output_dir / "index.html").write_text(placeholder, encoding="utf-8")

    # Built-in HTTP dashboard (opt-out via --no-http or http.enabled=false).
    http_cfg = cfg.get("http", {})
    http_enabled = http_cfg.get("enabled", True) and not args.no_http
    http_server = None
    if http_enabled:
        http_host = args.http_host or http_cfg.get("host", "0.0.0.0")
        http_port = int(args.http_port or http_cfg.get("port", 8080))
        http_server = start_http_server(output_dir, host=http_host, port=http_port)
        if http_server is not None:
            import socket
            try:
                lan_ip = socket.gethostbyname(socket.gethostname())
            except OSError:
                lan_ip = http_host if http_host != "0.0.0.0" else "<nano-ip>"
            print("[main] Dashboard:    http://{}:{}/  (also http://localhost:{}/)".format(
                lan_ip, http_port, http_port,
            ))

    # Run loop with SIGINT trap.
    stop_flag = {"stop": False}

    def handle_sig(signum, _frame):
        print("\n[main] Signal {} -- stopping after current frame.".format(signum))
        stop_flag["stop"] = True

    signal.signal(signal.SIGINT, handle_sig)
    signal.signal(signal.SIGTERM, handle_sig)

    t_start = time.monotonic()
    t_start_wall = time.time()  # for wall-clock timestamps in JSON
    deadline = t_start + args.duration if args.duration > 0 else None

    save_thumbs = bool(out_cfg.get("save_thumbnails", True))
    min_duration = float(trk_cfg.get("min_track_duration_s", 1.0))
    parked_disp = float(trk_cfg.get("parked_displacement_px", 50.0))
    html_idle_s = float(out_cfg.get("html_idle_seconds", 10.0))
    html_refresh_s = int(out_cfg.get("html_refresh_seconds", 15))
    heartbeat_interval = float(out_cfg.get("heartbeat_interval_s", 5.0))

    # Mutable runtime state (closed over by finalize / write_html / current_meta).
    frame_idx = 0
    frames_inferred = 0   # frames where yolo.infer() actually ran (excludes IR)
    frame_h = 0
    attrs_list = []       # type: List[dict]
    raw_track_count = 0   # finalized tracks before min-duration filter
    last_finalize_time = t_start
    html_dirty = False
    html_writes = 0
    last_log = t_start
    last_heartbeat = 0.0
    heartbeat_path = output_dir / ".heartbeat"

    # Reconnect state (RTSP only): wall-clock-based session_t makes per-conn
    # generator restarts transparent to the rest of the pipeline.
    is_file_source = getattr(src, "is_file", False)
    reconnect_backoff = 1.0  # seconds, doubles to 30 max on persistent failure

    # IR / night-mode state (see is_ir_frame() for rationale).
    ir_history = []       # type: List[bool]
    ir_mode = False
    ir_period_start_wall = None  # type: Optional[float]
    ir_periods = []       # type: List[dict]   # {start, end, duration_s}

    def current_meta():
        elapsed_ = time.monotonic() - t_start
        # Include any open IR period as still in-progress so consumers can see
        # it without waiting for a transition out.
        ir_out = list(ir_periods)
        if ir_period_start_wall is not None:
            ir_out.append({
                "start": format_wall(ir_period_start_wall),
                "end": None,
                "duration_s": round(time.time() - ir_period_start_wall, 1),
            })
        return {
            "session_label": session_label,
            "session_start": format_wall(t_start_wall),
            "session_start_unix": round(t_start_wall, 2),
            "rtsp_codec": codec,
            "engine": engine_path,
            "input_size": yolo.input_size,
            "frames_processed": frame_idx,
            "frames_inferred": frames_inferred,
            "duration_s": round(elapsed_, 1),
            "pipe_fps": round(frame_idx / elapsed_, 2) if elapsed_ > 0 else 0.0,
            "avg_infer_ms": round(yolo.avg_infer_ms, 2),
            "html_writes": html_writes,
            "raw_track_count": raw_track_count,
            "ir_periods": ir_out,
            "ir_mode_active": ir_mode,
        }

    def finalize(tr):
        # type: (Track) -> None
        """Compute attrs for a finalized track, persist, and update HTML state."""
        nonlocal raw_track_count, last_finalize_time, html_dirty
        raw_track_count += 1
        # Largest-bbox crop = most pixels on the vehicle.  Used both as the
        # HQ click-through image AND as the input to color voting (more
        # pixels => more reliable vote).  Previously HQ was a higher-quality
        # encode of the mid-journey crop -- but for far-lane tracks the
        # mid-journey crop is tiny and the HQ was just a marginally bigger
        # JPEG of the same tiny image; user couldn't read plates from it.
        # Picking the largest-bbox frame gives a meaningfully larger image
        # for tracks where the car had a close pass.
        best_crop = None  # type: Optional[np.ndarray]
        color = "unknown"
        if tr.crops:
            best = max(tr.crops, key=lambda c: c.crop.shape[0] * c.crop.shape[1])
            best_crop = best.crop
            color = vote_color(best_crop)
        attrs = compute_attributes(tr, frame_h or 1080, min_duration, parked_disp, color, t_start_wall)
        if attrs is None:
            return  # filtered: too short, or parked
        if save_thumbs and tr.crops:
            # Two crops per vehicle:
            #   vehicle_<id>.jpg     mid-journey crop @ q=85 -- dashboard tile;
            #                        mid-journey tends to have the best plate
            #                        angle (car square-on, not broadside).
            #   vehicle_<id>_hq.jpg  largest-bbox crop @ q=95 -- click-through
            #                        full-resolution; most pixels on vehicle
            #                        for inspection / ALPR / future re-color.
            midpoint_t = (tr.points[0].t + tr.points[-1].t) / 2.0
            mid = min(tr.crops, key=lambda c: abs(c.t - midpoint_t)).crop
            save_thumbnail(mid, output_dir / "vehicle_{}.jpg".format(tr.id))
            if best_crop is not None:
                save_thumbnail(best_crop, output_dir / "vehicle_{}_hq.jpg".format(tr.id), quality=95)
        event_log.append(attrs)
        attrs_list.append(attrs)
        last_finalize_time = time.monotonic()
        html_dirty = True

    def write_html():
        nonlocal html_writes, html_dirty
        html_writes += 1
        generate_html(attrs_list, output_dir, html_path, session_label,
                      current_meta(), refresh_seconds=html_refresh_s)
        html_dirty = False

    def process_frame(conn_t, frame):
        # type: (float, np.ndarray) -> None
        """Per-frame body: IR check, inference, tracking, idle HTML, heartbeat,
        status log.  Mutates loop-state through nonlocal closure vars."""
        nonlocal frame_idx, frames_inferred, frame_h
        nonlocal ir_mode, ir_period_start_wall, last_finalize_time
        nonlocal html_dirty, last_log, last_heartbeat

        frame_idx += 1
        if frame_h == 0:
            frame_h = frame.shape[0]

        # ``session_t`` is wall-clock seconds since session start for RTSP
        # (so reconnects keep a single contiguous timeline) and video-time
        # for file sources (so durations stay meaningful even when the Nano
        # processes faster or slower than realtime).
        if is_file_source:
            session_t = conn_t
        else:
            session_t = time.time() - t_start_wall

        # IR / night-mode check with hysteresis.
        ir_history.append(is_ir_frame(frame))
        if len(ir_history) > _IR_HYSTERESIS_FRAMES:
            ir_history.pop(0)
        if len(ir_history) == _IR_HYSTERESIS_FRAMES:
            if not ir_mode and all(ir_history):
                ir_mode = True
                ir_period_start_wall = time.time()
                print("[mode] entering IR/night at {} -- skipping inference".format(
                    format_wall(ir_period_start_wall),
                ))
                # Cut active tracks cleanly across the day/night boundary.
                for tr in tracker.flush():
                    finalize(tr)
            elif ir_mode and not any(ir_history):
                end_wall = time.time()
                ir_periods.append({
                    "start": format_wall(ir_period_start_wall),
                    "end": format_wall(end_wall),
                    "duration_s": round(end_wall - ir_period_start_wall, 1),
                })
                ir_period_start_wall = None
                ir_mode = False
                print("[mode] returning to day at {} -- resuming inference".format(
                    format_wall(end_wall),
                ))

        now = time.monotonic()

        if not ir_mode:
            dets = yolo.infer(frame)
            frames_inferred += 1
            expired = tracker.update(dets, frame_idx, session_t, frame)
            for tr in expired:
                finalize(tr)
        else:
            dets = []

        # Idle-triggered HTML regen: at least one track has finalized AND
        # `html_idle_s` seconds have passed since the most recent
        # finalization.  We deliberately do NOT require `len(active) == 0`:
        # scenes with permanently parked vehicles never empty out, which
        # would block HTML updates forever.  "Idle" here means "no new
        # finalizations recently", not "scene completely empty".
        if (html_dirty
                and not args.no_html
                and (now - last_finalize_time) >= html_idle_s):
            write_html()

        if now - last_heartbeat >= heartbeat_interval:
            try:
                heartbeat_path.write_text(
                    "{:.3f} {}\n".format(time.time(), "ir" if ir_mode else "day"),
                    encoding="utf-8",
                )
            except OSError as exc:
                print("[heartbeat] write failed: {}".format(exc))
            last_heartbeat = now

        log_every_s = 30.0 if ir_mode else 2.0
        if now - last_log >= log_every_s:
            elapsed = now - t_start
            fps = frame_idx / elapsed if elapsed > 0 else 0.0
            if ir_mode:
                print("[main] f={fr:>5}  pipe_fps={pf:>5.1f}  IR-mode (inference paused)  "
                      "done={dn:>4}".format(fr=frame_idx, pf=fps, dn=event_log.count))
            else:
                print(
                    "[main] f={fr:>5}  pipe_fps={pf:>5.1f}  infer={im:>5.1f}ms (avg {iem:>5.1f}ms)  "
                    "active={ac:>3}  done={dn:>4}  dets={nd:>2}  html={hw}".format(
                        fr=frame_idx, pf=fps, im=yolo.last_infer_ms, iem=yolo.avg_infer_ms,
                        ac=len(tracker._active), dn=event_log.count, nd=len(dets), hw=html_writes,
                    )
                )
            last_log = now

    try:
        # Outer reconnect loop: only meaningful for RTSP; files run once.
        while not stop_flag["stop"]:
            try:
                for _conn_idx, conn_t, frame in src.frames():
                    process_frame(conn_t, frame)
                    now_inner = time.monotonic()
                    if stop_flag["stop"] or (deadline is not None and now_inner >= deadline):
                        break
                # Source generator returned naturally.
                if is_file_source or stop_flag["stop"]:
                    break
                print("[rtsp] source ended after {} session frames; reconnecting in {:.1f}s".format(
                    frame_idx, reconnect_backoff,
                ))
            except Exception as exc:
                if is_file_source or stop_flag["stop"]:
                    raise
                print("[rtsp] source error: {}: {}; reconnecting in {:.1f}s".format(
                    type(exc).__name__, exc, reconnect_backoff,
                ))

            if stop_flag["stop"]:
                break

            # Flush in-flight tracks: they had a discontinuity at disconnect.
            for tr in tracker.flush():
                finalize(tr)

            # Sleep + reconnect with exponential backoff.
            try:
                src.close()
            except Exception:
                pass
            time.sleep(reconnect_backoff)
            try:
                src.open()
                reconnect_backoff = 1.0
                print("[rtsp] reconnected.")
            except Exception as exc:
                reconnect_backoff = min(reconnect_backoff * 2.0, 30.0)
                print("[rtsp] reopen failed: {}: {}; will retry (next backoff {:.1f}s)".format(
                    type(exc).__name__, exc, reconnect_backoff,
                ))
                # Loop back to try again.
                continue
    finally:
        # Flush any in-flight tracks (still active at shutdown).
        for tr in tracker.flush():
            finalize(tr)
        # Close any open IR period so the metadata reflects a complete history.
        if ir_period_start_wall is not None:
            end_wall = time.time()
            ir_periods.append({
                "start": format_wall(ir_period_start_wall),
                "end": format_wall(end_wall),
                "duration_s": round(end_wall - ir_period_start_wall, 1),
            })
            ir_period_start_wall = None
            ir_mode = False
        # Final consolidated outputs, regardless of dirty flag.
        if not args.no_html:
            write_html()
        if not args.no_json:
            save_json(attrs_list, current_meta(), final_json_path, meta_json_path)
            hourly_path = output_dir / "{}_hourly.json".format(session_label)
            hourly_path.write_text(
                json.dumps(build_hourly_rollup(attrs_list, ir_periods), indent=2),
                encoding="utf-8",
            )
            print("[output] Hourly:    {}".format(hourly_path))
        event_log.close()
        try:
            src.close()
        except Exception:
            pass
        if http_server is not None:
            try:
                http_server.shutdown()
            except Exception:
                pass

    elapsed = time.monotonic() - t_start
    pipe_fps = frame_idx / elapsed if elapsed > 0 else 0.0
    ir_total = sum(p.get("duration_s", 0.0) for p in ir_periods)
    print("\n[main] Loop ended.  {} frames in {:.1f}s -> {:.1f} pipe fps "
          "(inference ran on {} frames; IR-mode total {:.1f}s in {} periods)".format(
        frame_idx, elapsed, pipe_fps, frames_inferred, ir_total, len(ir_periods),
    ))
    print("[main] Summary: {} moving vehicles kept from {} raw tracks "
          "(filtered: too short or parked).  HTML written {} times.".format(
        len(attrs_list), raw_track_count, html_writes,
    ))


def parse_args():
    p = argparse.ArgumentParser(description="NanoTracker -- live RTSP vehicle tracker for Jetson Nano")
    p.add_argument("--config", default="camera_config.json", help="Camera + inference config (default: camera_config.json)")
    p.add_argument("--engine", default=None, help="Override engine_path from config")
    p.add_argument("--password", default=None, help="Camera password (overrides config + env)")
    p.add_argument("--duration", type=float, default=0.0,
                   help="Run for N seconds then exit (0 = until SIGINT). Use for perf tests.")
    p.add_argument("--no-json", action="store_true", help="Skip final JSON snapshot (JSONL still written)")
    p.add_argument("--no-html", action="store_true", help="Skip HTML summary")
    p.add_argument("--no-http", action="store_true", help="Disable the built-in HTTP dashboard server")
    p.add_argument("--http-host", default=None, help="HTTP bind host (overrides config; default 0.0.0.0)")
    p.add_argument("--http-port", type=int, default=None, help="HTTP bind port (overrides config; default 8080)")
    p.add_argument("--video", default=None,
                   help="Use a local MP4 file as input (NVDEC decode) instead of RTSP. For perf testing.")
    p.add_argument("--video-codec", default="h264", choices=["h264", "h265"],
                   help="Codec of --video file (default: h264)")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
