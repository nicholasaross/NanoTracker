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


class Track(object):
    """Mutable per-vehicle state."""
    __slots__ = ("id", "class_id", "points", "misses", "first_crop", "last_crop", "finalized")

    def __init__(self, track_id, class_id):
        self.id = track_id
        self.class_id = class_id
        self.points = []          # type: List[TrackPoint]
        self.misses = 0
        self.first_crop = None    # type: Optional[np.ndarray]
        self.last_crop = None     # type: Optional[np.ndarray]
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
        h, w = raw_frame.shape[:2]
        for i, d in enumerate(detections):
            if i in matched_dets:
                continue
            tr = Track(self._next_id, d.class_id)
            self._next_id += 1
            self._append_point(tr, d, frame, t, raw_frame)
            tr.first_crop = _safe_crop(raw_frame, d, w, h)
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
        crop = _safe_crop(raw_frame, d, w, h)
        if crop is not None:
            tr.last_crop = crop


def _safe_crop(frame, d, w, h):
    # type: (np.ndarray, Detection, int, int) -> Optional[np.ndarray]
    x1 = max(0, int(d.x1)); y1 = max(0, int(d.y1))
    x2 = min(w, int(d.x2)); y2 = min(h, int(d.y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2].copy()


# ----------------------------------------------------------------------
# Attribute computation (mirrors VehicleTracker/main.py:compute_attributes)
# ----------------------------------------------------------------------

COLOR_RANGES = [
    ((0, 0, 200),    (180, 30, 255),  "white"),
    ((0, 0, 0),      (180, 255, 50),  "black"),
    ((0, 0, 51),     (180, 40, 199),  "grey"),
    ((0, 100, 51),   (10, 255, 255),  "red"),
    ((170, 100, 51), (180, 255, 255), "red"),
    ((100, 100, 51), (130, 255, 255), "blue"),
    ((36, 100, 51),  (85, 255, 255),  "green"),
    ((20, 50, 180),  (30, 150, 255),  "silver"),
    ((20, 100, 100), (35, 255, 255),  "yellow"),
]


def get_dominant_color(crops):
    # type: (List[np.ndarray]) -> str
    import cv2  # type: ignore
    counts = {}  # type: Dict[str, int]
    for crop in crops:
        if crop is None or crop.size == 0:
            continue
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        for low, high, name in COLOR_RANGES:
            m = cv2.inRange(hsv, np.array(low), np.array(high))
            counts[name] = counts.get(name, 0) + int(cv2.countNonZero(m))
    if not counts:
        return "unknown"
    return max(counts, key=lambda k: counts[k])


def total_displacement(points):
    # type: (List[TrackPoint]) -> float
    tot = 0.0
    for i in range(1, len(points)):
        dx = points[i].cx - points[i - 1].cx
        dy = points[i].cy - points[i - 1].cy
        tot += (dx * dx + dy * dy) ** 0.5
    return tot


def format_time(seconds):
    # type: (float) -> str
    h = int(seconds) // 3600
    m = (int(seconds) % 3600) // 60
    s = int(seconds) % 60
    return "{:02d}:{:02d}:{:02d}".format(h, m, s)


def compute_attributes(tr, frame_h, min_duration_s, parked_disp_px):
    # type: (Track, int, float, float) -> Optional[dict]
    if len(tr.points) < 2:
        return None
    duration = tr.points[-1].t - tr.points[0].t
    if duration < min_duration_s:
        return None

    p0, pN = tr.points[0], tr.points[-1]
    net_disp = ((pN.cx - p0.cx) ** 2 + (pN.cy - p0.cy) ** 2) ** 0.5
    disp = total_displacement(tr.points)
    speed_px_s = net_disp / duration if duration > 0 else 0.0
    direction = "left to right" if pN.cx > p0.cx else "right to left"

    color = get_dominant_color([c for c in (tr.first_crop, tr.last_crop) if c is not None])

    avg_y = sum(p.cy for p in tr.points) / len(tr.points)
    third = frame_h / 3.0
    lane = "top" if avg_y < third else ("middle" if avg_y < 2 * third else "bottom")

    parked = net_disp < parked_disp_px

    avg_conf = sum(p.score for p in tr.points) / len(tr.points)
    return {
        "track_id": tr.id,
        "class_id": tr.class_id,
        "class_name": CLASS_NAMES.get(tr.class_id, "unknown"),
        "time_start_s": round(p0.t, 2),
        "time_end_s": round(pN.t, 2),
        "time_start": format_time(p0.t),
        "time_end": format_time(pN.t),
        "duration_visible": round(duration, 2),
        "direction": direction,
        "speed_px_s": round(speed_px_s, 1),
        "color": color,
        "lane": lane,
        "avg_confidence": round(avg_conf, 3),
        "displacement_px": round(disp, 1),
        "net_displacement_px": round(net_disp, 1),
        "num_detections": len(tr.points),
        "parked": parked,
    }


# ----------------------------------------------------------------------
# Output: JSON, thumbnails, HTML summary
# ----------------------------------------------------------------------

def save_thumbnail(crop, path):
    # type: (np.ndarray, Path) -> bool
    try:
        import cv2  # type: ignore
        return bool(cv2.imwrite(str(path), crop, [cv2.IMWRITE_JPEG_QUALITY, 85]))
    except Exception as e:  # pragma: no cover
        print("[output] thumbnail save failed for {}: {}".format(path, e))
        return False


def generate_html(attrs_list, output_dir, html_path, session_label, meta, refresh_seconds=15):
    # type: (List[dict], Path, Path, str, dict, int) -> None
    import html as html_mod

    class_counts = {}  # type: Dict[str, int]
    for v in attrs_list:
        class_counts[v["class_name"]] = class_counts.get(v["class_name"], 0) + 1
    moving = sum(1 for v in attrs_list if not v.get("parked"))
    parked = sum(1 for v in attrs_list if v.get("parked"))
    parts = ["{} {}{}".format(c, n, "s" if c != 1 else "") for n, c in sorted(class_counts.items())]
    counts_str = "{} moving, {} parked".format(moving, parked) if parked else "{} moving".format(moving)
    summary_text = "{} vehicle{} ({}): {}".format(
        len(attrs_list), "s" if len(attrs_list) != 1 else "", counts_str, ", ".join(parts)
    ) if attrs_list else "No vehicles detected"

    no_img = ('<div style="width:80px;height:54px;background:#333;border-radius:4px;'
              'display:flex;align-items:center;justify-content:center;color:#666;font-size:0.7rem;">N/A</div>')
    rows = []
    for v in attrs_list:
        imgs = []
        # Relative paths: HTML and thumbnails share output_dir, so the
        # browser fetches them as e.g. /vehicle_42_first.jpg.  This keeps
        # HTML size bounded (no base64 reflate per regen) and lets the
        # browser cache thumbnails across refreshes.
        for label in ("first", "last"):
            fname = "vehicle_{}_{}.jpg".format(v["track_id"], label)
            tp = output_dir / fname
            if tp.exists():
                imgs.append('<img src="{}" style="max-width:80px;max-height:54px;border-radius:4px;" loading="lazy">'.format(fname))
            else:
                imgs.append(no_img)
        thumb = '<div style="display:flex;gap:3px;">' + "".join(imgs) + "</div>"

        is_parked = v.get("parked", False)
        status = ('<span style="background:#553;color:#da5;padding:1px 6px;border-radius:3px;font-size:0.75rem;">parked</span>'
                  if is_parked else
                  '<span style="background:#354;color:#6c6;padding:1px 6px;border-radius:3px;font-size:0.75rem;">moving</span>')
        row_style = ' style="opacity:0.6;background:#1f1f1f;"' if is_parked else ''
        rows.append("""
        <tr{rs}>
          <td>{thumb}</td>
          <td data-sort="{tid}">#{tid}</td>
          <td>{st}</td>
          <td>{cn}</td>
          <td>{co}</td>
          <td data-sort="{ts}">{tstart} &rarr; {tend}</td>
          <td data-sort="{dur}">{dur}s</td>
          <td>{dir}</td>
          <td data-sort="{sp}">{sp} px/s</td>
          <td>{ln}</td>
          <td data-sort="{ac}">{ac:.3f}</td>
        </tr>""".format(
            rs=row_style, thumb=thumb, tid=v["track_id"], st=status,
            cn=html_mod.escape(v["class_name"]), co=html_mod.escape(v["color"]),
            ts=v["time_start_s"], tstart=v["time_start"], tend=v["time_end"],
            dur=v["duration_visible"], dir=html_mod.escape(v["direction"]),
            sp=v["speed_px_s"], ln=html_mod.escape(v["lane"]), ac=v["avg_confidence"],
        ))
    rows_joined = "\n".join(rows)

    meta_kv = " · ".join("{}: {}".format(k, v) for k, v in sorted(meta.items()))

    refresh_tag = '<meta http-equiv="refresh" content="{}">'.format(int(refresh_seconds)) if refresh_seconds > 0 else ''

    page = """<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
{refresh}
<title>NanoTracker Summary -- {label}</title>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ background:#1a1a1a; color:#e0e0e0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,monospace; padding:24px; }}
h1 {{ font-size:1.4rem; margin-bottom:8px; }}
.summary {{ font-size:0.9rem; color:#aaa; margin-bottom:8px; }}
.meta {{ font-size:0.75rem; color:#777; margin-bottom:20px; font-family:monospace; }}
table {{ border-collapse:collapse; width:100%; font-size:0.85rem; }}
th, td {{ padding:8px 10px; text-align:left; border-bottom:1px solid #333; }}
th {{ background:#252525; color:#ccc; cursor:pointer; user-select:none; position:sticky; top:0; }}
th:hover {{ color:#fff; }}
th.sorted-asc::after {{ content:" \\25B2"; }}
th.sorted-desc::after {{ content:" \\25BC"; }}
tr:hover {{ background:#252525; }}
</style></head><body>
<h1>NanoTracker Summary</h1>
<div class="summary">Session: {label} &mdash; {summary}</div>
<div class="meta">{meta}</div>
<table id="t"><thead><tr>
<th>Thumbnail</th><th data-col="1">ID</th><th data-col="2">Status</th><th data-col="3">Type</th>
<th data-col="4">Color</th><th data-col="5">Time</th><th data-col="6">Duration</th>
<th data-col="7">Direction</th><th data-col="8">Speed</th><th data-col="9">Lane</th>
<th data-col="10">Confidence</th>
</tr></thead><tbody>
{rows}
</tbody></table>
<script>
document.querySelectorAll('#t th[data-col]').forEach(th => {{
  th.addEventListener('click', () => {{
    const tb = document.querySelector('#t tbody');
    const col = parseInt(th.dataset.col);
    const rows = Array.from(tb.querySelectorAll('tr'));
    const asc = !th.classList.contains('sorted-asc');
    document.querySelectorAll('#t th').forEach(h => h.classList.remove('sorted-asc','sorted-desc'));
    th.classList.add(asc ? 'sorted-asc' : 'sorted-desc');
    rows.sort((a, b) => {{
      const ca = a.children[col], cb = b.children[col];
      let va = ca.dataset.sort !== undefined ? ca.dataset.sort : ca.textContent;
      let vb = cb.dataset.sort !== undefined ? cb.dataset.sort : cb.textContent;
      const na = parseFloat(va), nb = parseFloat(vb);
      if (!isNaN(na) && !isNaN(nb)) return asc ? na - nb : nb - na;
      return asc ? va.localeCompare(vb) : vb.localeCompare(va);
    }});
    rows.forEach(r => tb.appendChild(r));
  }});
}});
</script></body></html>""".format(
        refresh=refresh_tag,
        label=html_mod.escape(session_label),
        summary=html_mod.escape(summary_text),
        meta=html_mod.escape(meta_kv),
        rows=rows_joined,
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


def save_json(attrs_list, meta, json_path):
    # type: (List[dict], dict, Path) -> None
    json_path.write_text(json.dumps({"metadata": meta, "vehicles": attrs_list}, indent=2),
                         encoding="utf-8")
    print("[output] JSON data: {}".format(json_path))


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
    deadline = t_start + args.duration if args.duration > 0 else None

    save_thumbs = bool(out_cfg.get("save_thumbnails", True))
    min_duration = float(trk_cfg.get("min_track_duration_s", 1.0))
    parked_disp = float(trk_cfg.get("parked_displacement_px", 50.0))
    html_idle_s = float(out_cfg.get("html_idle_seconds", 10.0))
    html_refresh_s = int(out_cfg.get("html_refresh_seconds", 15))

    # Mutable runtime state (closed over by finalize / write_html / current_meta).
    frame_idx = 0
    frame_h = 0
    attrs_list = []       # type: List[dict]
    raw_track_count = 0   # finalized tracks before min-duration filter
    last_finalize_time = t_start
    html_dirty = False
    html_writes = 0
    last_log = t_start

    def current_meta():
        elapsed_ = time.monotonic() - t_start
        return {
            "session_label": session_label,
            "rtsp_codec": codec,
            "engine": engine_path,
            "input_size": yolo.input_size,
            "frames_processed": frame_idx + 1,
            "duration_s": round(elapsed_, 1),
            "pipe_fps": round((frame_idx + 1) / elapsed_, 2) if elapsed_ > 0 else 0.0,
            "avg_infer_ms": round(yolo.avg_infer_ms, 2),
            "html_writes": html_writes,
            "raw_track_count": raw_track_count,
        }

    def finalize(tr):
        # type: (Track) -> None
        """Compute attrs for a finalized track, persist, and update HTML state."""
        nonlocal raw_track_count, last_finalize_time, html_dirty
        raw_track_count += 1
        attrs = compute_attributes(tr, frame_h or 1080, min_duration, parked_disp)
        if attrs is None:
            return  # filtered out by min duration
        if save_thumbs:
            if tr.first_crop is not None:
                save_thumbnail(tr.first_crop, output_dir / "vehicle_{}_first.jpg".format(tr.id))
            if tr.last_crop is not None:
                save_thumbnail(tr.last_crop, output_dir / "vehicle_{}_last.jpg".format(tr.id))
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

    try:
        for frame_idx, t, frame in src.frames():
            # ``t`` semantics:
            #   - RTSP source: wall-clock seconds since first frame
            #   - File source: video time (frame_idx / fps) -- keeps speed/duration
            #     attributes meaningful even when the Nano processes faster or
            #     slower than realtime.
            if frame_h == 0:
                frame_h = frame.shape[0]

            dets = yolo.infer(frame)
            expired = tracker.update(dets, frame_idx, t, frame)
            for tr in expired:
                finalize(tr)

            now = time.monotonic()

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

            if now - last_log >= 2.0:
                elapsed = now - t_start
                fps = (frame_idx + 1) / elapsed if elapsed > 0 else 0.0
                print(
                    "[main] f={fr:>5}  pipe_fps={pf:>5.1f}  infer={im:>5.1f}ms (avg {iem:>5.1f}ms)  "
                    "active={ac:>3}  done={dn:>4}  dets={nd:>2}  html={hw}".format(
                        fr=frame_idx, pf=fps, im=yolo.last_infer_ms, iem=yolo.avg_infer_ms,
                        ac=len(tracker._active), dn=event_log.count, nd=len(dets), hw=html_writes,
                    )
                )
                last_log = now

            if stop_flag["stop"] or (deadline is not None and now >= deadline):
                break
    finally:
        # Flush any in-flight tracks (still active at shutdown).
        for tr in tracker.flush():
            finalize(tr)
        # Final consolidated outputs, regardless of dirty flag.
        if not args.no_html:
            write_html()
        if not args.no_json:
            save_json(attrs_list, current_meta(), final_json_path)
        event_log.close()
        src.close()
        if http_server is not None:
            try:
                http_server.shutdown()
            except Exception:
                pass

    elapsed = time.monotonic() - t_start
    pipe_fps = (frame_idx + 1) / elapsed if elapsed > 0 else 0.0
    moving = sum(1 for v in attrs_list if not v.get("parked"))
    parked = sum(1 for v in attrs_list if v.get("parked"))
    print("\n[main] Loop ended.  {} frames in {:.1f}s -> {:.1f} pipe fps".format(
        frame_idx + 1, elapsed, pipe_fps,
    ))
    print("[main] Summary: {} vehicles kept ({} moving, {} parked) from {} raw tracks. "
          "HTML written {} times.".format(
        len(attrs_list), moving, parked, raw_track_count, html_writes,
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
