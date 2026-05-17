"""Re-vote colors on a closed session.

Reads every `vehicle_*_color.jpg` (or `_hq.jpg`) under the session dir,
runs the current vote_color heuristic, and rewrites the affected fields
in `<session>_events.jsonl`, `<session>_data.json`, `<session>_hourly.json`,
and the embedded vehicle JSON in `<session>_summary.html`.

Standalone: does NOT import nano_tracker so it works without TRT/pycuda
initialised.  Underlying crop JPEGs are not modified.

Usage:
    python3 scripts/recolor_session.py /path/to/output/session_YYYYMMDD_HHMMSS
"""

import sys
import json
import re
import datetime
from pathlib import Path

import numpy as np
import cv2


# Inlined from nano_tracker.py -- keep in sync.
COLOR_RANGES = [
    ((0, 0, 200),   (180, 30, 255),  "white"),
    ((0, 0, 0),     (180, 40, 40),   "black"),
    ((0, 0, 40),    (180, 40, 200),  "grey"),
    ((0, 60, 50),   (10, 255, 255),  "red"),
    ((170, 60, 50), (180, 255, 255), "red"),
    ((100, 80, 50), (130, 255, 255), "blue"),
    ((36, 80, 50),  (85, 255, 255),  "green"),
    ((20, 50, 180), (30, 150, 255),  "silver"),
    ((20, 80, 80),  (35, 255, 255),  "yellow"),
]
_ACHROMATIC = frozenset(("white", "black", "grey", "silver"))
_CHROMATIC_PREFER_FRAC = 0.15
_COLOR_MIN_INNER_PIXELS = 2000
_CROP_PAD_FRAC = 0.2


def vote_color(crop, pad_frac=_CROP_PAD_FRAC):
    if crop is None or crop.size == 0:
        return "unknown"
    h, w = crop.shape[:2]
    inset_x = int(w * pad_frac / (1.0 + 2.0 * pad_frac))
    inset_y = int(h * pad_frac / (1.0 + 2.0 * pad_frac))
    inner = crop[inset_y:max(inset_y + 1, h - inset_y),
                 inset_x:max(inset_x + 1, w - inset_x)]
    if inner.size == 0 or (inner.shape[0] * inner.shape[1]) < _COLOR_MIN_INNER_PIXELS:
        return "unknown"
    hsv = cv2.cvtColor(inner, cv2.COLOR_BGR2HSV)
    counts = {}
    for low, high, name in COLOR_RANGES:
        m = cv2.inRange(hsv, np.array(low), np.array(high))
        counts[name] = counts.get(name, 0) + int(cv2.countNonZero(m))
    total = sum(counts.values())
    if total == 0:
        return "unknown"
    chromatic = {k: v for k, v in counts.items() if k not in _ACHROMATIC}
    achromatic = {k: v for k, v in counts.items() if k in _ACHROMATIC}
    best_chrom_count = max(chromatic.values()) if chromatic else 0
    if achromatic:
        best_ach_name = max(achromatic, key=lambda k: achromatic[k])
        if best_ach_name in ("white", "black") and achromatic[best_ach_name] > best_chrom_count:
            return best_ach_name
    if chromatic:
        best_chrom_name = max(chromatic, key=lambda k: chromatic[k])
        if chromatic[best_chrom_name] >= _CHROMATIC_PREFER_FRAC * total:
            return best_chrom_name
    return max(counts, key=lambda k: counts[k])


def build_hourly_rollup(records, ir_periods):
    by_hour = {}
    for v in records:
        unix_ts = v.get("time_start_unix")
        if unix_ts is None:
            continue
        hour_unix = int(unix_ts // 3600) * 3600
        hour_key = datetime.datetime.fromtimestamp(hour_unix).astimezone().isoformat(timespec="hours")
        bucket = by_hour.get(hour_key)
        if bucket is None:
            bucket = {
                "hour": hour_key, "count": 0,
                "by_class": {}, "by_color": {}, "by_direction": {}, "by_lane": {},
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


def find_color_crop(session_dir, track_id):
    # Prefer the dedicated color reference (older sessions had this) over
    # the HQ crop (newer sessions consolidated the two roles into _hq.jpg).
    for suffix in ("_color.jpg", "_hq.jpg"):
        p = session_dir / "vehicle_{}{}".format(track_id, suffix)
        if p.exists():
            return p
    return None


def atomic_write(path, content):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: recolor_session.py <session_dir>")
    session_dir = Path(sys.argv[1])
    if not session_dir.is_dir():
        sys.exit("not a directory: {}".format(session_dir))

    jsonl_paths = list(session_dir.glob("*_events.jsonl"))
    data_paths = list(session_dir.glob("*_data.json"))
    hourly_paths = list(session_dir.glob("*_hourly.json"))
    html_paths = list(session_dir.glob("*_summary.html"))
    if not jsonl_paths:
        sys.exit("no *_events.jsonl in {}".format(session_dir))

    jsonl_path = jsonl_paths[0]
    print("[recolor] session: {}".format(session_dir))
    print("[recolor] reading {}".format(jsonl_path))

    records = []
    with open(str(jsonl_path), "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print("[recolor] {} vehicle records loaded".format(len(records)))

    old_dist = {}
    new_dist = {}
    changed = 0
    missing = 0
    for r in records:
        tid = r["track_id"]
        old = r.get("color", "unknown")
        old_dist[old] = old_dist.get(old, 0) + 1
        crop_path = find_color_crop(session_dir, tid)
        if crop_path is None:
            new = old
            missing += 1
        else:
            img = cv2.imread(str(crop_path))
            new = vote_color(img) if img is not None else "unknown"
        new_dist[new] = new_dist.get(new, 0) + 1
        if new != old:
            r["color"] = new
            changed += 1

    print("[recolor] {} records updated; {} missing crop file".format(changed, missing))
    print("[recolor] distribution shift:")
    all_colors = sorted(set(list(old_dist) + list(new_dist)))
    for c in all_colors:
        o = old_dist.get(c, 0)
        n = new_dist.get(c, 0)
        if o or n:
            print("  {:<10} {:>4} -> {:>4}  ({:+d})".format(c, o, n, n - o))

    # Rewrite events.jsonl
    lines = [json.dumps(r, separators=(",", ":")) for r in records]
    atomic_write(jsonl_path, "\n".join(lines) + "\n")
    print("[recolor] wrote {}".format(jsonl_path))

    # Rewrite data.json
    if data_paths:
        atomic_write(data_paths[0], json.dumps(records, indent=2))
        print("[recolor] wrote {}".format(data_paths[0]))

    # Rewrite hourly.json (preserving existing ir_periods)
    if hourly_paths:
        existing = json.loads(hourly_paths[0].read_text(encoding="utf-8"))
        ir_periods = existing.get("ir_periods", [])
        rollup = build_hourly_rollup(records, ir_periods)
        atomic_write(hourly_paths[0], json.dumps(rollup, indent=2))
        print("[recolor] wrote {}".format(hourly_paths[0]))

    # Patch the embedded JSON blob inside summary.html.
    # generate_html() embeds the row-renderer subset of fields here:
    if html_paths:
        html_path = html_paths[0]
        html = html_path.read_text(encoding="utf-8")
        embedded = [
            {
                "track_id": r["track_id"],
                "class_name": r["class_name"],
                "color": r["color"],
                "time_start": r["time_start"],
                "time_start_unix": r["time_start_unix"],
                "duration_visible": r["duration_visible"],
                "direction": r["direction"],
                "speed_px_s": r["speed_px_s"],
                "lane": r["lane"],
                "avg_confidence": r["avg_confidence"],
            }
            for r in records
        ]
        new_blob = json.dumps(embedded, separators=(",", ":"))
        pattern = r'(<script id="vehicles-data" type="application/json">)(.*?)(</script>)'
        new_html, n = re.subn(pattern, lambda m: m.group(1) + new_blob + m.group(3),
                              html, flags=re.DOTALL)
        if n == 1:
            atomic_write(html_path, new_html)
            print("[recolor] wrote {}".format(html_path))
        else:
            print("[recolor] WARN: HTML did not match expected pattern; not modified")


if __name__ == "__main__":
    main()
