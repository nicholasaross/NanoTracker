"""Rerun the color heuristic over a closed session and rewrite outputs.

Ported from NanoTracker's ``scripts/recolor_session.py``. The original
script inlined `vote_color` + `build_hourly_rollup` to avoid pulling
nano_tracker -> trt_engine -> pycuda; in StreetTracker that machinery is
already factored into ``common/`` so this analyser is just a thin
re-driver.

What gets rewritten in the session dir (atomic writes throughout):

  - ``{session}_events.jsonl`` — every line's ``color`` field updated.
  - ``{session}_data.json`` — array of records rewritten.
  - ``{session}_hourly.json`` — rollup recomputed (ir_periods preserved).
  - ``{session}_summary.html`` — embedded JSON blob inside the
    ``<script id="vehicles-data">`` element rewritten.

Underlying crop JPEGs are not modified — only the recomputed labels are.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from streettracker.common.color import vote_color
from streettracker.common.hourly import build_hourly_rollup

# Suffixes searched for the color-reference crop, in priority order.
# Older sessions had a dedicated `_color.jpg`; newer ones consolidated
# the role into `_hq.jpg`. We accept either.
_COLOR_CROP_SUFFIXES: tuple[str, ...] = ("_color.jpg", "_hq.jpg")

# Both class prefixes used by StreetTracker for asset filenames.
_ASSET_PREFIXES: tuple[str, ...] = ("vehicle", "person")


@dataclass(slots=True)
class RecolorStats:
    """Result of a recolor pass — what changed, what was missing."""

    records_total: int = 0
    records_changed: int = 0
    records_missing_crop: int = 0
    distribution_before: dict[str, int] = field(default_factory=dict)
    distribution_after: dict[str, int] = field(default_factory=dict)


def find_color_crop(session_dir: Path, track_id: int) -> Path | None:
    """Locate the JPEG to vote on for ``track_id``.

    Searches both class prefixes (``vehicle_``, ``person_``) and both
    legacy suffixes. Returns the first match or ``None``.
    """
    for prefix in _ASSET_PREFIXES:
        for suffix in _COLOR_CROP_SUFFIXES:
            p = session_dir / f"{prefix}_{track_id}{suffix}"
            if p.exists():
                return p
    return None


def _atomic_write(path: Path, content: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)


_EMBEDDED_JSON_RE = re.compile(
    r'(<script id="vehicles-data" type="application/json">)(.*?)(</script>)',
    re.DOTALL,
)


def _embedded_row(r: dict[str, Any]) -> dict[str, Any]:
    """Subset of fields embedded in the summary HTML's ``vehicles-data``
    script tag — matches ``common.summary.generate_html``.
    """
    return {
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
        "asset_prefix": r.get("asset_prefix", "vehicle"),
        "main_snaps": list(r.get("main_snaps", [])),
    }


def recolor_session(session_dir: Path) -> RecolorStats:
    """Reread crops, re-vote colors, rewrite session outputs.

    Returns a ``RecolorStats`` summary for caller-side logging.

    Raises ``FileNotFoundError`` if no ``*_events.jsonl`` is present.
    """
    import cv2  # deferred — analysis hosts without cv2 can still import this module

    jsonl_paths = sorted(session_dir.glob("*_events.jsonl"))
    if not jsonl_paths:
        raise FileNotFoundError(f"No *_events.jsonl in {session_dir}")
    jsonl_path = jsonl_paths[0]
    data_paths = sorted(session_dir.glob("*_data.json"))
    hourly_paths = sorted(session_dir.glob("*_hourly.json"))
    html_paths = sorted(session_dir.glob("*_summary.html"))

    records: list[dict[str, Any]] = []
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))

    stats = RecolorStats(records_total=len(records))

    for r in records:
        old = r.get("color", "unknown")
        stats.distribution_before[old] = stats.distribution_before.get(old, 0) + 1
        crop_path = find_color_crop(session_dir, int(r["track_id"]))
        if crop_path is None:
            new = old
            stats.records_missing_crop += 1
        else:
            img = cv2.imread(str(crop_path))
            new = vote_color(img) if img is not None else "unknown"
        stats.distribution_after[new] = stats.distribution_after.get(new, 0) + 1
        if new != old:
            r["color"] = new
            stats.records_changed += 1

    # Rewrite events.jsonl
    lines = [json.dumps(r, separators=(",", ":")) for r in records]
    _atomic_write(jsonl_path, "\n".join(lines) + ("\n" if lines else ""))

    # Rewrite data.json
    if data_paths:
        _atomic_write(data_paths[0], json.dumps(records, indent=2))

    # Rewrite hourly.json — preserve existing ir_periods, recompute breakdowns.
    if hourly_paths:
        existing = json.loads(hourly_paths[0].read_text(encoding="utf-8"))
        ir_periods = existing.get("ir_periods", [])
        rollup = build_hourly_rollup(records, ir_periods)
        _atomic_write(hourly_paths[0], json.dumps(rollup, indent=2))

    # Patch embedded JSON inside summary.html.
    if html_paths:
        html_path = html_paths[0]
        html = html_path.read_text(encoding="utf-8")
        new_blob = json.dumps([_embedded_row(r) for r in records], separators=(",", ":"))
        new_html, n = _EMBEDDED_JSON_RE.subn(
            lambda m: m.group(1) + new_blob + m.group(3), html
        )
        if n == 1:
            _atomic_write(html_path, new_html)
        # else: leave HTML untouched; the summary will catch up on next
        # idle regen by the live runtime (or on the next recolor pass).

    return stats


def main(argv: list[str] | None = None) -> int:
    """CLI entry: ``streettracker recolor <session_dir>``."""
    import argparse

    parser = argparse.ArgumentParser(prog="streettracker recolor")
    parser.add_argument("session_dir", type=Path, help="closed session directory")
    args = parser.parse_args(argv)

    if not args.session_dir.is_dir():
        print(f"[recolor] not a directory: {args.session_dir}")
        return 1

    print(f"[recolor] session: {args.session_dir}")
    stats = recolor_session(args.session_dir)
    print(
        f"[recolor] {stats.records_changed}/{stats.records_total} records updated; "
        f"{stats.records_missing_crop} missing crop file"
    )
    print("[recolor] distribution shift:")
    all_colors = sorted(set(stats.distribution_before) | set(stats.distribution_after))
    for c in all_colors:
        before = stats.distribution_before.get(c, 0)
        after = stats.distribution_after.get(c, 0)
        if before or after:
            print(f"  {c:<10} {before:>4} -> {after:>4}  ({after - before:+d})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
