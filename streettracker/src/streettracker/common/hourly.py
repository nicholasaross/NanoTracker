"""Per-hour rollup of finalized tracks.

Ported from NanoTracker's `build_hourly_rollup()` (nano_tracker.py ~line
904). Output shape preserved so existing `_hourly.json` consumers keep
working.
"""

from __future__ import annotations

import datetime
from typing import Any

from streettracker.common.schema import IRPeriod, TrackRecord


def build_hourly_rollup(
    records: list[TrackRecord | dict[str, Any]],
    ir_periods: list[IRPeriod | dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Bucket records by wall-clock hour; summarise counts per hour.

    Returns ``{"hours": [...], "ir_periods": [...]}`` where each hour entry
    has the local-hour ISO key plus breakdowns by class / color / direction
    / lane. IR periods are included alongside so "no traffic this hour"
    can be distinguished from "we were asleep".
    """
    by_hour: dict[str, dict[str, Any]] = {}

    for r in records:
        d = r.to_json_dict() if isinstance(r, TrackRecord) else r
        unix_ts = d.get("time_start_unix")
        if unix_ts is None:
            continue
        hour_unix = int(unix_ts // 3600) * 3600
        hour_key = (
            datetime.datetime.fromtimestamp(hour_unix)
            .astimezone()
            .isoformat(timespec="hours")
        )
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
        for field_key, record_key in (
            ("by_class", "class_name"),
            ("by_color", "color"),
            ("by_direction", "direction"),
            ("by_lane", "lane"),
        ):
            val = d.get(record_key, "unknown")
            bucket[field_key][val] = bucket[field_key].get(val, 0) + 1

    ir_out: list[dict[str, Any]] = []
    for p in ir_periods or []:
        ir_out.append(p.to_json_dict() if isinstance(p, IRPeriod) else p)

    return {
        "hours": sorted(by_hour.values(), key=lambda b: b["hour"]),
        "ir_periods": ir_out,
    }
