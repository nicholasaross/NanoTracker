"""build_hourly_rollup() bucketing + breakdown counts."""

from __future__ import annotations

from dataclasses import replace

from streettracker.common.hourly import build_hourly_rollup
from streettracker.common.schema import IRPeriod, TrackRecord


def test_empty_input_yields_empty_hours(sample_track: TrackRecord) -> None:
    out = build_hourly_rollup([], [])
    assert out["hours"] == []
    assert out["ir_periods"] == []


def test_single_record_creates_one_bucket(sample_track: TrackRecord) -> None:
    out = build_hourly_rollup([sample_track], [])
    assert len(out["hours"]) == 1
    bucket = out["hours"][0]
    assert bucket["count"] == 1
    assert bucket["by_class"] == {"car": 1}
    assert bucket["by_color"] == {"blue": 1}
    assert bucket["by_direction"] == {"left to right": 1}
    assert bucket["by_lane"] == {"middle": 1}


def test_two_records_same_hour_share_bucket(sample_track: TrackRecord) -> None:
    second = replace(
        sample_track,
        track_id=43,
        time_start_unix=sample_track.time_start_unix + 60,  # same hour
        color="red",
    )
    out = build_hourly_rollup([sample_track, second], [])
    assert len(out["hours"]) == 1
    bucket = out["hours"][0]
    assert bucket["count"] == 2
    assert bucket["by_color"] == {"blue": 1, "red": 1}


def test_two_records_different_hours_get_two_buckets(
    sample_track: TrackRecord,
) -> None:
    later = replace(
        sample_track,
        track_id=44,
        time_start_unix=sample_track.time_start_unix + 3600,  # next hour
    )
    out = build_hourly_rollup([sample_track, later], [])
    assert len(out["hours"]) == 2
    # hours are sorted ascending by ISO key
    assert out["hours"][0]["hour"] < out["hours"][1]["hour"]


def test_ir_periods_passed_through() -> None:
    ir = [
        IRPeriod(
            start="2026-05-17T03:00:00+01:00",
            end="2026-05-17T05:00:00+01:00",
            duration_s=7200.0,
        )
    ]
    out = build_hourly_rollup([], ir)
    assert len(out["ir_periods"]) == 1
    assert out["ir_periods"][0]["duration_s"] == 7200.0


def test_plain_dict_records_work(sample_track: TrackRecord) -> None:
    # Records may come in as plain dicts when read from disk.
    out = build_hourly_rollup([sample_track.to_json_dict()], [])
    assert len(out["hours"]) == 1
    assert out["hours"][0]["by_class"] == {"car": 1}
