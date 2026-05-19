"""EventLog crash-safe append + JSON / JSONL read-back."""

from __future__ import annotations

import json
from pathlib import Path

from streettracker.common.output import (
    EventLog,
    format_wall,
    read_data_json,
    read_events_jsonl,
    save_data_json,
    save_json,
    save_meta_json,
)
from streettracker.common.schema import SessionMeta, TrackRecord


def test_event_log_appends_one_line_per_record(
    tmp_path: Path, sample_track: TrackRecord
) -> None:
    log_path = tmp_path / "events.jsonl"
    with EventLog(log_path) as log:
        log.append(sample_track)
        log.append(sample_track)
        log.append(sample_track)
    assert log.count == 3
    lines = [
        l for l in log_path.read_text().splitlines() if l.strip()  # noqa: E741
    ]
    assert len(lines) == 3
    parsed = json.loads(lines[0])
    assert parsed["track_id"] == sample_track.track_id


def test_event_log_appends_to_existing_file(
    tmp_path: Path, sample_track: TrackRecord
) -> None:
    log_path = tmp_path / "events.jsonl"
    with EventLog(log_path) as log:
        log.append(sample_track)
    with EventLog(log_path) as log:
        log.append(sample_track)
    lines = [l for l in log_path.read_text().splitlines() if l.strip()]  # noqa: E741
    assert len(lines) == 2


def test_event_log_accepts_plain_dict(tmp_path: Path) -> None:
    log_path = tmp_path / "events.jsonl"
    with EventLog(log_path) as log:
        log.append({"track_id": 1, "class_name": "car"})
    parsed = json.loads(log_path.read_text().strip())
    assert parsed["track_id"] == 1


def test_save_data_json_round_trip(tmp_path: Path, sample_track: TrackRecord) -> None:
    p = tmp_path / "data.json"
    save_data_json([sample_track, sample_track], p)
    back = read_data_json(p)
    assert len(back) == 2
    assert back[0] == sample_track


def test_save_meta_json_round_trip(
    tmp_path: Path, sample_session_meta: SessionMeta
) -> None:
    p = tmp_path / "meta.json"
    save_meta_json(sample_session_meta, p)
    parsed = json.loads(p.read_text())
    assert parsed["session_label"] == sample_session_meta.session_label
    assert parsed["frames_processed"] == sample_session_meta.frames_processed
    assert len(parsed["ir_periods"]) == 1


def test_save_json_writes_both(
    tmp_path: Path, sample_track: TrackRecord, sample_session_meta: SessionMeta
) -> None:
    data = tmp_path / "data.json"
    meta = tmp_path / "meta.json"
    save_json([sample_track], sample_session_meta, data, meta)
    assert data.exists()
    assert meta.exists()


def test_read_events_jsonl_skips_blank_lines(
    tmp_path: Path, sample_track: TrackRecord
) -> None:
    p = tmp_path / "events.jsonl"
    with EventLog(p) as log:
        log.append(sample_track)
    # Simulate partial-write tail (a blank line)
    with open(p, "a") as f:
        f.write("\n\n")
    back = read_events_jsonl(p)
    assert len(back) == 1
    assert back[0] == sample_track


def test_format_wall_includes_tz_offset() -> None:
    s = format_wall(1747488725.0)
    # Either "+HH:MM" or "-HH:MM" or "Z" or "+HHMM"; Python's isoformat with
    # astimezone() returns "+HH:MM" form.
    assert "T" in s
    assert s[-6] in ("+", "-") or s.endswith("Z")
