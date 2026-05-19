"""generate_html() smoke test: writes a parseable page + sidecar JSON."""

from __future__ import annotations

import json
from pathlib import Path

from streettracker.common.schema import TrackRecord
from streettracker.common.summary import generate_html


def test_generate_html_writes_page_and_sidecar(
    tmp_path: Path, sample_track: TrackRecord
) -> None:
    html_path = tmp_path / "session_summary.html"
    generate_html(
        records=[sample_track, sample_track],
        output_dir=tmp_path,
        html_path=html_path,
        session_label="session_test",
        meta={"frames_processed": 12000, "pipe_fps": 33.4},
        refresh_seconds=15,
    )
    assert html_path.exists()
    body = html_path.read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in body
    assert "StreetTracker Summary" in body
    assert "session_test" in body
    # The embedded JSON should contain the track ID.
    assert '"track_id":42' in body

    sidecar = tmp_path / "vehicles.json"
    assert sidecar.exists()
    payload = json.loads(sidecar.read_text())
    assert isinstance(payload, list)
    assert len(payload) == 2
    assert payload[0]["track_id"] == 42


def test_index_html_redirects_to_latest_summary(
    tmp_path: Path, sample_track: TrackRecord
) -> None:
    html_path = tmp_path / "session_summary.html"
    generate_html(
        records=[sample_track],
        output_dir=tmp_path,
        html_path=html_path,
        session_label="session_test",
        meta={},
    )
    index = tmp_path / "index.html"
    assert index.exists()
    assert html_path.name in index.read_text()
    assert "http-equiv=\"refresh\"" in index.read_text()


def test_empty_records_renders_no_detections_yet(tmp_path: Path) -> None:
    html_path = tmp_path / "session_summary.html"
    generate_html(
        records=[],
        output_dir=tmp_path,
        html_path=html_path,
        session_label="empty_session",
        meta={},
    )
    body = html_path.read_text()
    assert "No detections yet" in body
    sidecar = json.loads((tmp_path / "vehicles.json").read_text())
    assert sidecar == []


def test_poll_disabled_when_refresh_zero(
    tmp_path: Path, sample_track: TrackRecord
) -> None:
    html_path = tmp_path / "session_summary.html"
    generate_html(
        records=[sample_track],
        output_dir=tmp_path,
        html_path=html_path,
        session_label="t",
        meta={},
        refresh_seconds=0,
    )
    body = html_path.read_text()
    assert "POLL_MS=0;" in body
