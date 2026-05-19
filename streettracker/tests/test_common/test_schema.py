"""TrackRecord / SessionMeta round-trip + asset-prefix mapping."""

from __future__ import annotations

import json

from streettracker.common.schema import (
    IRPeriod,
    SessionMeta,
    TrackRecord,
    asset_prefix_for_class,
)


def test_track_record_round_trip(sample_track: TrackRecord) -> None:
    d = sample_track.to_json_dict()
    text = json.dumps(d)
    parsed = json.loads(text)
    back = TrackRecord.from_json_dict(parsed)
    assert back == sample_track


def test_track_record_tolerates_extra_fields(sample_track: TrackRecord) -> None:
    d = sample_track.to_json_dict()
    d["plate_text"] = "ABC123"  # hypothetical future ALPR column
    d["make_model"] = "Honda Civic"
    back = TrackRecord.from_json_dict(d)
    assert back == sample_track  # unknown fields silently dropped


def test_track_record_main_snaps_default() -> None:
    r = TrackRecord(
        track_id=1, class_id=2, class_name="car",
        time_start="t", time_end="t",
        time_start_unix=0.0, time_end_unix=0.0,
        time_start_s=0.0, time_end_s=0.0,
        duration_visible=0.0,
        direction="left to right", speed_px_s=0.0,
        color="unknown", lane="middle",
        avg_confidence=0.0, displacement_px=0.0,
        net_displacement_px=0.0, num_detections=0,
    )
    assert r.main_snaps == []
    assert r.asset_prefix == "vehicle"


def test_session_meta_round_trip(sample_session_meta: SessionMeta) -> None:
    d = sample_session_meta.to_json_dict()
    back = SessionMeta.from_json_dict(json.loads(json.dumps(d)))
    assert back.session_label == sample_session_meta.session_label
    assert back.frames_processed == sample_session_meta.frames_processed
    assert len(back.ir_periods) == 1
    assert isinstance(back.ir_periods[0], IRPeriod)
    assert back.ir_periods[0].duration_s == 9000.0


def test_asset_prefix_for_class() -> None:
    assert asset_prefix_for_class(0) == "person"
    assert asset_prefix_for_class(2) == "vehicle"   # car
    assert asset_prefix_for_class(5) == "vehicle"   # bus
    assert asset_prefix_for_class(7) == "vehicle"   # truck
