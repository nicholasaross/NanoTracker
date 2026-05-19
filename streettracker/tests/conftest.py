"""Shared pytest fixtures."""

from __future__ import annotations

import pytest

from streettracker.common.schema import IRPeriod, SessionMeta, TrackRecord


@pytest.fixture
def sample_track() -> TrackRecord:
    return TrackRecord(
        track_id=42,
        class_id=2,
        class_name="car",
        time_start="2026-05-17T14:32:05+01:00",
        time_end="2026-05-17T14:32:11+01:00",
        time_start_unix=1747488725.0,
        time_end_unix=1747488731.0,
        time_start_s=12.5,
        time_end_s=18.5,
        duration_visible=6.0,
        direction="left to right",
        speed_px_s=180.4,
        color="blue",
        lane="middle",
        avg_confidence=0.876,
        displacement_px=1080.6,
        net_displacement_px=900.0,
        num_detections=58,
        asset_prefix="vehicle",
        main_snaps=[1, 2],
    )


@pytest.fixture
def sample_session_meta() -> SessionMeta:
    return SessionMeta(
        session_label="session_20260517_143000",
        session_start_unix=1747488600.0,
        frames_processed=12000,
        pipe_fps=33.4,
        avg_infer_ms=11.2,
        ir_periods=[
            IRPeriod(
                start="2026-05-17T03:00:00+01:00",
                end="2026-05-17T05:30:00+01:00",
                duration_s=9000.0,
            )
        ],
    )
