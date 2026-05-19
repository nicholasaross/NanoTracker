"""RtspSource — URL-redaction logic (no live camera required)."""

from __future__ import annotations

from streettracker.sources.base import FrameSource
from streettracker.sources.rtsp import RtspSource


def test_redacted_url_masks_password() -> None:
    src = RtspSource("rtsp://admin:s3cret@192.168.1.72:554/h264Preview_01_sub")
    redacted = src._redacted_url()
    assert "s3cret" not in redacted
    assert "***" in redacted
    assert "admin" in redacted
    assert "192.168.1.72" in redacted


def test_redacted_url_no_creds_returns_unchanged() -> None:
    src = RtspSource("rtsp://192.168.1.72:554/feed")
    assert src._redacted_url() == "rtsp://192.168.1.72:554/feed"


def test_rtsp_source_satisfies_protocol() -> None:
    src = RtspSource("rtsp://example/feed")
    assert isinstance(src, FrameSource)
    assert src.is_file is False
    assert src.fps == 0.0


def test_frames_without_open_raises() -> None:
    src = RtspSource("rtsp://example/feed")
    import pytest
    with pytest.raises(RuntimeError, match="not opened"):
        next(src.frames())
