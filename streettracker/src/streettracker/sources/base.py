"""FrameSource protocol — the shape every video source obeys.

A frame source yields ``(frame_index, t_seconds, BGR_frame)`` tuples. For
live sources (RTSP) ``t`` is wall-clock seconds since the first yielded
frame; for file sources it is video-time (``frame_idx / fps``), so track
durations / speeds stay meaningful even when inference runs slower or
faster than realtime.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import numpy as np


@runtime_checkable
class FrameSource(Protocol):
    """Minimal interface for video / RTSP sources used by StreetTracker.

    Implementations: ``streettracker.sources.rtsp.RtspSource`` (live RTSP
    via OpenCV's FFmpeg backend) and ``streettracker.sources.file.FileSource``
    (MP4 via GStreamer + NVDEC on Orin, FFmpeg software fallback elsewhere).
    """

    is_file: bool
    fps: float

    def open(self) -> None:
        """Connect / open the underlying stream. Raises on failure."""

    def frames(self) -> Iterator[tuple[int, float, np.ndarray]]:
        """Yield ``(frame_index, t_seconds, BGR_frame)`` until the source ends."""

    def close(self) -> None:
        """Release the underlying handle. Idempotent."""
