"""Shared types used across inference / sources / runtime.

Kept in `common/` so importing these doesn't pull Ultralytics / torch /
opencv — useful for analysis-only tooling that just needs to read schemas.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class Detection:
    """Single detection in original (un-letterboxed) image coordinates.

    Ported from NanoTracker's `trt_engine.Detection`. ``track_id`` is
    ``None`` when the detection comes straight from the model; populated
    once the tracker has assigned it to a track.
    """

    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    class_id: int
    track_id: int | None = None

    @property
    def cx(self) -> float:
        return 0.5 * (self.x1 + self.x2)

    @property
    def cy(self) -> float:
        return 0.5 * (self.y1 + self.y2)

    @property
    def w(self) -> float:
        return self.x2 - self.x1

    @property
    def h(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return max(0.0, self.w) * max(0.0, self.h)
