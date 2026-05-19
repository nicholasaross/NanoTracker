"""Track record + session metadata dataclasses.

These are the canonical schemas — used by the device runtime when writing
session output, and by analysis tools when reading it. Field set ported
verbatim from NanoTracker's `compute_attributes()` return dict so existing
sessions remain parseable.

Records are JSON-serialised via `to_json_dict()` / `from_json_dict()` to
keep the wire format stable across Python version bumps. We don't pickle.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

# ----------------------------------------------------------------------
# Track record (one per finalized track)
# ----------------------------------------------------------------------

@dataclass(slots=True)
class TrackRecord:
    """One finalized track. Written as a single JSONL line and folded into
    the session's `_data.json` array at finalize time.

    Fields:
        identity:  track_id, class_id, class_name, asset_prefix
        time:      time_start[/end][/_unix][/_s], duration_visible
        motion:    direction, speed_px_s, displacement_px, net_displacement_px, lane
        detection: avg_confidence, num_detections
        attrs:     color
        assets:    main_snaps (list of int N values whose _main_N.jpg landed)
    """

    track_id: int
    class_id: int
    class_name: str
    time_start: str           # ISO-local with tz offset
    time_end: str
    time_start_unix: float
    time_end_unix: float
    time_start_s: float       # seconds since session start
    time_end_s: float
    duration_visible: float
    direction: str            # "left to right" | "right to left"
    speed_px_s: float
    color: str                # see common.color.vote_color()
    lane: str                 # "top" | "middle" | "bottom"
    avg_confidence: float
    displacement_px: float
    net_displacement_px: float
    num_detections: int
    asset_prefix: str = "vehicle"   # "vehicle" | "person"
    main_snaps: list[int] = field(default_factory=list)

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json_dict(cls, d: dict[str, Any]) -> TrackRecord:
        # Tolerate extra fields (forward compatibility with future analysers
        # that add columns). Drop unknown keys silently.
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})


# ----------------------------------------------------------------------
# Session metadata
# ----------------------------------------------------------------------

@dataclass(slots=True)
class IRPeriod:
    """A stretch of frames in which IR/night mode was active and inference
    was paused. Persisted so analysis can tell 'no traffic this hour' apart
    from 'we were asleep'."""

    start: str          # ISO local
    end: str
    duration_s: float

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SessionMeta:
    session_label: str
    session_start_unix: float
    frames_processed: int = 0
    pipe_fps: float = 0.0
    avg_infer_ms: float = 0.0
    ir_periods: list[IRPeriod] = field(default_factory=list)

    def to_json_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # asdict() recurses dataclasses; ir_periods is already in dict form.
        return d

    @classmethod
    def from_json_dict(cls, d: dict[str, Any]) -> SessionMeta:
        ir = [IRPeriod(**p) for p in d.get("ir_periods", [])]
        return cls(
            session_label=d["session_label"],
            session_start_unix=d["session_start_unix"],
            frames_processed=d.get("frames_processed", 0),
            pipe_fps=d.get("pipe_fps", 0.0),
            avg_infer_ms=d.get("avg_infer_ms", 0.0),
            ir_periods=ir,
        )


# ----------------------------------------------------------------------
# Asset-prefix helper (matches NanoTracker's _class_asset_prefix)
# ----------------------------------------------------------------------

_PERSON_CLASS_ID = 0  # COCO class 0 = person; everything else maps to vehicle.


def asset_prefix_for_class(class_id: int) -> str:
    return "person" if class_id == _PERSON_CLASS_ID else "vehicle"
