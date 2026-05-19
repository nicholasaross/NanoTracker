"""Detection: properties + equality + tracker-id default."""

from __future__ import annotations

from streettracker.common.types import Detection


def test_detection_default_track_id_is_none() -> None:
    d = Detection(x1=0, y1=0, x2=10, y2=20, score=0.5, class_id=2)
    assert d.track_id is None


def test_detection_properties() -> None:
    d = Detection(x1=10, y1=20, x2=30, y2=60, score=0.5, class_id=2)
    assert d.w == 20
    assert d.h == 40
    assert d.cx == 20.0
    assert d.cy == 40.0
    assert d.area == 800.0


def test_degenerate_box_has_zero_area() -> None:
    # x2 < x1: clamped to zero, not negative
    d = Detection(x1=30, y1=20, x2=10, y2=60, score=0.5, class_id=2)
    assert d.area == 0.0


def test_detection_is_hashable_and_frozen() -> None:
    d1 = Detection(x1=0, y1=0, x2=10, y2=20, score=0.5, class_id=2, track_id=7)
    d2 = Detection(x1=0, y1=0, x2=10, y2=20, score=0.5, class_id=2, track_id=7)
    assert d1 == d2
    assert hash(d1) == hash(d2)
    assert {d1, d2} == {d1}
