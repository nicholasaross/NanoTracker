"""COCO class lookup + VEHICLE_CLASSES default."""

from __future__ import annotations

from streettracker.common.coco import CLASS_NAMES, VEHICLE_CLASSES, class_name


def test_class_name_known() -> None:
    assert class_name(0) == "person"
    assert class_name(2) == "car"
    assert class_name(5) == "bus"
    assert class_name(7) == "truck"


def test_class_name_unknown_returns_unknown() -> None:
    assert class_name(99) == "unknown"
    assert class_name(-1) == "unknown"


def test_vehicle_classes_default() -> None:
    # Matches NanoTracker's historical default + VehicleTracker's COCO subset.
    assert VEHICLE_CLASSES == (2, 3, 5, 7)
    for cid in VEHICLE_CLASSES:
        assert cid in CLASS_NAMES


def test_class_names_contains_person() -> None:
    # The dashboard's "People" tab depends on this.
    assert 0 in CLASS_NAMES
    assert CLASS_NAMES[0] == "person"
