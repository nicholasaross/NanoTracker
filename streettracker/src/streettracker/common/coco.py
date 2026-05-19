"""COCO class IDs and names used by StreetTracker.

Subset of the 80-class COCO list — we don't need them all, and listing only
the classes we actually surface keeps the dashboard's "Type" column tidy.
Extend `CLASS_NAMES` if you turn on more classes via camera_config's
inference.class_filter.
"""

from __future__ import annotations

# Default class filter for vehicle tracking: car, motorcycle, bus, truck.
# Matches VehicleTracker's default. Persons (class 0) are usually added at
# the deployment level via camera_config.
VEHICLE_CLASSES: tuple[int, ...] = (2, 3, 5, 7)

# Full name map for every COCO class we expect to enable in practice.
# Classes not listed render as "unknown" in dashboards / JSON.
CLASS_NAMES: dict[int, str] = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    15: "cat",
    16: "dog",
}


def class_name(class_id: int) -> str:
    """Return the human-readable name for a class id, or ``"unknown"``."""
    return CLASS_NAMES.get(class_id, "unknown")
