"""HSV-vote color classifier for vehicle crops.

Ported verbatim from NanoTracker's `vote_color()` (nano_tracker.py line ~378).
Tuning that survives:

  - 9 HSV ranges (white/black/grey/red×2/blue/green/silver/yellow).
  - Strip ``pad_frac`` padding before counting — otherwise road grey
    drowns out paint.
  - Return "unknown" when the inner crop is under 2000 px (sub-bbox
    color votes are noise).
  - White/black plurality wins outright over chromatic; grey/silver
    plurality defers to a chromatic if it's >=15% of total votes.

cv2 import is deferred to call-time to keep this module cheap to import on
machines without opencv-python (e.g. tooling that just reads schemas).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


# (low, high, name) — HSV ranges. Order matters only for readability; the
# vote loop sums hits across ranges that share a name (e.g. red wraps 0).
COLOR_RANGES: list[tuple[tuple[int, int, int], tuple[int, int, int], str]] = [
    ((0, 0, 200),   (180, 30, 255),  "white"),
    ((0, 0, 0),     (180, 40, 40),   "black"),
    ((0, 0, 40),    (180, 40, 200),  "grey"),
    ((0, 60, 50),   (10, 255, 255),  "red"),
    ((170, 60, 50), (180, 255, 255), "red"),
    ((100, 80, 50), (130, 255, 255), "blue"),
    ((36, 80, 50),  (85, 255, 255),  "green"),
    ((20, 50, 180), (30, 150, 255),  "silver"),
    ((20, 80, 80),  (35, 255, 255),  "yellow"),
]

ACHROMATIC: frozenset[str] = frozenset(("white", "black", "grey", "silver"))

# When grey is the achromatic plurality, a chromatic >= this fraction of voted
# pixels wins (catches "real color buried under road-grey background").
CHROMATIC_PREFER_FRAC = 0.15

# Below this inner-bbox pixel count the vote is too noisy to be useful.
MIN_INNER_PIXELS = 2000

# Default crop padding (must match the `_safe_crop(..., pad_frac=)` used to
# produce the crop). NanoTracker's default is 0.15.
DEFAULT_PAD_FRAC = 0.15


def vote_color(crop: np.ndarray | None, pad_frac: float = DEFAULT_PAD_FRAC) -> str:
    """Pick a color label from a padded BGR crop.

    Returns one of the names in ``COLOR_RANGES`` or ``"unknown"``.

    The crop is assumed to be a `_safe_crop(..., pad_frac=p)` output:
    padding grows the bbox by ``p`` on each side, so the original bbox sits
    centered with ``p / (1 + 2p)`` inset on each side. We strip that inset
    before counting.
    """
    import cv2
    import numpy as np

    if crop is None or crop.size == 0:
        return "unknown"

    h, w = crop.shape[:2]
    inset_x = int(w * pad_frac / (1.0 + 2.0 * pad_frac))
    inset_y = int(h * pad_frac / (1.0 + 2.0 * pad_frac))
    inner = crop[
        inset_y : max(inset_y + 1, h - inset_y),
        inset_x : max(inset_x + 1, w - inset_x),
    ]
    if inner.size == 0 or (inner.shape[0] * inner.shape[1]) < MIN_INNER_PIXELS:
        return "unknown"

    hsv = cv2.cvtColor(inner, cv2.COLOR_BGR2HSV)
    counts: dict[str, int] = {}
    for low, high, name in COLOR_RANGES:
        mask = cv2.inRange(hsv, np.array(low), np.array(high))
        counts[name] = counts.get(name, 0) + int(cv2.countNonZero(mask))
    total = sum(counts.values())
    if total == 0:
        return "unknown"

    chromatic = {k: v for k, v in counts.items() if k not in ACHROMATIC}
    achromatic = {k: v for k, v in counts.items() if k in ACHROMATIC}
    best_chrom_count = max(chromatic.values()) if chromatic else 0

    # White/black plurality wins outright over any chromatic.
    if achromatic:
        best_ach_name = max(achromatic, key=lambda k: achromatic[k])
        if (
            best_ach_name in ("white", "black")
            and achromatic[best_ach_name] > best_chrom_count
        ):
            return best_ach_name

    # Grey/silver plurality: defer to dominant chromatic if substantive.
    if chromatic:
        best_chrom_name = max(chromatic, key=lambda k: chromatic[k])
        if chromatic[best_chrom_name] >= CHROMATIC_PREFER_FRAC * total:
            return best_chrom_name

    return max(counts, key=lambda k: counts[k])
