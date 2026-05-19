"""vote_color() behaviour: tiny crops → unknown, solid colors → expected label."""

from __future__ import annotations

import numpy as np
import pytest

from streettracker.common.color import (
    CHROMATIC_PREFER_FRAC,
    COLOR_RANGES,
    MIN_INNER_PIXELS,
    vote_color,
)

# Skip the whole module if cv2 isn't installed (e.g. running on a minimal
# CI image without opencv-python).
cv2 = pytest.importorskip("cv2")


def _solid_bgr(height: int, width: int, bgr: tuple[int, int, int]) -> np.ndarray:
    """Make a solid-color BGR image of (height, width)."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:, :, 0] = bgr[0]
    img[:, :, 1] = bgr[1]
    img[:, :, 2] = bgr[2]
    return img


def test_none_or_empty_returns_unknown() -> None:
    assert vote_color(None) == "unknown"
    assert vote_color(np.zeros((0, 0, 3), dtype=np.uint8)) == "unknown"


def test_below_min_inner_pixels_returns_unknown() -> None:
    # Inner area must be < MIN_INNER_PIXELS after stripping pad. With
    # pad_frac=0.15 a 30x30 crop has ~22x22 = 484 inner px.
    tiny = _solid_bgr(30, 30, (0, 0, 200))  # would be red if size allowed
    assert vote_color(tiny) == "unknown"


def test_solid_white_returns_white() -> None:
    # 100x100 solid white BGR. With pad strip we still have well over
    # MIN_INNER_PIXELS pixels.
    assert MIN_INNER_PIXELS < (100 * 100)  # sanity
    img = _solid_bgr(100, 100, (255, 255, 255))
    assert vote_color(img) == "white"


def test_solid_black_returns_black() -> None:
    img = _solid_bgr(100, 100, (0, 0, 0))
    assert vote_color(img) == "black"


def test_solid_blue_returns_blue() -> None:
    # Pure-ish blue BGR (255, 80, 0) → HSV hue ~108, sat=255, val=255.
    img = _solid_bgr(100, 100, (255, 80, 0))
    assert vote_color(img) == "blue"


def test_solid_red_returns_red() -> None:
    img = _solid_bgr(100, 100, (0, 0, 255))
    assert vote_color(img) == "red"


def test_grey_with_minor_blue_picks_blue() -> None:
    # Grey background with a >=15% blue strip should pick blue, per the
    # CHROMATIC_PREFER_FRAC rule (grey/silver plurality defers to chromatic).
    img = _solid_bgr(100, 100, (128, 128, 128))      # grey body
    img[:, :25] = (255, 80, 0)                       # 25% blue strip
    assert vote_color(img) == "blue"


def test_color_ranges_table_has_all_expected_labels() -> None:
    labels = {name for _, _, name in COLOR_RANGES}
    assert {"white", "black", "grey", "red", "blue", "green", "silver", "yellow"} <= labels


def test_chromatic_prefer_frac_is_reasonable() -> None:
    # Guard against accidental drift in tuning constants.
    assert 0.05 < CHROMATIC_PREFER_FRAC < 0.5
