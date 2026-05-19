"""analyze_crop() returns structured stats matching the heuristic's behaviour."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from streettracker.analysis.debug_color import analyze_crop, format_report

cv2 = pytest.importorskip("cv2")


def _write_solid(path: Path, h: int, w: int, bgr: tuple[int, int, int]) -> None:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :, 0] = bgr[0]
    img[:, :, 1] = bgr[1]
    img[:, :, 2] = bgr[2]
    cv2.imwrite(str(path), img)


def test_analyze_solid_red(tmp_path: Path) -> None:
    p = tmp_path / "red.jpg"
    _write_solid(p, 100, 100, (0, 0, 255))
    a = analyze_crop(p)
    assert a.result == "red"
    assert a.counts.get("red", 0) > 0
    assert a.total_voted > 0


def test_analyze_solid_white(tmp_path: Path) -> None:
    p = tmp_path / "white.jpg"
    _write_solid(p, 100, 100, (255, 255, 255))
    a = analyze_crop(p)
    assert a.result == "white"
    assert "plurality" in a.rule


def test_analyze_tiny_crop_returns_unknown(tmp_path: Path) -> None:
    p = tmp_path / "tiny.jpg"
    _write_solid(p, 30, 30, (0, 0, 255))   # too small to vote
    a = analyze_crop(p)
    assert a.result == "unknown"
    assert "min" in a.rule


def test_analyze_returns_unknown_for_missing_file(tmp_path: Path) -> None:
    a = analyze_crop(tmp_path / "nonexistent.jpg")
    assert a.result == "unknown"
    assert a.width == 0
    assert a.height == 0


def test_format_report_is_human_readable(tmp_path: Path) -> None:
    p = tmp_path / "red.jpg"
    _write_solid(p, 100, 100, (0, 0, 255))
    a = analyze_crop(p)
    report = format_report(a)
    assert str(p) in report
    assert "RESULT" in report
    assert "red" in report.lower()


def test_analyze_records_unvoted_hsv_stats(tmp_path: Path) -> None:
    """A magenta crop sits outside every COLOR_RANGES bucket; we should
    capture its unvoted HSV stats so the user can spot the hole."""
    p = tmp_path / "magenta.jpg"
    _write_solid(p, 100, 100, (255, 0, 255))
    a = analyze_crop(p)
    # No range covers pure magenta, so unvoted stats should be populated.
    assert a.unvoted_hue_mean is not None
    assert a.unvoted_sat_mean is not None
    assert a.unvoted_val_mean is not None
