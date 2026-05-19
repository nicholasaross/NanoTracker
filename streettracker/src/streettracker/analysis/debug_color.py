"""Diagnostic inspector for the color heuristic.

For when ``vote_color`` returns something surprising on a specific crop
and you want to see the raw per-range pixel counts + unvoted HSV stats
rather than just the final label.

Ported from NanoTracker's ``scripts/debug_color.py``. The structured-data
return value is new — the original printed straight to stdout. CLI keeps
the same human-readable output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from streettracker.common.color import (
    ACHROMATIC,
    CHROMATIC_PREFER_FRAC,
    COLOR_RANGES,
    DEFAULT_PAD_FRAC,
    MIN_INNER_PIXELS,
)


@dataclass(slots=True)
class CropAnalysis:
    """Detailed breakdown of a single crop's color vote."""

    path: Path
    width: int
    height: int
    inner_width: int
    inner_height: int
    counts: dict[str, int] = field(default_factory=dict)
    total_voted: int = 0
    result: str = "unknown"
    rule: str = ""
    unvoted_hue_mean: float | None = None
    unvoted_hue_std: float | None = None
    unvoted_sat_mean: float | None = None
    unvoted_val_mean: float | None = None

    @property
    def inner_size(self) -> int:
        return self.inner_width * self.inner_height

    @property
    def coverage_pct(self) -> float:
        return 100.0 * self.total_voted / self.inner_size if self.inner_size else 0.0


def _pick_with_rule(counts: dict[str, int], total: int) -> tuple[str, str]:
    chromatic = {k: v for k, v in counts.items() if k not in ACHROMATIC}
    achromatic = {k: v for k, v in counts.items() if k in ACHROMATIC}
    best_chrom = max(chromatic.values()) if chromatic else 0
    if achromatic:
        best_ach_name = max(achromatic, key=lambda k: achromatic[k])
        if (
            best_ach_name in ("white", "black")
            and achromatic[best_ach_name] > best_chrom
        ):
            return best_ach_name, "white/black plurality"
    if chromatic:
        best_chrom_name = max(chromatic, key=lambda k: chromatic[k])
        if chromatic[best_chrom_name] >= CHROMATIC_PREFER_FRAC * total:
            return best_chrom_name, "chromatic-over-grey rule"
    return max(counts, key=lambda k: counts[k]), "fallback max"


def analyze_crop(path: Path | str, pad_frac: float = DEFAULT_PAD_FRAC) -> CropAnalysis:
    """Inspect one crop and return a populated ``CropAnalysis``.

    Returns a ``CropAnalysis`` with ``result="unknown"`` if the file
    cannot be read or the inner crop is below ``MIN_INNER_PIXELS``.
    """
    import cv2
    import numpy as np

    p = Path(path)
    img = cv2.imread(str(p))
    if img is None:
        return CropAnalysis(path=p, width=0, height=0, inner_width=0, inner_height=0)

    h, w = img.shape[:2]
    inset_x = int(w * pad_frac / (1.0 + 2.0 * pad_frac))
    inset_y = int(h * pad_frac / (1.0 + 2.0 * pad_frac))
    inner = img[
        inset_y : max(inset_y + 1, h - inset_y),
        inset_x : max(inset_x + 1, w - inset_x),
    ]
    analysis = CropAnalysis(
        path=p, width=w, height=h,
        inner_width=inner.shape[1], inner_height=inner.shape[0],
    )

    hsv = cv2.cvtColor(inner, cv2.COLOR_BGR2HSV)
    for low, high, name in COLOR_RANGES:
        mask = cv2.inRange(hsv, np.array(low), np.array(high))
        analysis.counts[name] = analysis.counts.get(name, 0) + int(cv2.countNonZero(mask))
    analysis.total_voted = sum(analysis.counts.values())

    if analysis.inner_size < MIN_INNER_PIXELS:
        analysis.result = "unknown"
        analysis.rule = f"inner crop {analysis.inner_size} < {MIN_INNER_PIXELS} min"
    elif analysis.total_voted == 0:
        analysis.result = "unknown"
        analysis.rule = "no pixels voted"
    else:
        analysis.result, analysis.rule = _pick_with_rule(
            analysis.counts, analysis.total_voted
        )

    # Unvoted-pixel HSV stats — useful for spotting whether a missing
    # plurality is due to a hue we don't have a range for.
    if analysis.total_voted < analysis.inner_size:
        all_in_any = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for low, high, _ in COLOR_RANGES:
            all_in_any = cv2.bitwise_or(
                all_in_any, cv2.inRange(hsv, np.array(low), np.array(high))
            )
        unv_mask = all_in_any == 0
        unv_hsv = hsv[unv_mask]
        if unv_hsv.shape[0] > 0:
            analysis.unvoted_hue_mean = float(unv_hsv[:, 0].mean())
            analysis.unvoted_hue_std = float(unv_hsv[:, 0].std())
            analysis.unvoted_sat_mean = float(unv_hsv[:, 1].mean())
            analysis.unvoted_val_mean = float(unv_hsv[:, 2].mean())

    return analysis


def format_report(a: CropAnalysis) -> str:
    """Render a human-readable report — same shape as the original
    ``debug_color.py`` stdout for backward compatibility."""
    lines = [
        f"\n=== {a.path} ===",
        f"  crop {a.width}x{a.height}, inner {a.inner_width}x{a.inner_height}",
        f"  voted pixels: {a.total_voted} / {a.inner_size} "
        f"({a.coverage_pct:.0f}% coverage)",
    ]
    for name, c in sorted(a.counts.items(), key=lambda x: -x[1]):
        if c > 0:
            pct = 100.0 * c / a.total_voted if a.total_voted else 0
            lines.append(f"    {name:<8} {c:>7}  ({pct:.1f}% of voted)")
    if a.result == "unknown":
        lines.append(f"  RESULT: unknown ({a.rule})")
    else:
        lines.append(f"  RESULT: {a.result} ({a.rule})")
    if a.unvoted_hue_mean is not None:
        unv = a.inner_size - a.total_voted
        lines.append(
            f"  unvoted ({unv} px): H mean={a.unvoted_hue_mean:.0f} "
            f"std={a.unvoted_hue_std:.0f}, S mean={a.unvoted_sat_mean:.0f}, "
            f"V mean={a.unvoted_val_mean:.0f}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="streettracker debug-color")
    parser.add_argument("paths", type=Path, nargs="+", help="crop JPEG(s)")
    args = parser.parse_args(argv)
    for p in args.paths:
        print(format_report(analyze_crop(p)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
