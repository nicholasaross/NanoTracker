"""Shared types and helpers for ALPR pipelines.

Pipelines are Protocol-conforming: name + run(image_path, track_id, snap_index, class_name).
The runner (alpr.cli.run) treats them uniformly so adding a third pipeline is one file.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np


SNAP_FILENAME_RE = re.compile(
    r"^(?P<cls>person|vehicle)_(?P<tid>\d+)_main_(?P<n>\d+)\.jpg$"
)


@dataclass
class PlateDetection:
    bbox: tuple[int, int, int, int]
    det_confidence: float


@dataclass
class PlateRead:
    text: str
    ocr_confidence: float
    raw_text: str


@dataclass
class PlateResult:
    image_path: str
    image_name: str
    track_id: int
    snap_index: int
    class_name: str
    pipeline: str
    detection: PlateDetection | None
    read: PlateRead | None
    crop_path: str | None
    pipeline_ms: float
    error: str | None = None

    def to_json(self) -> dict:
        return {
            "image": self.image_name,
            "image_path": self.image_path,
            "track_id": self.track_id,
            "snap_index": self.snap_index,
            "class_name": self.class_name,
            "pipeline": self.pipeline,
            "det_bbox": list(self.detection.bbox) if self.detection else None,
            "det_conf": self.detection.det_confidence if self.detection else None,
            "ocr_text": self.read.text if self.read else None,
            "ocr_raw": self.read.raw_text if self.read else None,
            "ocr_conf": self.read.ocr_confidence if self.read else None,
            "crop_path": self.crop_path,
            "pipeline_ms": self.pipeline_ms,
            "error": self.error,
        }


@runtime_checkable
class Pipeline(Protocol):
    name: str
    def run(
        self,
        image_path: Path,
        track_id: int,
        snap_index: int,
        class_name: str,
        crop_out_dir: Path,
    ) -> PlateResult: ...


def parse_snap_filename(name: str) -> tuple[str, int, int] | None:
    """Return (class_name, track_id, snap_index) for a _main_*.jpg filename, or None."""
    m = SNAP_FILENAME_RE.match(name)
    if not m:
        return None
    return m.group("cls"), int(m.group("tid")), int(m.group("n"))


_ALNUM_RE = re.compile(r"[^A-Z0-9]")
_LABEL_CHARS_RE = re.compile(r"[^A-Z0-9.]")

# A '.' in a label is a single-character wildcard (matches any one OCR char) — useful
# when the human labeller can read some characters of the plate but not all.
LABEL_WILDCARD = "."


def normalize_plate_text(raw: str) -> str:
    """User-visible canonical form: uppercase, alnum only.  Preserves I/O/1/0 distinctions."""
    if not raw:
        return ""
    return _ALNUM_RE.sub("", raw.upper())


def normalize_label_text(raw: str) -> str:
    """Like normalize_plate_text, but keeps '.' as a single-char wildcard for labels."""
    if not raw:
        return ""
    return _LABEL_CHARS_RE.sub("", raw.upper())


# OCR character confusions worth collapsing for *scoring only* (not user-visible text).
# Direction is "ambiguous char -> canonical".  Both directions get the same canonical form
# so 'O' and '0' compare equal; we pick the digit/letter that's more common on plates.
_SCORING_CANON_MAP = str.maketrans({
    "O": "0",
    "I": "1",
    "Z": "2",
    "S": "5",
    "B": "8",
})


def canonical_for_scoring(text: str) -> str:
    """Aggressive normalization used only by the score CLI.

    Collapses common OCR confusions (O↔0, I↔1, Z↔2, S↔5, B↔8) so a real plate
    'B8K2' and an OCR output '8BK2' both canonicalize to '8825'-ish forms that
    Levenshtein-compare more fairly.  Do NOT use this for the user-facing plate text.
    """
    return normalize_plate_text(text).translate(_SCORING_CANON_MAP)


def canonical_translate(text: str) -> str:
    """Apply OCR-confusion collapse without stripping non-alnum.  Use for labels that
    may contain '.' wildcards — preserves them for downstream wildcard matching."""
    return text.translate(_SCORING_CANON_MAP)


def crop_with_padding(
    image: np.ndarray,
    bbox: tuple[int, int, int, int],
    pad_frac: float = 0.10,
) -> np.ndarray:
    """Crop image to bbox, padded by pad_frac of bbox side length, clamped to image bounds."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)
    px = int(round(bw * pad_frac))
    py = int(round(bh * pad_frac))
    cx1 = max(0, x1 - px)
    cy1 = max(0, y1 - py)
    cx2 = min(w, x2 + px)
    cy2 = min(h, y2 + py)
    return image[cy1:cy2, cx1:cx2]


def atomic_write_text(path: Path, content: str) -> None:
    """Write text via temp-file + replace, matching scripts/recolor_session.py:111-114."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)


def atomic_write_bytes(path: Path, content: bytes) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(content)
    tmp.replace(path)


class Timer:
    """Wallclock context manager: with Timer() as t: ...; t.ms"""
    def __enter__(self) -> "Timer":
        self._t0 = time.perf_counter()
        self.ms = 0.0
        return self

    def __exit__(self, *exc) -> None:
        self.ms = (time.perf_counter() - self._t0) * 1000.0
