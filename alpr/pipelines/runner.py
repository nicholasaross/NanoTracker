"""PipelineRunner composes a Detector + Recognizer into a uniform Pipeline.

This is the seam that lets the ablation config (bespoke detector + fast-plate-ocr OCR)
exist as a one-liner without duplicating the load / crop / save / time scaffolding.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from alpr.pipelines.base import (
    PlateDetection,
    PlateRead,
    PlateResult,
    Timer,
    atomic_write_bytes,
    crop_with_padding,
)


class Detector(Protocol):
    name: str
    def detect(self, image: np.ndarray) -> PlateDetection | None: ...


class Recognizer(Protocol):
    name: str
    def recognize(self, crop_bgr: np.ndarray) -> PlateRead | None: ...


class PipelineRunner:
    def __init__(self, name: str, detector: Detector, recognizer: Recognizer) -> None:
        self.name = name
        self._detector = detector
        self._recognizer = recognizer

    def run(
        self,
        image_path: Path,
        track_id: int,
        snap_index: int,
        class_name: str,
        crop_out_dir: Path,
    ) -> PlateResult:
        image_name = image_path.name
        with Timer() as t:
            try:
                image = cv2.imread(str(image_path))
                if image is None:
                    return PlateResult(
                        image_path=str(image_path), image_name=image_name,
                        track_id=track_id, snap_index=snap_index, class_name=class_name,
                        pipeline=self.name, detection=None, read=None, crop_path=None,
                        pipeline_ms=0.0, error="imread_failed",
                    )

                detection = self._detector.detect(image)
                if detection is None:
                    return PlateResult(
                        image_path=str(image_path), image_name=image_name,
                        track_id=track_id, snap_index=snap_index, class_name=class_name,
                        pipeline=self.name, detection=None, read=None, crop_path=None,
                        pipeline_ms=t.ms, error=None,
                    )

                crop = crop_with_padding(image, detection.bbox, pad_frac=0.10)
                crop_path = _save_crop(crop, crop_out_dir, image_name)
                read = self._recognizer.recognize(crop)
            except Exception as e:
                return PlateResult(
                    image_path=str(image_path), image_name=image_name,
                    track_id=track_id, snap_index=snap_index, class_name=class_name,
                    pipeline=self.name, detection=None, read=None, crop_path=None,
                    pipeline_ms=t.ms, error=f"{type(e).__name__}: {e}",
                )

        return PlateResult(
            image_path=str(image_path), image_name=image_name,
            track_id=track_id, snap_index=snap_index, class_name=class_name,
            pipeline=self.name, detection=detection, read=read, crop_path=crop_path,
            pipeline_ms=t.ms, error=None,
        )


def _save_crop(crop_bgr: np.ndarray, out_dir: Path, image_name: str) -> str:
    if crop_bgr.size == 0:
        return ""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / image_name
    ok, buf = cv2.imencode(".jpg", crop_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        return ""
    atomic_write_bytes(out_path, buf.tobytes())
    return str(out_path)
