"""Bespoke pipeline: license_plate_detector.pt (Ultralytics YOLOv8) + EasyOCR."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from alpr.pipelines.base import (
    PlateDetection,
    PlateRead,
    normalize_plate_text,
)


class BespokeDetector:
    name = "bespoke"

    def __init__(self, model_path: Path, imgsz: int = 640, det_conf: float = 0.25) -> None:
        if not model_path.exists():
            raise FileNotFoundError(
                f"Bespoke YOLO model not found at {model_path}.\n"
                f"Copy it from D:\\Projects\\NanoTracker\\license_plate_detector.pt "
                f"into alpr/models/ (the *.pt rule in .gitignore keeps it untracked)."
            )
        from ultralytics import YOLO  # lazy: torch is heavy

        self._yolo = YOLO(str(model_path))
        self._imgsz = imgsz
        self._det_conf = det_conf

    def detect(self, image: np.ndarray) -> PlateDetection | None:
        results = self._yolo.predict(
            image, imgsz=self._imgsz, conf=self._det_conf, verbose=False
        )
        best = None
        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue
            xyxy = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            idx = int(np.argmax(confs))
            candidate = PlateDetection(
                bbox=tuple(int(v) for v in xyxy[idx]),
                det_confidence=float(confs[idx]),
            )
            if best is None or candidate.det_confidence > best.det_confidence:
                best = candidate
        return best


class EasyOcrRecognizer:
    name = "easyocr"

    def __init__(self, langs: tuple[str, ...] = ("en",), use_gpu: bool = False) -> None:
        import easyocr
        self._reader = easyocr.Reader(list(langs), gpu=use_gpu, verbose=False)

    def recognize(self, crop_bgr: np.ndarray) -> PlateRead | None:
        if crop_bgr.size == 0:
            return None
        results = self._reader.readtext(crop_bgr, detail=1, paragraph=False)
        if not results:
            return None
        results.sort(key=lambda r: -r[2])
        raw = results[0][1]
        conf = float(results[0][2])
        return PlateRead(
            text=normalize_plate_text(raw),
            ocr_confidence=conf,
            raw_text=raw,
        )


