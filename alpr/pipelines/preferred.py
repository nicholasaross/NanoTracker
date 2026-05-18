"""Preferred pipeline: open-image-models (ONNX plate detection) + fast-plate-ocr (ONNX recognition).

Both are by ankandrew (https://github.com/ankandrew) and share a lightweight ONNX-only
posture.  No torch dep — onnxruntime backs both.
"""

from __future__ import annotations

import cv2
import numpy as np

from alpr.pipelines.base import (
    PlateDetection,
    PlateRead,
    normalize_plate_text,
)


class OpenImageModelsDetector:
    name = "oim"

    def __init__(
        self,
        detector_model: str = "yolo-v9-t-384-license-plate-end2end",
        det_conf: float = 0.25,
    ) -> None:
        from open_image_models import LicensePlateDetector
        self._detector = LicensePlateDetector(detection_model=detector_model)
        self._det_conf = det_conf

    def detect(self, image: np.ndarray) -> PlateDetection | None:
        detections = self._detector.predict(image)
        if not detections:
            return None
        best = None
        for d in detections:
            bbox, conf = _extract_bbox_conf(d)
            if conf is None or conf < self._det_conf:
                continue
            if best is None or conf > best.det_confidence:
                best = PlateDetection(bbox=bbox, det_confidence=conf)
        return best


class FastPlateOcrRecognizer:
    name = "fast-plate-ocr"

    def __init__(self, ocr_model: str = "global-plates-mobile-vit-v2-model") -> None:
        # Class name has shifted between versions; try the current name first, then legacy.
        try:
            from fast_plate_ocr import LicensePlateRecognizer as _Recognizer  # type: ignore
        except ImportError:
            from fast_plate_ocr import ONNXPlateRecognizer as _Recognizer  # type: ignore
        self._recognizer = _Recognizer(ocr_model)

    def recognize(self, crop_bgr: np.ndarray) -> PlateRead | None:
        if crop_bgr.size == 0:
            return None
        # fast-plate-ocr's global model is single-channel: per its docstring,
        # in-memory ndarrays "are assumed to already use the expected color mode",
        # so a 3-channel BGR crop raises an ONNXRuntime shape error.
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY) if crop_bgr.ndim == 3 else crop_bgr
        out = self._recognizer.run(gray, return_confidence=True)
        raw, conf = _unpack_ocr_output(out)
        if not raw:
            return None
        return PlateRead(
            text=normalize_plate_text(raw),
            ocr_confidence=conf,
            raw_text=raw,
        )


def _extract_bbox_conf(detection) -> tuple[tuple[int, int, int, int], float | None]:
    conf = getattr(detection, "confidence", None)
    if conf is None:
        conf = getattr(detection, "score", None)
    conf = float(conf) if conf is not None else None

    bb = (
        getattr(detection, "bounding_box", None)
        or getattr(detection, "bbox", None)
        or getattr(detection, "box", None)
    )
    if bb is not None and all(hasattr(bb, attr) for attr in ("x1", "y1", "x2", "y2")):
        bbox = (int(bb.x1), int(bb.y1), int(bb.x2), int(bb.y2))
    elif isinstance(detection, (tuple, list)) and len(detection) >= 4:
        bbox = tuple(int(v) for v in detection[:4])
    else:
        bbox = (0, 0, 0, 0)
    return bbox, conf


def _unpack_ocr_output(out) -> tuple[str, float]:
    """Tolerate the several shapes fast-plate-ocr has used across versions.

    v1.1.x: list[PlatePrediction] with .plate / .char_probs
    Older legacy: (list[str], probs_ndarray) tuple, or plain list[str], or str.
    """
    if isinstance(out, str):
        return out, 0.0
    if isinstance(out, list) and out:
        first = out[0]
        # v1.1.x: PlatePrediction dataclass
        plate = getattr(first, "plate", None)
        if plate is not None:
            char_probs = getattr(first, "char_probs", None)
            if char_probs is not None and hasattr(char_probs, "ndim"):
                # char_probs is (slots, n_chars) softmax — mean per-slot max prob is a sane confidence.
                conf = float(np.mean(np.max(char_probs, axis=-1)))
            else:
                conf = 0.0
            return str(plate), conf
        if isinstance(first, str):
            return first, 0.0
        if isinstance(first, tuple) and len(first) == 2:
            return str(first[0]), float(first[1])
    if isinstance(out, tuple) and len(out) == 2:
        text, conf = out
        text = text[0] if isinstance(text, list) else str(text)
        if hasattr(conf, "__iter__"):
            conf_val = float(np.mean(np.asarray(conf, dtype=float)))
        else:
            conf_val = float(conf)
        return text, conf_val
    return "", 0.0
