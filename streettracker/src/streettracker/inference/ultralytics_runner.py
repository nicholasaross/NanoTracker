"""Ultralytics YOLO + integrated tracker.

Replaces NanoTracker's hand-rolled ``trt_engine.TRTYolo`` (manual YOLOv8
decode + numpy NMS) and bespoke ``IoUTracker`` with a single Ultralytics
call::

    model = YOLO("best.engine")
    results = model.track(frame, persist=True, tracker="botsort.yaml", verbose=False)

Ultralytics handles the engine deserialization, letterbox preprocess,
output decode, NMS, AND tracker association in one library call.
Net code savings: ~400 lines vs the hand-rolled implementation.

Heavy imports (``torch``, ``ultralytics``) are deferred to ``__init__``
so the rest of the package stays cheap to import on analysis-only hosts.
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from streettracker.common.coco import VEHICLE_CLASSES
from streettracker.common.types import Detection

if TYPE_CHECKING:
    import numpy as np


class UltralyticsTracker:
    """YOLO + BotSORT (or ByteTrack) inference wrapper.

    The same instance handles both detection and tracking — Ultralytics'
    ``.track()`` call wraps the tracker around the underlying model and
    persists state across calls when ``persist=True``.
    """

    def __init__(
        self,
        weights_path: str | Path,
        *,
        tracker_config: str | Path = "botsort.yaml",
        conf: float = 0.30,
        iou: float = 0.45,
        class_filter: Iterable[int] | None = VEHICLE_CLASSES,
        input_size: int = 640,
        device: str | int = 0,  # 0 = first CUDA GPU; "cpu" / "mps" for dev box
        verbose: bool = False,
    ) -> None:
        from ultralytics import YOLO  # deferred — heavy import

        self.weights_path = str(weights_path)
        self.tracker_config = str(tracker_config)
        self.conf = float(conf)
        self.iou = float(iou)
        self.class_filter: list[int] | None = (
            sorted({int(c) for c in class_filter}) if class_filter else None
        )
        self.input_size = int(input_size)
        self.device = device
        self.verbose = verbose

        self._model = YOLO(self.weights_path)

        # Timing — exponential moving average over per-frame inference cost.
        self.last_infer_ms = 0.0
        self._infer_ema_ms = 0.0

    @property
    def avg_infer_ms(self) -> float:
        return self._infer_ema_ms

    @property
    def model(self) -> Any:
        """Underlying ``ultralytics.YOLO`` model (for advanced users)."""
        return self._model

    def track(self, frame: np.ndarray) -> list[Detection]:
        """Run detection + tracking on a single BGR frame.

        Returns a list of ``Detection`` with ``track_id`` populated (or
        ``None`` for detections that haven't been promoted to a track yet —
        e.g. very-low-confidence boxes that BotSORT discarded).
        """
        if frame is None or frame.size == 0:
            return []

        t0 = time.monotonic()
        results = self._model.track(
            source=frame,
            persist=True,
            tracker=self.tracker_config,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.input_size,
            classes=self.class_filter,
            device=self.device,
            verbose=self.verbose,
        )
        dt = (time.monotonic() - t0) * 1000.0
        self.last_infer_ms = dt
        self._infer_ema_ms = (
            dt if self._infer_ema_ms == 0.0 else (self._infer_ema_ms * 0.9 + dt * 0.1)
        )

        if not results:
            return []
        return _ultralytics_results_to_detections(results[0])

    def reset(self) -> None:
        """Clear tracker state (forgets all active tracks).

        Call between batch runs or when the camera moves substantially —
        otherwise BotSORT will try to associate new detections with stale
        IDs across the discontinuity.
        """
        # Ultralytics' tracker state lives on the predictor object. Re-running
        # with persist=False once flushes it; alternatively we could call
        # `self._model.predictor.trackers[0].reset()` but that touches a
        # private API.
        if hasattr(self._model, "predictor") and self._model.predictor is not None:
            trackers = getattr(self._model.predictor, "trackers", None)
            if trackers:
                for t in trackers:
                    if hasattr(t, "reset"):
                        t.reset()


def _ultralytics_results_to_detections(result: Any) -> list[Detection]:
    """Convert a single ``ultralytics.engine.results.Results`` into our
    ``Detection`` list.

    Pulled out as a free function so we can unit-test the conversion
    without a real model on hand (the test passes a stub object that
    quacks like Results).
    """
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []

    # `.xyxy`, `.conf`, `.cls`, `.id` are torch tensors. Use .cpu().tolist()
    # for the smallest dependency surface — no torch import here.
    xyxy = boxes.xyxy.cpu().tolist()
    conf = boxes.conf.cpu().tolist()
    cls = boxes.cls.cpu().tolist()
    ids = boxes.id.cpu().tolist() if boxes.id is not None else [None] * len(xyxy)

    out: list[Detection] = []
    for (x1, y1, x2, y2), s, c, tid in zip(xyxy, conf, cls, ids, strict=False):
        out.append(
            Detection(
                x1=float(x1),
                y1=float(y1),
                x2=float(x2),
                y2=float(y2),
                score=float(s),
                class_id=int(c),
                track_id=int(tid) if tid is not None else None,
            )
        )
    return out
