"""TensorRT YOLOv8 inference wrapper for Jetson Nano (TRT 8.2 API).

This module is the perf-critical core of NanoTracker.  It:

  1. Loads a prebuilt FP16 .engine file (built on the Nano via trtexec --
     engines are not portable across GPU architectures).
  2. Allocates pinned host buffers + device buffers via pycuda.
  3. Runs synchronous inference (execute_v2) on a single 640x640 BGR frame.
  4. Decodes the YOLOv8 output tensor (1, 84, 8400) into detections and
     applies confidence-thresholded class-aware non-maximum suppression
     using numpy only (no torch dependency).

The wrapper deliberately avoids any ultralytics or torch imports so it can
run on JetPack 4.6.1's Python 3.6.9 environment.
"""

import time
from typing import List, NamedTuple, Optional, Tuple

import numpy as np

try:
    import tensorrt as trt  # type: ignore
    import pycuda.driver as cuda  # type: ignore
    import pycuda.autoinit  # noqa: F401  -- side-effect: initialise primary context
except ImportError as exc:  # pragma: no cover -- import guard for dev-box use
    raise ImportError(
        "tensorrt and pycuda are required on the Nano.  Install via apt:\n"
        "  sudo apt install python3-libnvinfer python3-libnvinfer-dev\n"
        "  pip3 install --user pycuda==2021.1\n"
        "See docs/setup_nano.md for the full install order."
    ) from exc


# YOLOv8 COCO class IDs we care about.  Default filter is car-only (matches
# the VehicleTracker default); the per-deployment filter lives in
# inference.vehicle_classes in camera_config.json.  The name map covers every
# class anyone is likely to enable here so the dashboard's Type column reads
# correctly (person passes get labeled "person" instead of "unknown").
VEHICLE_CLASSES = (2, 3, 5, 7)  # car, motorcycle, bus, truck (legacy default)
CLASS_NAMES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle",
               5: "bus", 7: "truck", 15: "cat", 16: "dog"}


class Detection(NamedTuple):
    """Single post-NMS detection in original (un-letterboxed) image coordinates."""
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    class_id: int


class _HostDeviceBuffer:
    __slots__ = ("host", "device", "nbytes", "shape", "dtype")

    def __init__(self, host, device, nbytes, shape, dtype):
        self.host = host
        self.device = device
        self.nbytes = nbytes
        self.shape = shape
        self.dtype = dtype


class TRTYolo:
    """YOLOv8 TensorRT inference + decode.

    Constructor loads the engine and allocates buffers; ``infer(bgr_frame)``
    returns a list of Detection in the original frame's pixel coordinates.
    """

    def __init__(
        self,
        engine_path,        # type: str
        input_size=640,     # type: int
        conf_threshold=0.30,  # type: float
        iou_threshold=0.45,   # type: float
        class_filter=VEHICLE_CLASSES,  # type: Tuple[int, ...]
    ):
        self.input_size = int(input_size)
        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)
        self.class_filter = set(int(c) for c in class_filter) if class_filter else None

        self._logger = trt.Logger(trt.Logger.WARNING)
        self._runtime = trt.Runtime(self._logger)

        with open(engine_path, "rb") as f:
            engine_bytes = f.read()
        self.engine = self._runtime.deserialize_cuda_engine(engine_bytes)
        if self.engine is None:
            raise RuntimeError(
                "Failed to deserialise engine: {}.  Likely built against a different "
                "TRT version (this Nano needs an engine built locally with trtexec).".format(engine_path)
            )

        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        self._inputs = []   # type: List[_HostDeviceBuffer]
        self._outputs = []  # type: List[_HostDeviceBuffer]
        self._bindings = [] # type: List[int]
        self._allocate_buffers()

        # Inference timing for FPS reporting.
        self.last_infer_ms = 0.0
        self._infer_ema_ms = 0.0  # exponential moving average

    # ------------------------------------------------------------------
    # Buffer allocation (TRT 8.x binding API)
    # ------------------------------------------------------------------

    def _allocate_buffers(self):
        for i in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(i)
            shape = tuple(self.engine.get_binding_shape(i))
            dtype = trt.nptype(self.engine.get_binding_dtype(i))

            # Some YOLOv8 exports have a dynamic batch dim of -1; clamp to 1.
            shape = tuple(1 if d < 0 else d for d in shape)
            size = int(np.prod(shape))
            nbytes = size * np.dtype(dtype).itemsize

            host = cuda.pagelocked_empty(size, dtype)
            device = cuda.mem_alloc(nbytes)
            self._bindings.append(int(device))

            buf = _HostDeviceBuffer(host, device, nbytes, shape, dtype)
            if self.engine.binding_is_input(i):
                self._inputs.append(buf)
                self._input_name = name
                self._input_shape = shape  # (1, 3, H, W) expected
            else:
                self._outputs.append(buf)

        if len(self._inputs) != 1:
            raise RuntimeError(
                "Expected exactly one input binding, got {}".format(len(self._inputs))
            )
        if len(self._outputs) < 1:
            raise RuntimeError("Engine has no output bindings")

    # ------------------------------------------------------------------
    # Pre-processing: letterbox + normalize + HWC->CHW + BGR->RGB
    # ------------------------------------------------------------------

    def _letterbox(self, bgr):
        # type: (np.ndarray) -> Tuple[np.ndarray, float, int, int]
        """Resize keeping aspect ratio, pad with grey (114) to input_size.

        Returns the prepared float32 RGB CHW tensor, the scale factor, and
        the x/y padding offsets (needed to map detections back to original).
        """
        ih, iw = bgr.shape[:2]
        s = self.input_size
        scale = min(s / iw, s / ih)
        nw, nh = int(round(iw * scale)), int(round(ih * scale))
        # Use cv2 if available (faster than numpy); fall back to PIL.
        try:
            import cv2  # type: ignore
            resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        except ImportError:  # pragma: no cover
            from PIL import Image  # type: ignore
            resized = np.array(Image.fromarray(bgr).resize((nw, nh), Image.BILINEAR))

        canvas = np.full((s, s, 3), 114, dtype=np.uint8)
        dx = (s - nw) // 2
        dy = (s - nh) // 2
        canvas[dy:dy + nh, dx:dx + nw] = resized

        # BGR -> RGB, HWC -> CHW, uint8 -> float32 [0, 1], add batch dim.
        rgb = canvas[:, :, ::-1]
        chw = np.transpose(rgb, (2, 0, 1)).astype(np.float32) / 255.0
        return np.ascontiguousarray(chw[None, ...]), scale, dx, dy

    # ------------------------------------------------------------------
    # Inference + decode
    # ------------------------------------------------------------------

    def infer(self, bgr_frame):
        # type: (np.ndarray) -> List[Detection]
        if bgr_frame is None or bgr_frame.size == 0:
            return []
        ih, iw = bgr_frame.shape[:2]

        tensor, scale, dx, dy = self._letterbox(bgr_frame)
        np.copyto(self._inputs[0].host, tensor.ravel())

        t0 = time.monotonic()
        cuda.memcpy_htod_async(self._inputs[0].device, self._inputs[0].host, self.stream)
        self.context.execute_async_v2(self._bindings, self.stream.handle)
        for out in self._outputs:
            cuda.memcpy_dtoh_async(out.host, out.device, self.stream)
        self.stream.synchronize()
        dt = (time.monotonic() - t0) * 1000.0
        self.last_infer_ms = dt
        # 0.1 smoothing
        self._infer_ema_ms = dt if self._infer_ema_ms == 0 else (self._infer_ema_ms * 0.9 + dt * 0.1)

        # Reshape and decode primary output.  Most YOLOv8 exports give a
        # single output of shape (1, 4 + nc, num_anchors) e.g. (1, 84, 8400).
        out0 = self._outputs[0]
        raw = np.array(out0.host, copy=False).reshape(out0.shape)
        return self._decode(raw, scale, dx, dy, iw, ih)

    def _decode(self, raw, scale, dx, dy, iw, ih):
        # type: (np.ndarray, float, int, int, int, int) -> List[Detection]
        # raw shape: (1, 4 + nc, N) -> transpose to (N, 4 + nc)
        arr = raw[0].T  # (N, 4 + nc)
        if arr.shape[1] < 5:
            return []
        boxes_xywh = arr[:, :4]
        class_scores = arr[:, 4:]  # (N, nc)

        # If we have a class filter, restrict to those columns to avoid
        # NMS-ing detections we'd just drop.
        if self.class_filter is not None:
            keep_cols = sorted(c for c in self.class_filter if c < class_scores.shape[1])
            if not keep_cols:
                return []
            subset_scores = class_scores[:, keep_cols]
            class_ids_in_subset = subset_scores.argmax(axis=1)
            scores = subset_scores.max(axis=1)
            class_ids = np.array(keep_cols, dtype=np.int32)[class_ids_in_subset]
        else:
            class_ids = class_scores.argmax(axis=1).astype(np.int32)
            scores = class_scores.max(axis=1)

        mask = scores >= self.conf_threshold
        if not np.any(mask):
            return []
        boxes_xywh = boxes_xywh[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]

        # xywh (centre-format, letterbox space) -> xyxy (original image space)
        cx, cy, w, h = boxes_xywh[:, 0], boxes_xywh[:, 1], boxes_xywh[:, 2], boxes_xywh[:, 3]
        x1 = (cx - w / 2 - dx) / scale
        y1 = (cy - h / 2 - dy) / scale
        x2 = (cx + w / 2 - dx) / scale
        y2 = (cy + h / 2 - dy) / scale
        # Clamp to image bounds.
        x1 = np.clip(x1, 0, iw - 1)
        y1 = np.clip(y1, 0, ih - 1)
        x2 = np.clip(x2, 0, iw - 1)
        y2 = np.clip(y2, 0, ih - 1)
        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

        # Class-aware NMS.
        keep_indices = []
        for cls in np.unique(class_ids):
            sel = np.where(class_ids == cls)[0]
            kept = _nms(boxes_xyxy[sel], scores[sel], self.iou_threshold)
            keep_indices.extend(sel[k] for k in kept)

        dets = []
        for i in keep_indices:
            dets.append(Detection(
                x1=float(boxes_xyxy[i, 0]),
                y1=float(boxes_xyxy[i, 1]),
                x2=float(boxes_xyxy[i, 2]),
                y2=float(boxes_xyxy[i, 3]),
                score=float(scores[i]),
                class_id=int(class_ids[i]),
            ))
        return dets

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def avg_infer_ms(self):
        return self._infer_ema_ms


def _nms(boxes, scores, iou_threshold):
    # type: (np.ndarray, np.ndarray, float) -> List[int]
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        union = areas[i] + areas[order[1:]] - inter + 1e-9
        iou = inter / union
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    return keep
