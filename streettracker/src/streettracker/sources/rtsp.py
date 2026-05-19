"""Live RTSP frame source.

Renamed from NanoTracker's ``GstRtspSource`` — the class never actually
used GStreamer for the live path. Reolink's RTSP stream + GStreamer
exposes a SPS/PPS-handling quirk in OpenCV's GStreamer backend
(``nvv4l2decoder`` accepts the first frame then drops every subsequent
one with "Stream format not found"; ``avdec_h264`` never emits past
PAUSED). OpenCV's FFmpeg backend works reliably at ~20 FPS against the
same camera.

Decode is software here — but on a Tegra X1 / Orin Nano, TRT inference
(~10-50 ms/frame) dominates frame budget, so software decode costs at
most ~2 FPS vs NVDEC.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterator
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


class RtspSource:
    """Iterator-style wrapper around ``cv2.VideoCapture(url, CAP_FFMPEG)``.

    Usage::

        src = RtspSource(url, codec="h265")
        src.open()
        for frame_idx, t, frame_bgr in src.frames():
            ...
        src.close()

    ``t`` is wall-clock seconds since the first yielded frame.
    """

    is_file: bool = False
    fps: float = 0.0  # nominal — 0 means "live, no fixed fps"

    def __init__(
        self,
        rtsp_url: str,
        codec: str = "h265",
        transport: str = "tcp",
        connect_timeout_s: float = 15.0,
    ) -> None:
        self.rtsp_url = rtsp_url
        self.codec = codec
        self.transport = transport
        self.connect_timeout_s = connect_timeout_s
        self._cap = None  # type: ignore[assignment]
        self._first_frame: np.ndarray | None = None

    def _redacted_url(self) -> str:
        """RTSP URL with the password masked for logging."""
        if "@" not in self.rtsp_url:
            return self.rtsp_url
        creds, host = self.rtsp_url.rsplit("@", 1)
        scheme_user = creds.rsplit(":", 1)[0]
        return f"{scheme_user}:***@{host}"

    def open(self) -> None:
        import cv2

        # OpenCV reads OPENCV_FFMPEG_CAPTURE_OPTIONS in cap_ffmpeg.cpp.
        # TCP transport is far more reliable than UDP over WiFi.
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = f"rtsp_transport;{self.transport}"

        print(f"[rtsp] Opening via OpenCV FFmpeg backend: {self._redacted_url()}")
        t0 = time.monotonic()
        self._cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        if not self._cap.isOpened():
            raise RuntimeError(
                "Failed to open RTSP via FFmpeg backend. Check reachability "
                "(`ffprobe rtsp://...`) and credentials."
            )
        print(f"[rtsp] Stream opened in {time.monotonic() - t0:.1f}s")

        # Pull first frame to confirm the stream is actually emitting video,
        # rather than just having opened a connection that emits audio-only.
        deadline = time.monotonic() + self.connect_timeout_s
        while time.monotonic() < deadline:
            ok, frame = self._cap.read()
            if ok and frame is not None and frame.size > 0:
                h, w = frame.shape[:2]
                print(f"[rtsp] First frame: {w}x{h}")
                self._first_frame = frame
                return
            time.sleep(0.1)
        raise RuntimeError(
            f"Connected but no frames in {self.connect_timeout_s}s — "
            "stream may be audio-only or unhealthy."
        )

    def frames(self) -> Iterator[tuple[int, float, np.ndarray]]:
        if self._cap is None:
            raise RuntimeError("Source not opened; call .open() first")

        idx = 0
        t0: float | None = None
        if self._first_frame is not None:
            t0 = time.monotonic()
            yield idx, 0.0, self._first_frame
            idx += 1
            self._first_frame = None

        consecutive_failures = 0
        while True:
            ok, frame = self._cap.read()
            if not ok or frame is None or frame.size == 0:
                consecutive_failures += 1
                if consecutive_failures > 30:
                    print("[rtsp] 30 consecutive read failures, ending stream.")
                    return
                time.sleep(0.05)
                continue
            consecutive_failures = 0
            if t0 is None:
                t0 = time.monotonic()
                t = 0.0
            else:
                t = time.monotonic() - t0
            yield idx, t, frame
            idx += 1

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
