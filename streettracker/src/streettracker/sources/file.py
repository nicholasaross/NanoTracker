"""MP4 file frame source with NVDEC on Orin, FFmpeg software fallback.

Ported from NanoTracker's ``GstFileSource``. The GStreamer pipeline path
is preserved because it works reliably for *files* (the SPS/PPS quirk that
breaks live Reolink RTSP doesn't manifest from filesrc). On hosts where
OpenCV wasn't built with GStreamer (most dev boxes), we fall back to
``cv2.CAP_FFMPEG`` software decode automatically.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


def build_file_pipeline(file_path: str, codec: str = "h264") -> str:
    """GStreamer pipeline for a local MP4 file with NVDEC decode.

    ``sync=false drop=false`` so the decoder runs as fast as the inference
    loop allows — right for perf assessment (every frame processed) rather
    than realtime playback simulation.
    """
    codec = codec.lower()
    if codec == "h265":
        parse = "h265parse"
    elif codec == "h264":
        parse = "h264parse"
    else:
        raise ValueError(f"codec must be 'h264' or 'h265', got: {codec}")

    return (
        f"filesrc location={file_path} ! qtdemux ! {parse} "
        "! nvv4l2decoder "
        "! nvvidconv ! video/x-raw,format=BGRx "
        "! videoconvert ! video/x-raw,format=BGR "
        "! appsink sync=false drop=false max-buffers=2 emit-signals=false"
    )


class FileSource:
    """Iterator-style wrapper for an MP4 file.

    Yields ``(frame_index, video_time, BGR_frame)``. ``video_time`` is
    ``frame_idx / fps`` so tracker durations / speeds stay meaningful even
    when the host processes faster or slower than realtime playback.
    """

    is_file: bool = True

    def __init__(
        self,
        file_path: str | Path,
        codec: str = "h264",
        assumed_fps: float = 30.0,
    ) -> None:
        p = Path(file_path)
        if not p.exists():
            raise FileNotFoundError(f"Video file not found: {file_path}")
        self.file_path = str(p.resolve())
        self.codec = codec
        self.fps = float(assumed_fps)  # may be overridden after probe in open()
        self._cap = None  # type: ignore[assignment]
        self._used_nvdec = False
        self.pipeline_str = build_file_pipeline(self.file_path, codec)

    def open(self) -> None:
        import cv2

        # Probe nominal fps via cv2's FFmpeg backend (also our fallback path).
        probe = cv2.VideoCapture(self.file_path, cv2.CAP_FFMPEG)
        if probe.isOpened():
            f = probe.get(cv2.CAP_PROP_FPS)
            if f and f > 0:
                self.fps = f
            probe.release()
        print(f"[file] Nominal fps: {self.fps:.2f}")

        # Path A: GStreamer + NVDEC (Orin's fast path).
        build_info = cv2.getBuildInformation()
        has_gstreamer = (
            "GStreamer:                   YES" in build_info
            or "GStreamer:                       YES" in build_info
        )
        if has_gstreamer:
            print(f"[file] Trying GStreamer NVDEC pipeline: {self.pipeline_str}")
            self._cap = cv2.VideoCapture(self.pipeline_str, cv2.CAP_GSTREAMER)
            if self._cap.isOpened():
                print("[file] NVDEC pipeline open")
                self._used_nvdec = True
                return
            self._cap.release()
            print("[file] GStreamer pipeline failed to open; falling back to FFmpeg.")
        else:
            print(
                "[file] OpenCV not built with GStreamer support. "
                "Using FFmpeg software decode — NVDEC unused for this run."
            )

        # Path B: cv2 FFmpeg backend (software H.264 decode).
        self._cap = cv2.VideoCapture(self.file_path, cv2.CAP_FFMPEG)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Failed to open video file via GStreamer or FFmpeg: {self.file_path}"
            )
        print("[file] FFmpeg fallback open (software decode)")

    @property
    def used_nvdec(self) -> bool:
        """True if the GStreamer NVDEC pipeline opened successfully."""
        return self._used_nvdec

    def frames(self) -> Iterator[tuple[int, float, np.ndarray]]:
        if self._cap is None:
            raise RuntimeError("Source not opened; call .open() first")
        idx = 0
        consecutive_failures = 0
        while True:
            ok, frame = self._cap.read()
            if not ok or frame is None or frame.size == 0:
                consecutive_failures += 1
                if consecutive_failures > 5:
                    # End of file (or decoder gave up).
                    return
                time.sleep(0.01)
                continue
            consecutive_failures = 0
            t = idx / self.fps if self.fps > 0 else 0.0
            yield idx, t, frame
            idx += 1

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
