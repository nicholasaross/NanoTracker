"""GStreamer RTSP source with NVDEC (nvv4l2decoder) for Jetson.

Uses OpenCV's GStreamer backend (cv2.VideoCapture(gst, CAP_GSTREAMER)) which
is the simplest path that still gets NVDEC acceleration.  NVIDIA's L4T
OpenCV is built with GStreamer support, so this should work out of the box
on JetPack 4.6.1.

Pipeline shape:

  rtspsrc location=URL protocols=tcp latency=200
    ! rtp{h264,h265}depay ! {h264,h265}parse
    ! nvv4l2decoder
    ! nvvidconv ! video/x-raw,format=BGRx
    ! videoconvert ! video/x-raw,format=BGR
    ! appsink sync=false drop=true max-buffers=1 emit-signals=false

`drop=true max-buffers=1` discards stale frames -- we always want the most
recent frame for live inference rather than a backed-up queue.
"""

import time
from typing import Optional

import numpy as np

try:
    import cv2  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "opencv-python is required.  On the Nano, use the system package:\n"
        "  sudo apt install python3-opencv\n"
        "(JetPack's OpenCV is built with GStreamer + NVDEC support.)"
    ) from exc


def build_pipeline(rtsp_url, codec="h265", latency_ms=200, transport="tcp", use_nvdec=False):
    # type: (str, str, int, str, bool) -> str
    """RTSP pipeline.  Software decode by default.

    NVDEC (nvv4l2decoder) decodes Reolink streams unreliably on JetPack 4.6.1
    -- the decoder accepts the first frame, then starts dropping with
    'Stream format not found' until the next IDR.  Reolink's keyframe
    interval is long enough that the pipeline often appears to freeze.
    Software decode (avdec_{h264,h265}) has been observed stable on the
    same stream.  Set use_nvdec=True to opt into the NVDEC path.
    Decode is not the bottleneck (~5-15 ms vs 47 ms inference), so the
    perf hit from software decode is negligible.
    """
    codec = codec.lower()
    if codec == "h265":
        depay, parse, sw_dec = "rtph265depay", "h265parse", "avdec_h265"
    elif codec == "h264":
        depay, parse, sw_dec = "rtph264depay", "h264parse", "avdec_h264"
    else:
        raise ValueError("codec must be 'h264' or 'h265', got: {}".format(codec))

    if use_nvdec:
        decode_chain = "nvv4l2decoder ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert"
    else:
        decode_chain = "{sw} ! videoconvert".format(sw=sw_dec)

    return (
        "rtspsrc location={url} protocols={transport} latency={latency} "
        "! {depay} ! {parse} "
        "! {decode_chain} ! video/x-raw,format=BGR "
        "! appsink sync=false drop=true max-buffers=1 emit-signals=false"
    ).format(
        url=rtsp_url,
        transport=transport,
        latency=int(latency_ms),
        depay=depay,
        parse=parse,
        decode_chain=decode_chain,
    )


def build_file_pipeline(file_path, codec="h264"):
    # type: (str, str) -> str
    """GStreamer pipeline for a local MP4 file with NVDEC decode.

    Use sync=false and drop=false so the decoder runs as fast as the
    inference loop allows -- right for a perf assessment (every frame is
    processed) rather than realtime playback simulation.
    """
    codec = codec.lower()
    if codec == "h265":
        parse = "h265parse"
    elif codec == "h264":
        parse = "h264parse"
    else:
        raise ValueError("codec must be 'h264' or 'h265', got: {}".format(codec))

    return (
        "filesrc location={path} ! qtdemux ! {parse} "
        "! nvv4l2decoder "
        "! nvvidconv ! video/x-raw,format=BGRx "
        "! videoconvert ! video/x-raw,format=BGR "
        "! appsink sync=false drop=false max-buffers=2 emit-signals=false"
    ).format(path=file_path, parse=parse)


class GstRtspSource:
    """Iterator-style wrapper around cv2.VideoCapture for an RTSP NVDEC pipeline.

    Usage:

        src = GstRtspSource(url, codec="h265")
        src.open()
        for frame_idx, t, frame_bgr in src.frames():
            ...
        src.close()

    ``t`` is wall-clock seconds since the first yielded frame.  For file
    sources (GstFileSource) it is video-time (frame_idx / fps).
    """

    is_file = False
    fps = 0.0  # nominal; 0 means "live, no fixed fps"

    def __init__(
        self,
        rtsp_url,           # type: str
        codec="h265",       # type: str
        latency_ms=200,     # type: int
        transport="tcp",    # type: str
        connect_timeout_s=15.0,  # type: float
    ):
        self.rtsp_url = rtsp_url
        self.codec = codec
        self.latency_ms = latency_ms
        self.transport = transport
        self.connect_timeout_s = connect_timeout_s
        self._cap = None  # type: Optional[cv2.VideoCapture]
        self.pipeline_str = build_pipeline(rtsp_url, codec, latency_ms, transport)

    def open(self):
        # Decode path selection.
        #
        # We tried two GStreamer pipelines for live RTSP (nvv4l2decoder NVDEC
        # and avdec_h264 software).  Both behave unreliably against the
        # Reolink sub-stream over WiFi on this Nano: NVDEC accepts the first
        # frame and then drops every subsequent one with "Stream format not
        # found"; avdec_h264 never emits a single frame past PLAYING state.
        # Cause is likely a Reolink/GStreamer SPS/PPS-handling quirk that
        # ReolinkDemo on the dev box also documents (they use a separate
        # ffmpeg subprocess for the same reason).
        #
        # cv2.VideoCapture(url, CAP_FFMPEG) -- using OpenCV's bundled
        # FFmpeg backend directly against the RTSP URL -- works reliably
        # at ~20 FPS for this camera, and we've already established that
        # inference (47 ms) is the bottleneck.  Decode is software here;
        # NVDEC remains in use for MP4 file inputs via GstFileSource.

        print("[rtsp] Opening via OpenCV FFmpeg backend: {}".format(
            self.rtsp_url.replace(self.rtsp_url.split("@", 1)[0].split(":", 2)[-1], "***")
            if "@" in self.rtsp_url else self.rtsp_url,
        ))
        # Hint FFmpeg to use TCP transport for RTSP (low loss vs UDP on WiFi).
        # OPENCV_FFMPEG_CAPTURE_OPTIONS is read by OpenCV's cap_ffmpeg.cpp.
        import os as _os
        _os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;{}".format(self.transport)

        t0 = time.monotonic()
        self._cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        if not self._cap.isOpened():
            raise RuntimeError(
                "Failed to open RTSP via FFmpeg backend.  Check the URL is reachable "
                "(ffprobe rtsp://...) and credentials are correct."
            )
        print("[rtsp] Stream opened in {:.1f}s".format(time.monotonic() - t0))

        # Pull first frame to confirm the stream is actually emitting video.
        deadline = time.monotonic() + self.connect_timeout_s
        while time.monotonic() < deadline:
            ok, frame = self._cap.read()
            if ok and frame is not None and frame.size > 0:
                h, w = frame.shape[:2]
                print("[rtsp] First frame: {}x{}".format(w, h))
                self._first_frame = frame
                return
            time.sleep(0.1)
        raise RuntimeError(
            "Connected but no frames in {}s -- camera might be sending audio-only or stream is unhealthy.".format(
                self.connect_timeout_s,
            )
        )

    def frames(self):
        """Generator yielding (frame_index, t_seconds, BGR frame).

        ``t`` is wall-clock seconds since the first yielded frame.

        Iteration ends when the source stops producing frames (camera
        disconnect, end of stream).
        """
        if self._cap is None:
            raise RuntimeError("Source not opened; call .open() first")

        idx = 0
        t0 = None
        if getattr(self, "_first_frame", None) is not None:
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
                    print("[gst] 30 consecutive read failures, ending stream.")
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

    def close(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class GstFileSource:
    """GStreamer file source with NVDEC decode for perf testing on recorded video.

    Behaves like GstRtspSource but reads a local MP4 instead of an RTSP
    stream.  Yields (frame_index, video_time, BGR_frame) where video_time
    is frame_idx / fps -- so tracker durations / speeds remain meaningful
    even when the Nano processes faster or slower than realtime.
    """

    is_file = True

    def __init__(self, file_path, codec="h264", assumed_fps=30.0):
        # type: (str, str, float) -> None
        from pathlib import Path
        p = Path(file_path)
        if not p.exists():
            raise FileNotFoundError("Video file not found: {}".format(file_path))
        self.file_path = str(p.resolve())
        self.codec = codec
        self.fps = float(assumed_fps)  # may be overridden after probe
        self._cap = None  # type: Optional[cv2.VideoCapture]
        self.pipeline_str = build_file_pipeline(self.file_path, codec)

    def open(self):
        # Probe nominal fps via cv2 FFmpeg backend (also our fallback path).
        probe = cv2.VideoCapture(self.file_path, cv2.CAP_FFMPEG)
        if probe.isOpened():
            f = probe.get(cv2.CAP_PROP_FPS)
            if f and f > 0:
                self.fps = f
            probe.release()
        print("[gst-file] Nominal fps: {:.2f}".format(self.fps))

        # Path A: try GStreamer + NVDEC.
        bi = cv2.getBuildInformation()
        has_gstreamer = "GStreamer:                   YES" in bi or "GStreamer:                       YES" in bi
        if has_gstreamer:
            print("[gst-file] Trying GStreamer NVDEC pipeline: {}".format(self.pipeline_str))
            self._cap = cv2.VideoCapture(self.pipeline_str, cv2.CAP_GSTREAMER)
            if self._cap.isOpened():
                print("[gst-file] NVDEC pipeline open")
                return
            self._cap.release()
            print("[gst-file] GStreamer pipeline failed to open, falling back to FFmpeg software decode")
        else:
            print("[gst-file] OpenCV not built with GStreamer (apt python3-opencv 3.2 on Bionic). "
                  "Falling back to FFmpeg software decode -- NVDEC unused for this run.")

        # Path B: fall back to cv2's FFmpeg backend (software H.264 decode on CPU).
        self._cap = cv2.VideoCapture(self.file_path, cv2.CAP_FFMPEG)
        if not self._cap.isOpened():
            raise RuntimeError(
                "Failed to open video file by either GStreamer or FFmpeg: {}".format(self.file_path)
            )
        print("[gst-file] FFmpeg fallback open (software decode)")

    def frames(self):
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

    def close(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None
