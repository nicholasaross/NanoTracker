"""FileSource — open and iterate a tiny synthetic MP4."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from streettracker.sources.base import FrameSource
from streettracker.sources.file import FileSource, build_file_pipeline

cv2 = pytest.importorskip("cv2")


@pytest.fixture
def tiny_mp4(tmp_path: Path) -> Path:
    """Synthetic 1-second 30fps MP4 (30 frames, 64x64) for source tests."""
    path = tmp_path / "tiny.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 30.0, (64, 64))
    if not writer.isOpened():
        pytest.skip("cv2.VideoWriter could not open mp4v encoder")
    try:
        for i in range(30):
            # Solid color gradient frame so each is distinguishable.
            frame = np.full((64, 64, 3), (i * 8, 0, 255 - i * 8), dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()
    assert path.stat().st_size > 0
    return path


def test_file_source_open_and_read(tiny_mp4: Path) -> None:
    src = FileSource(tiny_mp4, codec="h264", assumed_fps=30.0)
    src.open()
    frames = list(src.frames())
    src.close()
    assert len(frames) >= 25  # mp4v encoders sometimes lose 1-2 trailing frames
    idx0, t0, frame0 = frames[0]
    assert idx0 == 0
    assert t0 == 0.0
    assert frame0.shape == (64, 64, 3)


def test_file_source_fps_probed(tiny_mp4: Path) -> None:
    src = FileSource(tiny_mp4)
    src.open()
    # cv2's FFmpeg backend reports fps as 30 (sometimes 30.000004) — be loose.
    assert 29.0 < src.fps < 31.0
    src.close()


def test_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        FileSource("/nonexistent/video.mp4")


def test_invalid_codec_raises() -> None:
    with pytest.raises(ValueError, match="codec"):
        build_file_pipeline("/some/file.mp4", codec="vp9")


def test_pipeline_string_includes_nvdec() -> None:
    p = build_file_pipeline("/tmp/x.mp4", codec="h264")
    assert "nvv4l2decoder" in p
    assert "h264parse" in p
    assert "/tmp/x.mp4" in p


def test_file_source_satisfies_protocol(tiny_mp4: Path) -> None:
    src = FileSource(tiny_mp4)
    assert isinstance(src, FrameSource)
    assert src.is_file is True


def test_frames_without_open_raises(tiny_mp4: Path) -> None:
    src = FileSource(tiny_mp4)
    with pytest.raises(RuntimeError, match="not opened"):
        next(src.frames())


def test_close_is_idempotent(tiny_mp4: Path) -> None:
    src = FileSource(tiny_mp4)
    src.open()
    src.close()
    src.close()  # no error
