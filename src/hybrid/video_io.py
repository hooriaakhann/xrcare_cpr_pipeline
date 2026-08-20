"""Shared video-reading layer.

Pairs every decoded frame with its real presentation timestamp (PTS) rather
than assuming a constant `1/FPS` interval — the source videos are VFR. PTS
comes from `ffprobe` (frame-accurate for VFR/HEVC containers); OpenCV is used
only to decode pixel data, never for timing (`CAP_PROP_POS_MSEC` is unreliable
on these files). Frame index and timestamp travel together as one `Frame`
through the rest of the pipeline.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from hybrid.exceptions import TimestampError, VideoReadError
from hybrid.logging_config import get_logger

logger = get_logger(__name__)

# Tolerance for cv2 decoded-frame-count vs. ffprobe PTS-count disagreement
# before it's logged as a warning (containers can disagree by a frame or two
# at EOF; anything bigger points at a real demuxing problem).
_FRAME_COUNT_MISMATCH_WARN_FRACTION = 0.01


@dataclass(frozen=True)
class Frame:
    """One decoded frame with its real timestamp. `image` is the full,
    uncropped BGR frame (HxWx3) — ROI cropping happens downstream, never here.
    """

    index: int
    timestamp_sec: float
    image: np.ndarray


@dataclass(frozen=True)
class VideoMeta:
    path: Path
    width: int
    height: int
    frame_count: int
    duration_sec: float


def _ffprobe_frame_pts(path: Path) -> list[float]:
    """Real per-frame presentation timestamps, in decode order.

    Raises TimestampError if ffprobe fails, returns malformed output, no
    timestamps, or a non-monotonic sequence (any of which mean the frame
    index -> timestamp pairing downstream cannot be trusted).
    """
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "frame=pts_time",
        "-print_format",
        "json",
        str(path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except FileNotFoundError as e:
        raise TimestampError(f"ffprobe is not on PATH; cannot read frame timestamps for {path}") from e
    except subprocess.CalledProcessError as e:
        raise TimestampError(f"ffprobe failed to read frame timestamps for {path}: {e.stderr.strip()}") from e

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        raise TimestampError(f"ffprobe returned malformed JSON for {path}: {e}") from e

    frames = data.get("frames", [])
    pts = [float(fr["pts_time"]) for fr in frames if fr.get("pts_time") is not None]
    if not pts:
        raise TimestampError(f"ffprobe returned no frame timestamps for {path}")
    if any(b < a for a, b in zip(pts, pts[1:])):
        raise TimestampError(f"non-monotonic frame timestamps in {path} — corrupt or unsupported stream")
    return pts


def probe_video_meta(path: Path) -> VideoMeta:
    """Metadata derived from the same PTS source used by VideoReader, so
    `frame_count`/`duration_sec` always agree with what iterating will yield.
    """
    path = Path(path)
    if not path.exists():
        raise VideoReadError(f"Video file not found: {path}")

    pts = _ffprobe_frame_pts(path)

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise VideoReadError(f"OpenCV could not open video: {path}")
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()

    return VideoMeta(path=path, width=width, height=height, frame_count=len(pts), duration_sec=pts[-1])


class VideoReader:
    """Sequential reader over one video. Iterating yields `Frame`s pairing
    each cv2-decoded frame with its real ffprobe PTS by decode order.

    Usage:
        with VideoReader(path) as reader:
            for frame in reader:
                ...
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        if not self.path.exists():
            raise VideoReadError(f"Video file not found: {self.path}")

        self._pts = _ffprobe_frame_pts(self.path)

        self._cap = cv2.VideoCapture(str(self.path))
        if not self._cap.isOpened():
            raise VideoReadError(f"OpenCV could not open video: {self.path}")

        cv_frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if cv_frame_count and abs(cv_frame_count - len(self._pts)) > max(
            1, int(_FRAME_COUNT_MISMATCH_WARN_FRACTION * len(self._pts))
        ):
            logger.warning(
                "%s: cv2 frame count (%d) and ffprobe PTS count (%d) disagree by more than %.0f%%",
                self.path,
                cv_frame_count,
                len(self._pts),
                _FRAME_COUNT_MISMATCH_WARN_FRACTION * 100,
            )

        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._consumed = False

    @property
    def frame_count(self) -> int:
        return len(self._pts)

    @property
    def duration_sec(self) -> float:
        return self._pts[-1] if self._pts else 0.0

    def __iter__(self) -> Iterator[Frame]:
        return self.frames()

    def frames(self) -> Iterator[Frame]:
        if self._consumed:
            raise VideoReadError(f"{self.path}: VideoReader is single-pass and has already been iterated")
        self._consumed = True

        index = 0
        while True:
            ok, image = self._cap.read()
            if not ok:
                break
            if index >= len(self._pts):
                raise TimestampError(
                    f"{self.path}: decoded more frames ({index + 1}) than ffprobe reported "
                    f"timestamps ({len(self._pts)})"
                )
            yield Frame(index=index, timestamp_sec=self._pts[index], image=image)
            index += 1

        if index == 0:
            raise VideoReadError(f"No frames could be decoded from {self.path}")
        if index < len(self._pts):
            logger.warning(
                "%s: decoded only %d of %d expected frames (ffprobe count)", self.path, index, len(self._pts)
            )

    def close(self) -> None:
        self._cap.release()

    def __enter__(self) -> VideoReader:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
