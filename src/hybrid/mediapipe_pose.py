"""MediaPipe Pose Landmarker (visualization-only branch).

A second, separate MediaPipe model from Phase 2's hand localizer
(`mediapipe_roi.py`) -- detects the 33-point body pose skeleton
(shoulders, elbows, hips, ...) per frame. Not used by any estimator or by
CPM computation at all; exists solely so `overlay_video.py` can draw it
for visual sanity-checking/demo purposes. Uses a pretrained MediaPipe
PoseLandmarker checkpoint (see `hybrid.models`); never fine-tuned
(CLAUDE.md rule 4).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode

from hybrid.caching import CacheManager
from hybrid.config import HybridConfig
from hybrid.logging_config import get_logger
from hybrid.models import POSE_LANDMARKER, ensure_model
from hybrid.video_io import Frame, VideoReader

logger = get_logger(__name__)

NUM_POSE_LANDMARKS = 33

# Standard MediaPipe/BlazePose 33-point skeleton topology -- fixed anatomy,
# not a tunable parameter, so kept as a code constant rather than config.
# (mp.solutions is unavailable in this installed mediapipe build -- see
# mediapipe_roi.py's own module docstring -- so this can't be imported
# from the library itself and is spelled out explicitly.)
POSE_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 7),
    (0, 4),
    (4, 5),
    (5, 6),
    (6, 8),
    (9, 10),
    (11, 12),
    (11, 13),
    (13, 15),
    (15, 17),
    (15, 19),
    (15, 21),
    (17, 19),
    (12, 14),
    (14, 16),
    (16, 18),
    (16, 20),
    (16, 22),
    (18, 20),
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (25, 27),
    (27, 29),
    (29, 31),
    (27, 31),
    (24, 26),
    (26, 28),
    (28, 30),
    (30, 32),
    (28, 32),
)


@dataclass(frozen=True)
class PoseDetection:
    frame_index: int
    timestamp_sec: float
    detected: bool
    landmarks_xy: tuple[tuple[float, float], ...] | None = None  # 33 points, pixel coords


@dataclass
class MediaPipePoseVideoResult:
    video_id: str
    frame_width: int
    frame_height: int
    detections: list[PoseDetection]
    detection_rate: float  # fraction of frames with detected == True
    model_sha256: str


def _landmarks_to_pixels(landmarks, width: int, height: int) -> tuple[tuple[float, float], ...]:
    return tuple((lm.x * width, lm.y * height) for lm in landmarks)


class MediaPipePoseLocalizer:
    """Wraps a MediaPipe Tasks `PoseLandmarker` in VIDEO mode. Mirrors
    `MediaPipeHandLocalizer` (mediapipe_roi.py) -- same lifecycle, same
    inter-frame temporal continuity rationale.
    """

    def __init__(self, config: HybridConfig):
        self._pose_config = config.mediapipe_pose
        model_path = ensure_model(POSE_LANDMARKER, config.paths.models_dir)

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=self._pose_config.min_pose_detection_confidence,
            min_pose_presence_confidence=self._pose_config.min_pose_presence_confidence,
            min_tracking_confidence=self._pose_config.min_tracking_confidence,
        )
        self._landmarker = PoseLandmarker.create_from_options(options)

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> MediaPipePoseLocalizer:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def detect_frame(self, frame: Frame) -> PoseDetection:
        height, width = frame.image.shape[:2]
        rgb = cv2.cvtColor(frame.image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(round(frame.timestamp_sec * 1000))

        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        if not result.pose_landmarks:
            return PoseDetection(frame_index=frame.index, timestamp_sec=frame.timestamp_sec, detected=False)

        points = _landmarks_to_pixels(result.pose_landmarks[0], width, height)
        return PoseDetection(
            frame_index=frame.index,
            timestamp_sec=frame.timestamp_sec,
            detected=True,
            landmarks_xy=points,
        )


def run_mediapipe_pose_on_video(video_path, config: HybridConfig, video_id: str) -> MediaPipePoseVideoResult:
    """Run pose localization over every frame of one development video.
    Unlike the hand branch, never raises on zero detections -- this branch
    is purely decorative (overlay_video.py), not required by any estimator,
    so a video where pose is never found just means an empty skeleton in
    the diagnostic video, not a broken pipeline.
    """
    with VideoReader(video_path) as reader, MediaPipePoseLocalizer(config) as localizer:
        width, height = reader.width, reader.height
        detections = [localizer.detect_frame(frame) for frame in reader]

    detected_count = sum(1 for d in detections if d.detected)
    detection_rate = detected_count / len(detections) if detections else 0.0

    logger.info(
        "%s: pose detected in %d/%d frames (%.1f%%)",
        video_id,
        detected_count,
        len(detections),
        detection_rate * 100,
    )

    return MediaPipePoseVideoResult(
        video_id=video_id,
        frame_width=width,
        frame_height=height,
        detections=detections,
        detection_rate=detection_rate,
        model_sha256=POSE_LANDMARKER.sha256,
    )


def _pose_config_hash(config: HybridConfig) -> str:
    """Cache key scope: only the pose sub-config + pinned model checksum,
    same rationale as mediapipe_roi.py's `_mediapipe_config_hash`."""
    payload = config.mediapipe_pose.model_dump_json() + POSE_LANDMARKER.sha256
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def run_mediapipe_pose_on_video_cached(
    video_path, config: HybridConfig, video_id: str, cache_manager: CacheManager
) -> MediaPipePoseVideoResult:
    """Cached wrapper around `run_mediapipe_pose_on_video` -- same caching
    discipline as every other expensive branch (CLAUDE.md)."""
    config_hash = _pose_config_hash(config)
    key = cache_manager.make_key("mediapipe_pose", video_id, config_hash)

    if config.caching.enabled and cache_manager.exists(key):
        logger.info("%s: loaded MediaPipe Pose result from cache (key=%s)", video_id, key)
        return cache_manager.load(key)

    result = run_mediapipe_pose_on_video(video_path, config, video_id)
    if config.caching.enabled:
        cache_manager.save(key, result)
    return result
