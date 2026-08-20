"""MediaPipe Hand/Wrist Localization (Phase 2).

Semantic localization only — this branch never computes CPM itself. Its job
is to (a) detect the hand/wrist per frame, (b) derive a dynamic CPR
foreground ROI (hand + wrist + part of forearm), and (c) hand CoTracker
(Phase 3) a set of initialization points inside that ROI. The full frame is
always preserved for ego-motion estimation (Phase 4) — nothing here
permanently crops the input video. Uses a pretrained MediaPipe HandLandmarker
checkpoint (see `hybrid.models`); never fine-tuned (CLAUDE.md rule 4).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

from hybrid.caching import CacheManager
from hybrid.config import HybridConfig, MediaPipeConfig
from hybrid.exceptions import HandNotDetectedError
from hybrid.logging_config import get_logger
from hybrid.models import HAND_LANDMARKER, ensure_model
from hybrid.video_io import Frame, VideoReader

logger = get_logger(__name__)

# MediaPipe Hands 21-landmark indices used for ROI / forearm-direction math.
WRIST = 0
MIDDLE_FINGER_MCP = 9
NUM_HAND_LANDMARKS = 21


@dataclass(frozen=True)
class RoiBox:
    x_min: int
    y_min: int
    x_max: int
    y_max: int

    def clip(self, width: int, height: int) -> RoiBox:
        return RoiBox(
            x_min=max(0, self.x_min),
            y_min=max(0, self.y_min),
            x_max=min(width, self.x_max),
            y_max=min(height, self.y_max),
        )

    @property
    def width(self) -> int:
        return max(0, self.x_max - self.x_min)

    @property
    def height(self) -> int:
        return max(0, self.y_max - self.y_min)


@dataclass(frozen=True)
class HandDetection:
    frame_index: int
    timestamp_sec: float
    detected: bool
    confidence: float | None = None
    wrist_xy: tuple[float, float] | None = None
    landmarks_xy: tuple[tuple[float, float], ...] | None = None  # 21 points, pixel coords
    bbox: RoiBox | None = None  # tight hand bounding box
    roi: RoiBox | None = None  # expanded dynamic CPR foreground ROI


@dataclass
class MediaPipeVideoResult:
    video_id: str
    frame_width: int
    frame_height: int
    detections: list[HandDetection]
    detection_rate: float  # fraction of frames with detected == True
    longest_gap_frames: int
    longest_gap_sec: float
    model_sha256: str


def roi_exclusion_mask(roi: RoiBox | None, width: int, height: int, dilate_px: int = 0) -> np.ndarray:
    """255 everywhere except 0 inside `roi` (dilated by `dilate_px`). Used to
    keep CPR foreground pixels out of background-only computations (Phase 4
    feature detection, Phase 6 background flow reading).
    """
    mask = np.full((height, width), 255, dtype=np.uint8)
    if roi is not None:
        x0 = max(0, roi.x_min - dilate_px)
        y0 = max(0, roi.y_min - dilate_px)
        x1 = min(width, roi.x_max + dilate_px)
        y1 = min(height, roi.y_max + dilate_px)
        mask[y0:y1, x0:x1] = 0
    return mask


def build_roi_lookup(detections: list[HandDetection]) -> dict[int, RoiBox]:
    """Per-frame ROI, forward-filled across brief MediaPipe detection gaps
    (Phase 2 showed these are rare/short in practice) so downstream branches
    (Phase 4 ego-motion, Phase 6 optical flow) get a sensible ROI rather than
    "no exclusion" on every gap.
    """
    lookup: dict[int, RoiBox] = {}
    last_roi: RoiBox | None = None
    for d in sorted(detections, key=lambda x: x.frame_index):
        if d.detected and d.roi is not None:
            last_roi = d.roi
        if last_roi is not None:
            lookup[d.frame_index] = last_roi
    return lookup


def _landmarks_to_pixels(landmarks, width: int, height: int) -> tuple[tuple[float, float], ...]:
    return tuple((lm.x * width, lm.y * height) for lm in landmarks)


def _bbox_from_points(points: tuple[tuple[float, float], ...]) -> RoiBox:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return RoiBox(x_min=int(min(xs)), y_min=int(min(ys)), x_max=int(max(xs)), y_max=int(max(ys)))


def _expand_roi(
    bbox: RoiBox,
    wrist: tuple[float, float],
    middle_mcp: tuple[float, float],
    mp_config: MediaPipeConfig,
    width: int,
    height: int,
) -> RoiBox:
    """Pad the tight hand bbox, then extend it toward the forearm — the
    direction from the middle-finger MCP through the wrist, continued past
    the wrist — so the ROI covers part of the forearm, not just the hand
    (Phase 2 spec: hand + wrist + part of forearm + surrounding motion area).
    """
    pad_x = bbox.width * mp_config.roi_padding_ratio
    pad_y = bbox.height * mp_config.roi_padding_ratio
    x_min, y_min = bbox.x_min - pad_x, bbox.y_min - pad_y
    x_max, y_max = bbox.x_max + pad_x, bbox.y_max + pad_y

    dx, dy = wrist[0] - middle_mcp[0], wrist[1] - middle_mcp[1]
    norm = float(np.hypot(dx, dy))
    if norm > 1e-6:
        long_side = max(bbox.width, bbox.height)
        ext = long_side * mp_config.forearm_extension_ratio
        ext_x = wrist[0] + (dx / norm) * ext
        ext_y = wrist[1] + (dy / norm) * ext
        x_min, y_min = min(x_min, ext_x), min(y_min, ext_y)
        x_max, y_max = max(x_max, ext_x), max(y_max, ext_y)

    return RoiBox(int(x_min), int(y_min), int(x_max), int(y_max)).clip(width, height)


class MediaPipeHandLocalizer:
    """Wraps a MediaPipe Tasks `HandLandmarker` in VIDEO mode (uses
    inter-frame temporal continuity — appropriate here since we process one
    full development video sequentially, not a live stream).
    """

    def __init__(self, config: HybridConfig):
        self._mp_config = config.mediapipe
        model_path = ensure_model(HAND_LANDMARKER, config.paths.models_dir)

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=RunningMode.VIDEO,
            num_hands=self._mp_config.max_num_hands,
            min_hand_detection_confidence=self._mp_config.min_hand_detection_confidence,
            min_hand_presence_confidence=self._mp_config.min_hand_presence_confidence,
            min_tracking_confidence=self._mp_config.min_tracking_confidence,
        )
        self._landmarker = HandLandmarker.create_from_options(options)

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> MediaPipeHandLocalizer:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def detect_frame(self, frame: Frame) -> HandDetection:
        height, width = frame.image.shape[:2]
        rgb = cv2.cvtColor(frame.image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(round(frame.timestamp_sec * 1000))

        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        if not result.hand_landmarks:
            return HandDetection(frame_index=frame.index, timestamp_sec=frame.timestamp_sec, detected=False)

        landmarks = result.hand_landmarks[0]
        points = _landmarks_to_pixels(landmarks, width, height)
        bbox = _bbox_from_points(points)
        confidence = None
        if result.handedness and result.handedness[0]:
            confidence = float(result.handedness[0][0].score)
        roi = _expand_roi(bbox, points[WRIST], points[MIDDLE_FINGER_MCP], self._mp_config, width, height)

        return HandDetection(
            frame_index=frame.index,
            timestamp_sec=frame.timestamp_sec,
            detected=True,
            confidence=confidence,
            wrist_xy=points[WRIST],
            landmarks_xy=points,
            bbox=bbox,
            roi=roi,
        )


def _longest_gap(detections: list[HandDetection]) -> tuple[int, float]:
    """Longest consecutive run of `detected == False`, in frames and seconds."""
    longest_frames = 0
    longest_sec = 0.0
    run_start: int | None = None
    for i, d in enumerate(detections):
        if not d.detected:
            if run_start is None:
                run_start = i
            continue
        if run_start is not None:
            run_len = i - run_start
            run_sec = detections[i - 1].timestamp_sec - detections[run_start].timestamp_sec
            if run_len > longest_frames:
                longest_frames, longest_sec = run_len, run_sec
            run_start = None
    if run_start is not None:
        run_len = len(detections) - run_start
        run_sec = detections[-1].timestamp_sec - detections[run_start].timestamp_sec
        if run_len > longest_frames:
            longest_frames, longest_sec = run_len, run_sec
    return longest_frames, longest_sec


def run_mediapipe_on_video(video_path, config: HybridConfig, video_id: str) -> MediaPipeVideoResult:
    """Run hand localization over every frame of one development video.
    Raises HandNotDetectedError if the hand is never detected at all (a
    video where MediaPipe finds nothing is not a usable signal downstream).
    """
    with VideoReader(video_path) as reader, MediaPipeHandLocalizer(config) as localizer:
        width, height = reader.width, reader.height
        detections = [localizer.detect_frame(frame) for frame in reader]

    detected_count = sum(1 for d in detections if d.detected)
    detection_rate = detected_count / len(detections) if detections else 0.0
    longest_gap_frames, longest_gap_sec = _longest_gap(detections)

    if longest_gap_sec > config.mediapipe.max_detection_gap_sec:
        logger.warning(
            "%s: longest hand-detection gap is %.2fs (%d frames) — exceeds configured threshold of %.2fs",
            video_id,
            longest_gap_sec,
            longest_gap_frames,
            config.mediapipe.max_detection_gap_sec,
        )

    if detected_count == 0:
        raise HandNotDetectedError(f"{video_id}: hand never detected in any of {len(detections)} frames")

    logger.info(
        "%s: hand detected in %d/%d frames (%.1f%%), longest gap %.2fs",
        video_id,
        detected_count,
        len(detections),
        detection_rate * 100,
        longest_gap_sec,
    )

    return MediaPipeVideoResult(
        video_id=video_id,
        frame_width=width,
        frame_height=height,
        detections=detections,
        detection_rate=detection_rate,
        longest_gap_frames=longest_gap_frames,
        longest_gap_sec=longest_gap_sec,
        model_sha256=HAND_LANDMARKER.sha256,
    )


def _mediapipe_config_hash(config: HybridConfig) -> str:
    """Cache key scope: only the MediaPipe sub-config + pinned model
    checksum, so tuning unrelated parameters (Butterworth cutoff, fusion
    weights, ...) never invalidates this branch's cache.
    """
    import hashlib

    payload = config.mediapipe.model_dump_json() + HAND_LANDMARKER.sha256
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def run_mediapipe_on_video_cached(
    video_path, config: HybridConfig, video_id: str, cache_manager: CacheManager
) -> MediaPipeVideoResult:
    """Cached wrapper around `run_mediapipe_on_video` (Phase 2 spec: expensive
    branch output cached per-video/per-config so tuning downstream phases
    doesn't re-run MediaPipe).
    """
    config_hash = _mediapipe_config_hash(config)
    key = cache_manager.make_key("mediapipe", video_id, config_hash)

    if config.caching.enabled and cache_manager.exists(key):
        logger.info("%s: loaded MediaPipe result from cache (key=%s)", video_id, key)
        return cache_manager.load(key)

    result = run_mediapipe_on_video(video_path, config, video_id)
    if config.caching.enabled:
        cache_manager.save(key, result)
    return result


def get_cotracker_init_points(detection: HandDetection, num_points: int) -> np.ndarray:
    """Candidate (x, y) pixel points inside the ROI for CoTracker (Phase 3)
    to track: the hand landmarks first (richest available signal), then a
    grid over the remaining ROI area (covers wrist/forearm beyond the
    landmarks) until `num_points` is reached.
    """
    if not detection.detected or detection.roi is None:
        raise HandNotDetectedError(
            f"Cannot generate CoTracker init points for frame {detection.frame_index}: no hand detected"
        )
    if num_points < 1:
        raise ValueError(f"num_points must be >= 1, got {num_points}")

    points: list[tuple[float, float]] = list(detection.landmarks_xy or [])[:num_points]

    remaining = num_points - len(points)
    if remaining > 0:
        roi = detection.roi
        grid_n = int(np.ceil(np.sqrt(remaining))) + 2
        xs = np.linspace(roi.x_min, roi.x_max, grid_n)[1:-1]
        ys = np.linspace(roi.y_min, roi.y_max, grid_n)[1:-1]
        grid_points = [(float(x), float(y)) for y in ys for x in xs]
        points.extend(grid_points[:remaining])

    return np.array(points[:num_points], dtype=np.float64)
