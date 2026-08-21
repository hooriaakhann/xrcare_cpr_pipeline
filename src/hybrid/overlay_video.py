"""Diagnostic overlay video (visual, presentation-only -- not part of
inference/estimation).

For each development video, renders one MP4 to
`runs/development/<video_id>/overlay.mp4` combining, on the original
frames: MediaPipe's dynamic CPR ROI box (Phase 2) and connected hand
skeleton, MediaPipe Pose's connected body skeleton (`mediapipe_pose.py`,
a second, separate model), and CoTracker's raw tracked points colored by
visibility (Phase 3) -- plus a scrolling strip beneath the frame showing
the raw vs. ego-motion-corrected tracker motion signal (Phase 5) with a
moving playhead. A single sanity-check/demo artifact per video, not a new
inference output; the pose skeleton in particular feeds nothing
downstream -- see mediapipe_pose.py's own docstring.

Reuses cached branch results the same way Phase 16's diagnostics.py does;
nothing expensive re-runs if a video's branches are already cached. Unlike
diagnostics.py, this module does decode and re-encode the actual video
frames (needed to draw on them), so it's slower than writing CSV/JSON, but
still just decode+draw+encode -- no CPM-relevant model inference of its
own (Pose Landmarker inference happens once, cached, in
mediapipe_pose.py, same discipline as every other branch).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from hybrid.caching import CacheManager
from hybrid.config import HybridConfig
from hybrid.corrected_trajectory import correct_tracker_trajectory
from hybrid.cotracker_tracker import run_cotracker_on_video_cached
from hybrid.dataset import DevVideo
from hybrid.ego_motion import run_ego_motion_on_video_cached
from hybrid.exceptions import VideoWriteError
from hybrid.logging_config import get_logger
from hybrid.mediapipe_pose import POSE_CONNECTIONS, PoseDetection, run_mediapipe_pose_on_video_cached
from hybrid.mediapipe_roi import HAND_CONNECTIONS, HandDetection, run_mediapipe_on_video_cached
from hybrid.video_io import VideoReader

logger = get_logger(__name__)

# Fixed rendering colors (BGR) -- cosmetic only, not config fields; see
# OverlayVideoConfig's docstring for why.
_ROI_COLOR = (0, 220, 0)
_HAND_LANDMARK_COLOR = (255, 0, 255)
_HAND_BONE_COLOR = (255, 140, 255)
_POSE_LANDMARK_COLOR = (0, 165, 255)
_POSE_BONE_COLOR = (0, 215, 255)
_VISIBLE_POINT_COLOR = (0, 220, 0)
_OCCLUDED_POINT_COLOR = (0, 0, 220)
_RAW_CURVE_COLOR = (140, 140, 140)
_CORRECTED_CURVE_COLOR = (0, 200, 255)
_PLAYHEAD_COLOR = (255, 255, 255)
_STRIP_BG_COLOR = (30, 30, 30)
_STRIP_MARGIN_PX = 10


def _draw_skeleton(
    image: np.ndarray,
    points: tuple[tuple[float, float], ...],
    connections: tuple[tuple[int, int], ...],
    joint_color: tuple[int, int, int],
    bone_color: tuple[int, int, int],
    point_radius_px: int,
) -> None:
    """In-place: connected skeleton -- bone lines first, then joint dots on
    top, so joints stay visible where multiple bones meet."""
    pixel_points = [(int(round(x)), int(round(y))) for x, y in points]
    for i, j in connections:
        if i < len(pixel_points) and j < len(pixel_points):
            cv2.line(image, pixel_points[i], pixel_points[j], bone_color, max(1, point_radius_px // 2))
    for x, y in pixel_points:
        cv2.circle(image, (x, y), point_radius_px, joint_color, -1)


def _draw_mediapipe_overlay(
    image: np.ndarray, detection: HandDetection, point_radius_px: int, box_thickness_px: int
) -> None:
    """In-place: ROI box + connected hand skeleton for one frame, if a hand was detected."""
    if not detection.detected:
        return
    if detection.roi is not None:
        cv2.rectangle(
            image,
            (detection.roi.x_min, detection.roi.y_min),
            (detection.roi.x_max, detection.roi.y_max),
            _ROI_COLOR,
            box_thickness_px,
        )
    if detection.landmarks_xy is not None:
        landmark_radius = max(1, point_radius_px // 2)
        _draw_skeleton(
            image, detection.landmarks_xy, HAND_CONNECTIONS, _HAND_LANDMARK_COLOR, _HAND_BONE_COLOR, landmark_radius
        )


def _draw_pose_overlay(image: np.ndarray, detection: PoseDetection, point_radius_px: int) -> None:
    """In-place: connected body-pose skeleton for one frame, if a pose was detected."""
    if not detection.detected or detection.landmarks_xy is None:
        return
    landmark_radius = max(1, point_radius_px // 2)
    _draw_skeleton(
        image, detection.landmarks_xy, POSE_CONNECTIONS, _POSE_LANDMARK_COLOR, _POSE_BONE_COLOR, landmark_radius
    )


def _draw_cotracker_overlay(
    image: np.ndarray, xs: np.ndarray, ys: np.ndarray, visible: np.ndarray, point_radius_px: int
) -> None:
    """In-place: one dot per tracked point, green if visible else red."""
    for x, y, v in zip(xs, ys, visible):
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        color = _VISIBLE_POINT_COLOR if v else _OCCLUDED_POINT_COLOR
        cv2.circle(image, (int(round(x)), int(round(y))), point_radius_px, color, -1)


def _polyline_segments(
    timestamps: np.ndarray,
    values: np.ndarray,
    duration_sec: float,
    width: int,
    y_top: int,
    y_bottom: int,
    v_min: float,
    v_max: float,
) -> list[np.ndarray]:
    """Contiguous non-NaN (x_px, y_px) runs, ready for cv2.polylines -- split
    at NaN gaps rather than drawing a straight line across a gap in the
    signal (e.g. a frame with zero visible tracked points)."""
    if duration_sec <= 0:
        x_px = np.zeros_like(timestamps)
    else:
        x_px = np.clip((timestamps / duration_sec) * (width - 1), 0, width - 1)
    v_range = max(v_max - v_min, 1e-9)
    y_px = y_bottom - (values - v_min) / v_range * (y_bottom - y_top)

    finite = np.isfinite(values)
    segments: list[np.ndarray] = []
    run: list[list[int]] = []
    for i in range(len(values)):
        if finite[i]:
            run.append([int(x_px[i]), int(y_px[i])])
        elif run:
            segments.append(np.array(run, dtype=np.int32))
            run = []
    if run:
        segments.append(np.array(run, dtype=np.int32))
    return segments


def _render_motion_strip_base(
    width: int,
    height: int,
    timestamps: np.ndarray,
    raw_signal: np.ndarray,
    corrected_signal: np.ndarray,
    duration_sec: float,
) -> np.ndarray:
    """Static background for the motion-strip panel: both curves drawn once
    over the video's full duration. The moving playhead is drawn per-frame
    on a copy of this, not re-rendered every frame."""
    strip = np.full((height, width, 3), _STRIP_BG_COLOR, dtype=np.uint8)
    finite_values = np.concatenate(
        [raw_signal[np.isfinite(raw_signal)], corrected_signal[np.isfinite(corrected_signal)]]
    )
    if finite_values.size == 0:
        return strip

    v_min, v_max = float(np.min(finite_values)), float(np.max(finite_values))
    y_top, y_bottom = _STRIP_MARGIN_PX, height - _STRIP_MARGIN_PX
    for values, color in [(raw_signal, _RAW_CURVE_COLOR), (corrected_signal, _CORRECTED_CURVE_COLOR)]:
        segments = _polyline_segments(timestamps, values, duration_sec, width, y_top, y_bottom, v_min, v_max)
        if segments:
            cv2.polylines(strip, segments, isClosed=False, color=color, thickness=1, lineType=cv2.LINE_AA)
    return strip


def save_overlay_video_for_video(dev_video: DevVideo, config: HybridConfig, cache_manager: CacheManager) -> Path:
    video_id = dev_video.video_id
    ov_config = config.overlay_video
    run_dir = config.paths.runs_dir / "development" / video_id
    run_dir.mkdir(parents=True, exist_ok=True)
    output_path = run_dir / "overlay.mp4"

    mp_result = run_mediapipe_on_video_cached(dev_video.split_path, config, video_id, cache_manager)
    pose_result = run_mediapipe_pose_on_video_cached(dev_video.split_path, config, video_id, cache_manager)
    ct_result = run_cotracker_on_video_cached(dev_video.split_path, mp_result, config, video_id, cache_manager)
    ego_result = run_ego_motion_on_video_cached(dev_video.split_path, mp_result, config, video_id, cache_manager)
    corrected = correct_tracker_trajectory(ct_result, ego_result, video_id)

    frame_idx_to_row = {int(fi): row for row, fi in enumerate(ct_result.frame_indices)}

    with VideoReader(dev_video.split_path) as reader:
        duration_sec = reader.duration_sec
        fps = reader.frame_count / duration_sec if duration_sec > 0 else 30.0
        strip_height = ov_config.motion_strip_height_px
        strip_base = _render_motion_strip_base(
            reader.width,
            strip_height,
            corrected.timestamps_sec,
            corrected.raw_tracker_motion_y,
            corrected.corrected_tracker_motion_y,
            duration_sec,
        )

        fourcc = cv2.VideoWriter_fourcc(*ov_config.fourcc)
        composite_size = (reader.width, reader.height + strip_height)
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, composite_size)
        if not writer.isOpened():
            raise VideoWriteError(
                f"{video_id}: could not open VideoWriter for {output_path} (fourcc={ov_config.fourcc!r})"
            )

        try:
            for frame in reader:
                top = frame.image.copy()

                if 0 <= frame.index < len(mp_result.detections):
                    _draw_mediapipe_overlay(
                        top,
                        mp_result.detections[frame.index],
                        ov_config.point_radius_px,
                        ov_config.roi_box_thickness_px,
                    )

                if 0 <= frame.index < len(pose_result.detections):
                    _draw_pose_overlay(top, pose_result.detections[frame.index], ov_config.point_radius_px)

                row = frame_idx_to_row.get(frame.index)
                if row is not None:
                    _draw_cotracker_overlay(
                        top,
                        ct_result.tracks_x[row],
                        ct_result.tracks_y[row],
                        ct_result.visibility[row],
                        ov_config.point_radius_px,
                    )

                strip = strip_base.copy()
                x_px = (
                    int(np.clip((frame.timestamp_sec / duration_sec) * (reader.width - 1), 0, reader.width - 1))
                    if duration_sec > 0
                    else 0
                )
                cv2.line(strip, (x_px, 0), (x_px, strip_height - 1), _PLAYHEAD_COLOR, 1)

                composite = np.vstack([top, strip])
                writer.write(composite)
        finally:
            writer.release()

    logger.info("%s: wrote diagnostic overlay video to %s", video_id, output_path)
    return output_path


def save_overlay_videos_for_all_videos(
    dev_videos: list[DevVideo], config: HybridConfig, cache_manager: CacheManager
) -> list[Path]:
    return [save_overlay_video_for_video(v, config, cache_manager) for v in dev_videos]
