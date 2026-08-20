"""RANSAC Affine Ego-Motion Compensation (Phase 4).

Estimates the camera/head motion of the smart-glasses wearer, separate from
the CPR compression motion, so Phase 5 can subtract it from the CoTracker
trajectory. For each consecutive frame pair, background features are
detected *outside* the current CPR foreground ROI (Phase 2's MediaPipe
output, dilated for margin) so compression motion never leaks into the
camera-motion estimate, tracked with Lucas-Kanade optical flow, and fit with
a RANSAC-robust similarity transform: `cv2.estimateAffinePartial2D`
(translation + rotation + uniform scale) rather than the full 6-DOF
`estimateAffine2D` -- a wearer's head motion is well-modeled as a similarity
transform, and the extra 2 DOF of full affine (independent x/y scale + shear)
would just fit noise for this physical motion (see ADR 0001, Phase 20).

Each frame pair gets a fresh feature detection rather than one persistent
long-lived track: simpler, and self-correcting -- one bad pair never
propagates into the next.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import cv2
import numpy as np

from hybrid.caching import CacheManager
from hybrid.config import EgoMotionConfig, HybridConfig
from hybrid.exceptions import EgoMotionUnreliableError
from hybrid.logging_config import get_logger
from hybrid.mediapipe_roi import MediaPipeVideoResult, RoiBox
from hybrid.mediapipe_roi import build_roi_lookup as _build_roi_lookup
from hybrid.mediapipe_roi import roi_exclusion_mask as _roi_exclusion_mask
from hybrid.models import HAND_LANDMARKER
from hybrid.signal_utils import longest_bad_run
from hybrid.video_io import VideoReader

logger = get_logger(__name__)


@dataclass(frozen=True)
class FrameEgoMotion:
    frame_index: int
    timestamp_sec: float
    num_background_features: int
    num_matched: int
    num_inliers: int
    inlier_ratio: float
    translation_x: float
    translation_y: float
    rotation_rad: float
    scale: float
    transform_valid: bool
    confidence: float  # inlier_ratio when valid, else 0.0 -- "driven toward zero"


@dataclass
class EgoMotionVideoResult:
    video_id: str
    seed: int
    frames: list[FrameEgoMotion]  # length T, index-aligned with video frames; frame 0 is the identity reference
    unreliable_periods: list[tuple[int, int]]  # (start_idx, end_idx) into `frames`, inclusive
    longest_unreliable_sec: float


def _decompose_similarity(matrix: np.ndarray) -> tuple[float, float, float, float]:
    """2x3 similarity matrix [[a,b,tx],[c,d,ty]] -> (tx, ty, rotation_rad, scale)."""
    a, _b, tx = matrix[0]
    c, _d, ty = matrix[1]
    scale = math.hypot(a, c)
    rotation_rad = math.atan2(c, a)
    return float(tx), float(ty), float(rotation_rad), float(scale)


def _estimate_frame_pair(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    roi: RoiBox | None,
    em_config: EgoMotionConfig,
) -> tuple[int, int, int, float, np.ndarray | None]:
    """Returns (num_features, num_matched, num_inliers, inlier_ratio, M_or_None)."""
    height, width = prev_gray.shape
    mask = _roi_exclusion_mask(roi, width, height, em_config.roi_exclusion_dilate_px)

    features = cv2.goodFeaturesToTrack(
        prev_gray,
        maxCorners=em_config.max_features,
        qualityLevel=em_config.quality_level,
        minDistance=em_config.min_distance,
        mask=mask,
    )
    if features is None:
        return 0, 0, 0, 0.0, None
    if len(features) < 3:
        return len(features), 0, 0, 0.0, None

    tracked, status, _err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, features, None)
    status = status.reshape(-1).astype(bool)
    prev_pts = features[status].reshape(-1, 2)
    curr_pts = tracked[status].reshape(-1, 2)

    if len(prev_pts) < 3:
        return len(features), len(prev_pts), 0, 0.0, None

    matrix, inliers = cv2.estimateAffinePartial2D(
        prev_pts,
        curr_pts,
        method=cv2.RANSAC,
        ransacReprojThreshold=em_config.ransac_reproj_threshold,
    )
    if matrix is None:
        return len(features), len(prev_pts), 0, 0.0, None

    num_inliers = int(inliers.sum()) if inliers is not None else 0
    inlier_ratio = num_inliers / len(prev_pts)
    return len(features), len(prev_pts), num_inliers, inlier_ratio, matrix


def run_ego_motion_on_video(
    video_path, mediapipe_result: MediaPipeVideoResult, config: HybridConfig, video_id: str
) -> EgoMotionVideoResult:
    em_config = config.ego_motion
    seed = config.project.seed
    cv2.setRNGSeed(seed)  # reproducible RANSAC (CLAUDE.md: seed recorded with every run)

    roi_lookup = _build_roi_lookup(mediapipe_result.detections)

    frame_results: list[FrameEgoMotion] = []
    prev_gray: np.ndarray | None = None
    prev_index: int | None = None

    with VideoReader(video_path) as reader:
        for frame in reader:
            gray = cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY)

            if prev_gray is None:
                frame_results.append(
                    FrameEgoMotion(
                        frame_index=frame.index,
                        timestamp_sec=frame.timestamp_sec,
                        num_background_features=0,
                        num_matched=0,
                        num_inliers=0,
                        inlier_ratio=1.0,
                        translation_x=0.0,
                        translation_y=0.0,
                        rotation_rad=0.0,
                        scale=1.0,
                        transform_valid=True,
                        confidence=1.0,
                    )
                )
                prev_gray, prev_index = gray, frame.index
                continue

            roi = roi_lookup.get(prev_index)
            num_feat, num_matched, num_inliers, inlier_ratio, matrix = _estimate_frame_pair(
                prev_gray, gray, roi, em_config
            )

            if matrix is None:
                tx = ty = rotation_rad = 0.0
                scale = 1.0
                transform_valid = False
            else:
                tx, ty, rotation_rad, scale = _decompose_similarity(matrix)
                transform_valid = (
                    inlier_ratio >= em_config.min_inlier_ratio
                    and abs(math.degrees(rotation_rad)) <= em_config.max_rotation_deg
                    and em_config.min_scale <= scale <= em_config.max_scale
                )

            confidence = inlier_ratio if transform_valid else 0.0
            if not transform_valid:
                logger.warning(
                    "%s: unreliable ego-motion transform at frame %d (t=%.2fs): "
                    "inlier_ratio=%.2f, rotation=%.1fdeg, scale=%.2f, matched=%d",
                    video_id,
                    frame.index,
                    frame.timestamp_sec,
                    inlier_ratio,
                    math.degrees(rotation_rad),
                    scale,
                    num_matched,
                )

            frame_results.append(
                FrameEgoMotion(
                    frame_index=frame.index,
                    timestamp_sec=frame.timestamp_sec,
                    num_background_features=num_feat,
                    num_matched=num_matched,
                    num_inliers=num_inliers,
                    inlier_ratio=inlier_ratio,
                    translation_x=tx,
                    translation_y=ty,
                    rotation_rad=rotation_rad,
                    scale=scale,
                    transform_valid=transform_valid,
                    confidence=confidence,
                )
            )
            prev_gray, prev_index = gray, frame.index

    timestamps = np.array([f.timestamp_sec for f in frame_results])
    is_unreliable = np.array([not f.transform_valid for f in frame_results])
    longest_frames, longest_sec, unreliable_periods = longest_bad_run(is_unreliable, timestamps)

    if longest_sec > 0:
        logger.info(
            "%s: longest unreliable ego-motion span is %.2fs (%d frames)", video_id, longest_sec, longest_frames
        )
    if longest_sec > em_config.max_unreliable_span_sec:
        raise EgoMotionUnreliableError(
            f"{video_id}: longest unreliable ego-motion span ({longest_sec:.2f}s) exceeds "
            f"max_unreliable_span_sec={em_config.max_unreliable_span_sec}s"
        )

    mean_confidence = float(np.mean([f.confidence for f in frame_results]))
    logger.info(
        "%s: ego-motion estimated for %d frames, mean confidence %.2f, seed=%d",
        video_id,
        len(frame_results),
        mean_confidence,
        seed,
    )

    return EgoMotionVideoResult(
        video_id=video_id,
        seed=seed,
        frames=frame_results,
        unreliable_periods=unreliable_periods,
        longest_unreliable_sec=longest_sec,
    )


def _ego_motion_config_hash(config: HybridConfig) -> str:
    """Scoped to ego_motion + mediapipe sub-configs (+ pinned model checksum)
    since the ROI exclusion mask is derived from MediaPipe's output, plus the
    project seed (RANSAC reproducibility).
    """
    payload = (
        config.ego_motion.model_dump_json()
        + config.mediapipe.model_dump_json()
        + HAND_LANDMARKER.sha256
        + str(config.project.seed)
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def run_ego_motion_on_video_cached(
    video_path,
    mediapipe_result: MediaPipeVideoResult,
    config: HybridConfig,
    video_id: str,
    cache_manager: CacheManager,
) -> EgoMotionVideoResult:
    config_hash = _ego_motion_config_hash(config)
    key = cache_manager.make_key("ego_motion", video_id, config_hash)

    if config.caching.enabled and cache_manager.exists(key):
        logger.info("%s: loaded ego-motion result from cache (key=%s)", video_id, key)
        return cache_manager.load(key)

    result = run_ego_motion_on_video(video_path, mediapipe_result, config, video_id)
    if config.caching.enabled:
        cache_manager.save(key, result)
    return result
