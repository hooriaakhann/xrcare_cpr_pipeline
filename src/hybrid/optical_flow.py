"""Farneback Optical Flow (Phase 6).

Dense optical flow between consecutive frames, split into a CPR-foreground
reading and a background reading (camera-motion context) -- never a single
whole-frame mean, which the spec explicitly rules out. The residual
(foreground - background) is this branch's per-frame CPR motion signal,
`flow_motion_y(t)`: analogous to what Phase 5 computes from CoTracker, but
from dense flow instead of sparse point tracks, and needing no cumulative
composition since flow is already a frame-to-frame differential, not an
absolute position.

The foreground reading is the median Y-flow of only the top-moving pixels
in the ROI (by magnitude, `motion_percentile`), not a plain median over the
whole ROI. The Phase 2 ROI is deliberately generous (hand + wrist + forearm
+ margin) so it can seed CoTracker well, but that means most of its pixels
are static skin/background at any instant -- only whatever part of the hand
is actually mid-stroke carries the compression signal. A plain median is
dominated by that static majority: verified against real footage that it
stays pinned within +/-0.05px for an entire video, while CoTracker's
already-validated tracks show real ~8px mean / 29px max frame-to-frame
motion over the same frames. Restricting to the top-moving pixels before
taking the median recovers a clean signal that tracks the compression cycle.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import cv2
import numpy as np

from hybrid.caching import CacheManager
from hybrid.config import HybridConfig, OpticalFlowConfig
from hybrid.exceptions import OpticalFlowUnstableError
from hybrid.logging_config import get_logger
from hybrid.mediapipe_roi import MediaPipeVideoResult, RoiBox, build_roi_lookup, roi_exclusion_mask
from hybrid.models import HAND_LANDMARKER
from hybrid.signal_utils import longest_bad_run
from hybrid.video_io import VideoReader

logger = get_logger(__name__)


@dataclass(frozen=True)
class FrameOpticalFlow:
    frame_index: int
    timestamp_sec: float
    foreground_flow_y: float  # median vertical flow within the ROI
    background_flow_y: float  # median vertical flow outside the ROI
    residual_flow_y: float  # foreground - background: the CPR-only signal
    flow_magnitude: float  # mean |flow| within the ROI (diagnostic)
    valid: bool


@dataclass
class OpticalFlowVideoResult:
    video_id: str
    frames: list[FrameOpticalFlow]  # length T; frame 0 has all-zero, valid=True (no previous frame)
    flow_motion_y: np.ndarray  # (T,) residual_flow_y per frame -- the branch's output waveform
    unstable_periods: list[tuple[int, int]]
    longest_unstable_sec: float


def _identity_frame(frame_index: int, timestamp_sec: float) -> FrameOpticalFlow:
    return FrameOpticalFlow(
        frame_index=frame_index,
        timestamp_sec=timestamp_sec,
        foreground_flow_y=0.0,
        background_flow_y=0.0,
        residual_flow_y=0.0,
        flow_magnitude=0.0,
        valid=True,
    )


def _compute_frame_flow(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    roi: RoiBox | None,
    of_config: OpticalFlowConfig,
) -> tuple[float, float, float, float, bool]:
    """Returns (foreground_flow_y, background_flow_y, residual_flow_y, flow_magnitude, valid)."""
    height, width = prev_gray.shape
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray,
        curr_gray,
        None,
        of_config.pyr_scale,
        of_config.levels,
        of_config.winsize,
        of_config.iterations,
        of_config.poly_n,
        of_config.poly_sigma,
        0,
    )
    flow_x, flow_y = flow[..., 0], flow[..., 1]

    if roi is not None:
        fg_mask = np.zeros((height, width), dtype=bool)
        fg_mask[roi.y_min : roi.y_max, roi.x_min : roi.x_max] = True
    else:
        fg_mask = np.zeros((height, width), dtype=bool)
    bg_mask = roi_exclusion_mask(roi, width, height, dilate_px=0) > 0

    if not fg_mask.any() or not bg_mask.any():
        return 0.0, 0.0, 0.0, 0.0, False

    # The ROI is deliberately generous (hand + wrist + forearm + margin,
    # Phase 2), so most of its pixels are static skin/background at any
    # instant -- only a minority (whatever part of the hand is actually
    # mid-stroke) carries the real compression signal. A plain median over
    # the whole region is dominated by that static majority and stays
    # pinned near zero throughout a video (verified against real footage:
    # CoTracker's already-validated frame-to-frame motion showed a real
    # ~8px mean / 29px max displacement here, while a plain-median flow
    # reading never left +/-0.04px). Restricting to the top
    # `100 - motion_percentile`% most-moving pixels by magnitude before
    # taking the median recovers a clean signal that tracks the compression
    # cycle (also verified against real footage -- a clean oscillation with
    # the right period, where the plain median was flat).
    fg_flow_x, fg_flow_y = flow_x[fg_mask], flow_y[fg_mask]
    fg_magnitude_per_px = np.hypot(fg_flow_x, fg_flow_y)
    motion_thresh = np.percentile(fg_magnitude_per_px, of_config.motion_percentile)
    moving = fg_magnitude_per_px >= motion_thresh

    foreground_flow_y = float(np.median(fg_flow_y[moving]))
    background_flow_y = float(np.median(flow_y[bg_mask]))
    residual_flow_y = foreground_flow_y - background_flow_y
    flow_magnitude = float(np.mean(fg_magnitude_per_px[moving]))

    valid = flow_magnitude <= of_config.max_flow_magnitude
    return foreground_flow_y, background_flow_y, residual_flow_y, flow_magnitude, valid


def run_optical_flow_on_video(
    video_path, mediapipe_result: MediaPipeVideoResult, config: HybridConfig, video_id: str
) -> OpticalFlowVideoResult:
    of_config = config.optical_flow
    roi_lookup = build_roi_lookup(mediapipe_result.detections)

    frame_results: list[FrameOpticalFlow] = []
    prev_gray: np.ndarray | None = None
    prev_index: int | None = None

    with VideoReader(video_path) as reader:
        for frame in reader:
            gray = cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY)

            if prev_gray is None:
                frame_results.append(_identity_frame(frame.index, frame.timestamp_sec))
                prev_gray, prev_index = gray, frame.index
                continue

            roi = roi_lookup.get(prev_index)
            fg_y, bg_y, residual_y, magnitude, valid = _compute_frame_flow(prev_gray, gray, roi, of_config)

            if not valid:
                logger.warning(
                    "%s: unstable optical flow at frame %d (t=%.2fs): magnitude=%.1fpx (max=%.1fpx)",
                    video_id,
                    frame.index,
                    frame.timestamp_sec,
                    magnitude,
                    of_config.max_flow_magnitude,
                )

            frame_results.append(
                FrameOpticalFlow(
                    frame_index=frame.index,
                    timestamp_sec=frame.timestamp_sec,
                    foreground_flow_y=fg_y,
                    background_flow_y=bg_y,
                    residual_flow_y=residual_y,
                    flow_magnitude=magnitude,
                    valid=valid,
                )
            )
            prev_gray, prev_index = gray, frame.index

    timestamps = np.array([f.timestamp_sec for f in frame_results])
    is_unstable = np.array([not f.valid for f in frame_results])
    longest_frames, longest_sec, unstable_periods = longest_bad_run(is_unstable, timestamps)

    if longest_sec > 0:
        logger.info(
            "%s: longest unstable optical-flow span is %.2fs (%d frames)", video_id, longest_sec, longest_frames
        )
    if longest_sec > of_config.max_unstable_span_sec:
        raise OpticalFlowUnstableError(
            f"{video_id}: longest unstable optical-flow span ({longest_sec:.2f}s) exceeds "
            f"max_unstable_span_sec={of_config.max_unstable_span_sec}s"
        )

    flow_motion_y = np.array([f.residual_flow_y for f in frame_results])
    logger.info(
        "%s: optical flow computed for %d frames, mean |residual_flow_y|=%.2fpx",
        video_id,
        len(frame_results),
        float(np.mean(np.abs(flow_motion_y))),
    )

    return OpticalFlowVideoResult(
        video_id=video_id,
        frames=frame_results,
        flow_motion_y=flow_motion_y,
        unstable_periods=unstable_periods,
        longest_unstable_sec=longest_sec,
    )


def _optical_flow_config_hash(config: HybridConfig) -> str:
    payload = config.optical_flow.model_dump_json() + config.mediapipe.model_dump_json() + HAND_LANDMARKER.sha256
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def run_optical_flow_on_video_cached(
    video_path,
    mediapipe_result: MediaPipeVideoResult,
    config: HybridConfig,
    video_id: str,
    cache_manager: CacheManager,
) -> OpticalFlowVideoResult:
    config_hash = _optical_flow_config_hash(config)
    key = cache_manager.make_key("optical_flow", video_id, config_hash)

    if config.caching.enabled and cache_manager.exists(key):
        logger.info("%s: loaded optical-flow result from cache (key=%s)", video_id, key)
        return cache_manager.load(key)

    result = run_optical_flow_on_video(video_path, mediapipe_result, config, video_id)
    if config.caching.enabled:
        cache_manager.save(key, result)
    return result
