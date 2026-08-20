"""Motion Waveform Generation (Phase 7).

Combines the two CPR motion signals built so far -- corrected_tracker_
motion_y(t) (Phase 5) and flow_motion_y(t) (Phase 6) -- into one fused
`motion_wave(t)`. The two signals live on very different scales (tracker:
pixel-position deltas over hundreds of px; flow: per-frame pixel
displacement, single digits), so they are never simply averaged raw. Each
is first robustly standardized (median-centered, MAD-scaled -- robust to
the occasional outlier frame, unlike a mean/std z-score), then combined with
a confidence-weighted average: strong CoTracker visibility and a reliable
ego-motion correction raise the tracker weight; a flagged-unstable optical
flow frame drops the flow weight to zero for that frame. Raw signals,
normalized signals, per-frame weights, and the fused wave are all kept on
the result for later debugging, per spec.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hybrid.corrected_trajectory import CorrectedTrackerResult
from hybrid.exceptions import HybridError
from hybrid.logging_config import get_logger
from hybrid.optical_flow import OpticalFlowVideoResult

logger = get_logger(__name__)

# Consistency constant so a MAD-based scale is comparable to a std-dev for
# an approximately-normal signal (standard robust-statistics convention).
_MAD_TO_STD = 1.4826
_DEGENERATE_SCALE_EPS = 1e-9
_ZERO_WEIGHT_EPS = 1e-9


class WaveformAlignmentError(HybridError):
    """Tracker and optical-flow results don't cover the same frames -- cannot
    fuse them."""


@dataclass
class MotionWaveResult:
    video_id: str
    frame_indices: np.ndarray  # (T,)
    timestamps_sec: np.ndarray  # (T,)
    raw_tracker_signal: np.ndarray  # (T,) = corrected_tracker_motion_y, passthrough
    raw_flow_signal: np.ndarray  # (T,) = flow_motion_y, passthrough
    normalized_tracker_signal: np.ndarray  # (T,)
    normalized_flow_signal: np.ndarray  # (T,)
    tracker_weight: np.ndarray  # (T,) = mean point visibility * ego-motion confidence
    flow_weight: np.ndarray  # (T,) = 1.0 if that frame's flow was valid, else 0.0
    motion_wave: np.ndarray  # (T,) the fused signal; NaN where both weights are ~0


def _robust_standardize(x: np.ndarray) -> np.ndarray:
    """Median-centered, MAD-scaled. NaNs pass through as NaN. A degenerate
    (constant or all-NaN) signal normalizes to all-zero rather than
    dividing by zero.
    """
    if np.all(np.isnan(x)):
        return np.full_like(x, np.nan)

    median = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - median))
    scale = _MAD_TO_STD * mad
    if not np.isfinite(scale) or scale < _DEGENERATE_SCALE_EPS:
        return np.where(np.isnan(x), np.nan, 0.0)
    return (x - median) / scale


def generate_motion_wave(
    tracker_result: CorrectedTrackerResult, flow_result: OpticalFlowVideoResult, video_id: str
) -> MotionWaveResult:
    flow_frame_indices = np.array([f.frame_index for f in flow_result.frames])
    if not np.array_equal(tracker_result.frame_indices, flow_frame_indices):
        raise WaveformAlignmentError(
            f"{video_id}: tracker ({len(tracker_result.frame_indices)} frames) and optical-flow "
            f"({len(flow_frame_indices)} frames) results don't cover the same frame indices -- cannot fuse"
        )

    raw_tracker = tracker_result.corrected_tracker_motion_y
    raw_flow = flow_result.flow_motion_y
    normalized_tracker = _robust_standardize(raw_tracker)
    normalized_flow = _robust_standardize(raw_flow)

    # Tracker weight needs both good CoTracker visibility AND a reliable
    # ego-motion correction to trust the corrected trajectory at that frame.
    visible_ratio = tracker_result.visibility.mean(axis=1)
    tracker_weight = visible_ratio * tracker_result.ego_motion_confidence

    flow_weight = np.array([1.0 if f.valid else 0.0 for f in flow_result.frames])

    safe_tracker = np.nan_to_num(normalized_tracker, nan=0.0)
    safe_flow = np.nan_to_num(normalized_flow, nan=0.0)

    total_weight = tracker_weight + flow_weight
    with np.errstate(invalid="ignore", divide="ignore"):
        motion_wave = (tracker_weight * safe_tracker + flow_weight * safe_flow) / total_weight
    motion_wave = np.where(total_weight > _ZERO_WEIGHT_EPS, motion_wave, np.nan)

    logger.info(
        "%s: fused motion wave over %d frames, mean tracker_weight=%.3f mean flow_weight=%.3f, %d NaN frame(s)",
        video_id,
        len(motion_wave),
        float(np.mean(tracker_weight)),
        float(np.mean(flow_weight)),
        int(np.isnan(motion_wave).sum()),
    )

    return MotionWaveResult(
        video_id=video_id,
        frame_indices=tracker_result.frame_indices,
        timestamps_sec=tracker_result.timestamps_sec,
        raw_tracker_signal=raw_tracker,
        raw_flow_signal=raw_flow,
        normalized_tracker_signal=normalized_tracker,
        normalized_flow_signal=normalized_flow,
        tracker_weight=tracker_weight,
        flow_weight=flow_weight,
        motion_wave=motion_wave,
    )
