"""Ego-Motion Corrected CoTracker Trajectories (Phase 5).

Removes the estimated camera/head motion (Phase 4) from the raw CoTracker
point trajectories (Phase 3), so what remains approximates the CPR
compression motion alone:

    observed hand trajectory = CPR movement + camera movement
    corrected trajectory     ~= CPR movement

Each frame's ego-motion transform (Phase 4) maps "frame t-1 background
coordinates" to "frame t background coordinates" for a similarity transform
(translation + rotation + uniform scale). Composing these frame by frame
gives a cumulative transform C_t from frame 0's coordinate frame to frame
t's; a tracked point's position is corrected by applying C_t^-1 -- i.e.
re-expressing every frame's observation in frame 0's fixed reference frame.
If a point were truly stationary in the real world, it would then read as
constant regardless of how much the camera moved; the residual is what's
left of the point's actual motion.

Represented as complex numbers (z = x + iy, transform z' = a*z + b with
a = scale*e^(i*rotation)): composition and inversion are then plain complex
arithmetic, which stays numerically well-behaved over long chains (500+
frames) without explicit matrix bookkeeping.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from hybrid.cotracker_tracker import CoTrackerVideoResult
from hybrid.ego_motion import EgoMotionVideoResult
from hybrid.exceptions import HybridError
from hybrid.logging_config import get_logger
from hybrid.signal_utils import median_visible

logger = get_logger(__name__)


class TrajectoryAlignmentError(HybridError):
    """CoTracker and ego-motion results don't cover the same frames -- cannot
    align them for correction."""


@dataclass
class CorrectedTrackerResult:
    video_id: str
    frame_indices: np.ndarray  # (T,)
    timestamps_sec: np.ndarray  # (T,)
    raw_tracks_x: np.ndarray  # (T, N)
    raw_tracks_y: np.ndarray  # (T, N)
    corrected_tracks_x: np.ndarray  # (T, N)
    corrected_tracks_y: np.ndarray  # (T, N)
    visibility: np.ndarray  # (T, N) bool, passthrough from CoTracker
    raw_tracker_motion_y: np.ndarray  # (T,) passthrough from Phase 3
    corrected_tracker_motion_y: np.ndarray  # (T,) median Y of visible corrected points
    ego_motion_confidence: np.ndarray  # (T,) passthrough from Phase 4


def _similarity_as_complex(tx: float, ty: float, rotation_rad: float, scale: float) -> tuple[complex, complex]:
    a = scale * complex(math.cos(rotation_rad), math.sin(rotation_rad))
    b = complex(tx, ty)
    return a, b


def _compose(a1: complex, b1: complex, a2: complex, b2: complex) -> tuple[complex, complex]:
    """Composite transform of "apply 1 then 2": z -> a2*(a1*z + b1) + b2."""
    return a2 * a1, a2 * b1 + b2


def _cumulative_transforms(ego_result: EgoMotionVideoResult) -> tuple[np.ndarray, np.ndarray]:
    """(cum_a, cum_b): complex coefficients mapping frame-0 coordinates to
    frame-t coordinates, for every t.
    """
    t_len = len(ego_result.frames)
    cum_a = np.empty(t_len, dtype=complex)
    cum_b = np.empty(t_len, dtype=complex)
    cum_a[0], cum_b[0] = 1.0 + 0.0j, 0.0 + 0.0j
    for t in range(1, t_len):
        f = ego_result.frames[t]
        a_t, b_t = _similarity_as_complex(f.translation_x, f.translation_y, f.rotation_rad, f.scale)
        cum_a[t], cum_b[t] = _compose(cum_a[t - 1], cum_b[t - 1], a_t, b_t)
    return cum_a, cum_b


def correct_tracker_trajectory(
    ct_result: CoTrackerVideoResult, ego_result: EgoMotionVideoResult, video_id: str
) -> CorrectedTrackerResult:
    ego_frame_indices = np.array([f.frame_index for f in ego_result.frames])
    if not np.array_equal(ct_result.frame_indices, ego_frame_indices):
        raise TrajectoryAlignmentError(
            f"{video_id}: CoTracker ({len(ct_result.frame_indices)} frames) and ego-motion "
            f"({len(ego_frame_indices)} frames) don't cover the same frame indices -- cannot align for correction"
        )

    cum_a, cum_b = _cumulative_transforms(ego_result)

    observed = ct_result.tracks_x + 1j * ct_result.tracks_y  # (T, N) complex
    corrected = (observed - cum_b[:, None]) / cum_a[:, None]
    corrected_x = corrected.real
    corrected_y = corrected.imag

    corrected_motion_y = median_visible(corrected_y, ct_result.visibility)
    ego_confidence = np.array([f.confidence for f in ego_result.frames])

    final_cum_scale = abs(cum_a[-1])
    if final_cum_scale > 3.0 or final_cum_scale < 1.0 / 3.0:
        logger.warning(
            "%s: cumulative ego-motion scale drifted to %.2fx by the last frame -- "
            "possible estimation drift over this video's length",
            video_id,
            final_cum_scale,
        )

    logger.info(
        "%s: applied ego-motion correction to %d points across %d frames",
        video_id,
        ct_result.tracks_x.shape[1],
        len(ct_result.frame_indices),
    )

    return CorrectedTrackerResult(
        video_id=video_id,
        frame_indices=ct_result.frame_indices,
        timestamps_sec=ct_result.timestamps_sec,
        raw_tracks_x=ct_result.tracks_x,
        raw_tracks_y=ct_result.tracks_y,
        corrected_tracks_x=corrected_x,
        corrected_tracks_y=corrected_y,
        visibility=ct_result.visibility,
        raw_tracker_motion_y=ct_result.tracker_motion_y,
        corrected_tracker_motion_y=corrected_motion_y,
        ego_motion_confidence=ego_confidence,
    )
