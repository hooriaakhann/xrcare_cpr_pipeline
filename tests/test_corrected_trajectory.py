import math

import numpy as np
import pytest

from hybrid.corrected_trajectory import (
    TrajectoryAlignmentError,
    _compose,
    _cumulative_transforms,
    _similarity_as_complex,
    correct_tracker_trajectory,
)
from hybrid.cotracker_tracker import CoTrackerVideoResult
from hybrid.ego_motion import EgoMotionVideoResult, FrameEgoMotion

# ---------------------------------------------------------------------------
# Pure-logic tests
# ---------------------------------------------------------------------------


def test_similarity_as_complex_identity():
    a, b = _similarity_as_complex(0.0, 0.0, 0.0, 1.0)
    assert a == pytest.approx(1.0 + 0.0j)
    assert b == pytest.approx(0.0 + 0.0j)


def test_compose_with_identity_is_noop():
    a1, b1 = _similarity_as_complex(3.0, -2.0, 0.3, 1.1)
    id_a, id_b = _similarity_as_complex(0.0, 0.0, 0.0, 1.0)
    a2, b2 = _compose(a1, b1, id_a, id_b)
    assert a2 == pytest.approx(a1)
    assert b2 == pytest.approx(b1)


def _ego_result(frame_transforms, video_id="v"):
    frames = [
        FrameEgoMotion(
            frame_index=t,
            timestamp_sec=t * 0.1,
            num_background_features=10,
            num_matched=8,
            num_inliers=8,
            inlier_ratio=1.0,
            translation_x=tx,
            translation_y=ty,
            rotation_rad=rot,
            scale=scale,
            transform_valid=True,
            confidence=1.0,
        )
        for t, (tx, ty, rot, scale) in enumerate(frame_transforms)
    ]
    return EgoMotionVideoResult(
        video_id=video_id, seed=42, frames=frames, unreliable_periods=[], longest_unreliable_sec=0.0
    )


def test_cumulative_transforms_matches_manual_composition():
    frame_transforms = [(0.0, 0.0, 0.0, 1.0), (2.0, 1.0, 0.1, 1.02), (1.0, -1.0, -0.05, 0.98)]
    ego_result = _ego_result(frame_transforms)
    cum_a, cum_b = _cumulative_transforms(ego_result)

    assert cum_a[0] == pytest.approx(1.0 + 0.0j)
    assert cum_b[0] == pytest.approx(0.0 + 0.0j)

    # manual: C_1 = M_1, C_2 = M_2 . M_1 (apply M_1 first, then M_2)
    a1, b1 = _similarity_as_complex(*frame_transforms[1])
    assert cum_a[1] == pytest.approx(a1)
    assert cum_b[1] == pytest.approx(b1)

    a2, b2 = _similarity_as_complex(*frame_transforms[2])
    expected_a2, expected_b2 = _compose(a1, b1, a2, b2)
    assert cum_a[2] == pytest.approx(expected_a2)
    assert cum_b[2] == pytest.approx(expected_b2)


# ---------------------------------------------------------------------------
# correct_tracker_trajectory
# ---------------------------------------------------------------------------


def _ct_result(tracks_x, tracks_y, video_id="v"):
    t_len, n = tracks_x.shape
    visibility = np.ones((t_len, n), dtype=bool)
    return CoTrackerVideoResult(
        video_id=video_id,
        num_points=n,
        frame_indices=np.arange(t_len),
        timestamps_sec=np.arange(t_len) * 0.1,
        tracks_x=tracks_x,
        tracks_y=tracks_y,
        visibility=visibility,
        num_visible_per_frame=np.full(t_len, n),
        valid_ratio_per_frame=np.ones(t_len),
        reinit_events=[],
        track_loss_periods=[],
        longest_track_loss_sec=0.0,
        tracker_motion_y=tracks_y[:, 0],
    )


def test_correct_tracker_trajectory_noop_when_ego_motion_is_identity():
    num_frames = 5
    tracks_x = np.tile(np.arange(num_frames, dtype=float).reshape(-1, 1) * 3.0, (1, 2))
    tracks_y = np.tile(np.arange(num_frames, dtype=float).reshape(-1, 1) * -2.0, (1, 2))
    ct_result = _ct_result(tracks_x, tracks_y)
    ego_result = _ego_result([(0.0, 0.0, 0.0, 1.0)] * num_frames)

    result = correct_tracker_trajectory(ct_result, ego_result, "v")

    np.testing.assert_allclose(result.corrected_tracks_x, result.raw_tracks_x, atol=1e-9)
    np.testing.assert_allclose(result.corrected_tracks_y, result.raw_tracks_y, atol=1e-9)


def test_correct_tracker_trajectory_recovers_stationary_background_point():
    num_frames = 6
    rng = np.random.default_rng(1)
    frame_transforms = [(0.0, 0.0, 0.0, 1.0)]
    for _ in range(1, num_frames):
        frame_transforms.append(
            (rng.uniform(-5, 5), rng.uniform(-5, 5), rng.uniform(-0.05, 0.05), rng.uniform(0.95, 1.05))
        )

    # simulate a real-world-stationary point observed through each frame's
    # camera motion, independently of the module under test
    p = complex(100.0, 200.0)
    observed = [p]
    for t in range(1, num_frames):
        tx, ty, rot, scale = frame_transforms[t]
        a = scale * complex(math.cos(rot), math.sin(rot))
        b = complex(tx, ty)
        p = a * p + b
        observed.append(p)

    tracks_x = np.array([[z.real] for z in observed])
    tracks_y = np.array([[z.imag] for z in observed])
    ct_result = _ct_result(tracks_x, tracks_y)
    ego_result = _ego_result(frame_transforms)

    result = correct_tracker_trajectory(ct_result, ego_result, "v")

    np.testing.assert_allclose(result.corrected_tracks_x[:, 0], 100.0, atol=1e-6)
    np.testing.assert_allclose(result.corrected_tracks_y[:, 0], 200.0, atol=1e-6)


def test_correct_tracker_trajectory_mismatched_frames_raises():
    tracks_x = np.zeros((5, 1))
    tracks_y = np.zeros((5, 1))
    ct_result = _ct_result(tracks_x, tracks_y)
    ego_result = _ego_result([(0.0, 0.0, 0.0, 1.0)] * 4)  # different length

    with pytest.raises(TrajectoryAlignmentError):
        correct_tracker_trajectory(ct_result, ego_result, "v")


def test_correct_tracker_trajectory_passes_through_diagnostics():
    num_frames = 4
    tracks_x = np.zeros((num_frames, 1))
    tracks_y = np.arange(num_frames, dtype=float).reshape(-1, 1)
    ct_result = _ct_result(tracks_x, tracks_y)
    ego_result = _ego_result([(0.0, 0.0, 0.0, 1.0)] * num_frames)

    result = correct_tracker_trajectory(ct_result, ego_result, "v")

    np.testing.assert_array_equal(result.visibility, ct_result.visibility)
    np.testing.assert_array_equal(result.raw_tracker_motion_y, ct_result.tracker_motion_y)
    np.testing.assert_array_equal(result.ego_motion_confidence, np.ones(num_frames))
    np.testing.assert_array_equal(result.frame_indices, ct_result.frame_indices)
