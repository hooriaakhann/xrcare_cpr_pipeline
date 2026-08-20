import numpy as np
import pytest

from hybrid.corrected_trajectory import CorrectedTrackerResult
from hybrid.motion_wave import WaveformAlignmentError, _robust_standardize, generate_motion_wave
from hybrid.optical_flow import FrameOpticalFlow

# ---------------------------------------------------------------------------
# _robust_standardize
# ---------------------------------------------------------------------------


def test_robust_standardize_centers_and_scales():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    out = _robust_standardize(x)
    assert np.median(out) == pytest.approx(0.0, abs=1e-9)
    # MAD of [1..5] around median 3 is median(|[-2,-1,0,1,2]|) = 1 -> scale = 1.4826
    np.testing.assert_allclose(out, (x - 3.0) / 1.4826, atol=1e-9)


def test_robust_standardize_constant_array_is_zero_not_nan_or_inf():
    x = np.full(5, 7.0)
    out = _robust_standardize(x)
    assert np.all(np.isfinite(out))
    np.testing.assert_array_equal(out, np.zeros(5))


def test_robust_standardize_preserves_nan():
    x = np.array([1.0, np.nan, 3.0, 5.0])
    out = _robust_standardize(x)
    assert np.isnan(out[1])
    assert np.isfinite(out[0])
    assert np.isfinite(out[2])
    assert np.isfinite(out[3])


# ---------------------------------------------------------------------------
# generate_motion_wave
# ---------------------------------------------------------------------------


def _tracker_result(num_frames, num_points, visibility, ego_confidence, motion_y, video_id="v"):
    tracks = np.zeros((num_frames, num_points))
    return CorrectedTrackerResult(
        video_id=video_id,
        frame_indices=np.arange(num_frames),
        timestamps_sec=np.arange(num_frames) * 0.1,
        raw_tracks_x=tracks,
        raw_tracks_y=tracks,
        corrected_tracks_x=tracks,
        corrected_tracks_y=tracks,
        visibility=visibility,
        raw_tracker_motion_y=motion_y,
        corrected_tracker_motion_y=motion_y,
        ego_motion_confidence=ego_confidence,
    )


def _flow_result(num_frames, valid_flags, motion_y, video_id="v"):
    frames = [
        FrameOpticalFlow(
            frame_index=i,
            timestamp_sec=i * 0.1,
            foreground_flow_y=0.0,
            background_flow_y=0.0,
            residual_flow_y=float(motion_y[i]),
            flow_magnitude=1.0,
            valid=bool(valid_flags[i]),
        )
        for i in range(num_frames)
    ]
    from hybrid.optical_flow import OpticalFlowVideoResult

    return OpticalFlowVideoResult(
        video_id=video_id,
        frames=frames,
        flow_motion_y=np.asarray(motion_y, dtype=float),
        unstable_periods=[],
        longest_unstable_sec=0.0,
    )


def test_generate_motion_wave_mismatched_frames_raises():
    n = 5
    visibility = np.ones((n, 2), dtype=bool)
    tracker = _tracker_result(n, 2, visibility, np.ones(n), np.arange(n, dtype=float))
    flow = _flow_result(4, np.ones(4, dtype=bool), np.arange(4, dtype=float))

    with pytest.raises(WaveformAlignmentError):
        generate_motion_wave(tracker, flow, "v")


def test_generate_motion_wave_full_weight_is_average_of_normalized_signals():
    n = 6
    rng = np.random.default_rng(0)
    tracker_motion = rng.normal(size=n) * 10 + 100
    flow_motion = rng.normal(size=n) * 2

    visibility = np.ones((n, 3), dtype=bool)
    ego_conf = np.ones(n)
    valid_flags = np.ones(n, dtype=bool)

    tracker = _tracker_result(n, 3, visibility, ego_conf, tracker_motion)
    flow = _flow_result(n, valid_flags, flow_motion)

    result = generate_motion_wave(tracker, flow, "v")

    np.testing.assert_allclose(result.tracker_weight, np.ones(n))
    np.testing.assert_allclose(result.flow_weight, np.ones(n))
    expected = (result.normalized_tracker_signal + result.normalized_flow_signal) / 2.0
    np.testing.assert_allclose(result.motion_wave, expected, atol=1e-9)


def test_generate_motion_wave_tracker_only_when_flow_invalid():
    n = 5
    visibility = np.ones((n, 2), dtype=bool)
    tracker = _tracker_result(n, 2, visibility, np.ones(n), np.linspace(0, 10, n))
    flow = _flow_result(n, np.zeros(n, dtype=bool), np.linspace(0, 100, n))

    result = generate_motion_wave(tracker, flow, "v")

    np.testing.assert_array_equal(result.flow_weight, np.zeros(n))
    np.testing.assert_allclose(result.motion_wave, result.normalized_tracker_signal, atol=1e-9)


def test_generate_motion_wave_flow_only_when_tracker_invisible():
    n = 5
    visibility = np.zeros((n, 2), dtype=bool)  # nothing visible -> tracker_weight = 0 everywhere
    tracker = _tracker_result(n, 2, visibility, np.ones(n), np.full(n, np.nan))
    flow = _flow_result(n, np.ones(n, dtype=bool), np.linspace(-5, 5, n))

    result = generate_motion_wave(tracker, flow, "v")

    np.testing.assert_array_equal(result.tracker_weight, np.zeros(n))
    np.testing.assert_allclose(result.motion_wave, result.normalized_flow_signal, atol=1e-9)


def test_generate_motion_wave_nan_when_both_weights_zero():
    n = 4
    visibility = np.zeros((n, 2), dtype=bool)
    tracker = _tracker_result(n, 2, visibility, np.ones(n), np.full(n, np.nan))
    flow = _flow_result(n, np.zeros(n, dtype=bool), np.zeros(n))

    result = generate_motion_wave(tracker, flow, "v")

    assert np.all(np.isnan(result.motion_wave))


def test_generate_motion_wave_ego_motion_confidence_reduces_tracker_weight():
    n = 3
    visibility = np.ones((n, 2), dtype=bool)
    ego_conf = np.array([1.0, 0.0, 0.5])
    tracker = _tracker_result(n, 2, visibility, ego_conf, np.array([1.0, 2.0, 3.0]))
    flow = _flow_result(n, np.zeros(n, dtype=bool), np.zeros(n))  # flow always invalid -> isolates tracker weight

    result = generate_motion_wave(tracker, flow, "v")

    np.testing.assert_allclose(result.tracker_weight, ego_conf)


def test_generate_motion_wave_passes_through_raw_signals():
    n = 4
    visibility = np.ones((n, 2), dtype=bool)
    tracker_motion = np.array([1.0, 2.0, 3.0, 4.0])
    flow_motion = np.array([0.1, 0.2, 0.3, 0.4])
    tracker = _tracker_result(n, 2, visibility, np.ones(n), tracker_motion)
    flow = _flow_result(n, np.ones(n, dtype=bool), flow_motion)

    result = generate_motion_wave(tracker, flow, "v")

    np.testing.assert_array_equal(result.raw_tracker_signal, tracker_motion)
    np.testing.assert_array_equal(result.raw_flow_signal, flow_motion)
    np.testing.assert_array_equal(result.frame_indices, tracker.frame_indices)
