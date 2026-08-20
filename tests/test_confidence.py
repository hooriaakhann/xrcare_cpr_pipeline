import numpy as np
import pytest

from hybrid.confidence import BranchConfidences, ConfidenceRangeError, collect_confidences
from hybrid.cotracker_tracker import CoTrackerVideoResult
from hybrid.ego_motion import EgoMotionVideoResult, FrameEgoMotion
from hybrid.estimators import AllEstimatesResult, AutocorrelationEstimate, CwtEstimate, FftEstimate, PeakEstimate
from hybrid.mediapipe_roi import MediaPipeVideoResult
from hybrid.optical_flow import FrameOpticalFlow, OpticalFlowVideoResult
from hybrid.repnet_branch import RepNetResult


def _mediapipe_result(detection_rate, video_id="v"):
    return MediaPipeVideoResult(
        video_id=video_id,
        frame_width=32,
        frame_height=32,
        detections=[],
        detection_rate=detection_rate,
        longest_gap_frames=0,
        longest_gap_sec=0.0,
        model_sha256="deadbeef",
    )


def _cotracker_result(valid_ratios, video_id="v"):
    n = len(valid_ratios)
    return CoTrackerVideoResult(
        video_id=video_id,
        num_points=1,
        frame_indices=np.arange(n),
        timestamps_sec=np.arange(n) * 0.1,
        tracks_x=np.zeros((n, 1)),
        tracks_y=np.zeros((n, 1)),
        visibility=np.ones((n, 1), dtype=bool),
        num_visible_per_frame=np.ones(n, dtype=int),
        valid_ratio_per_frame=np.array(valid_ratios),
        reinit_events=[],
        track_loss_periods=[],
        longest_track_loss_sec=0.0,
        tracker_motion_y=np.zeros(n),
    )


def _ego_motion_result(confidences, video_id="v"):
    frames = [
        FrameEgoMotion(
            frame_index=i,
            timestamp_sec=i * 0.1,
            num_background_features=10,
            num_matched=8,
            num_inliers=8,
            inlier_ratio=c,
            translation_x=0.0,
            translation_y=0.0,
            rotation_rad=0.0,
            scale=1.0,
            transform_valid=c > 0,
            confidence=c,
        )
        for i, c in enumerate(confidences)
    ]
    return EgoMotionVideoResult(
        video_id=video_id, seed=42, frames=frames, unreliable_periods=[], longest_unreliable_sec=0.0
    )


def _optical_flow_result(valid_flags, video_id="v"):
    frames = [
        FrameOpticalFlow(
            frame_index=i,
            timestamp_sec=i * 0.1,
            foreground_flow_y=0.0,
            background_flow_y=0.0,
            residual_flow_y=0.0,
            flow_magnitude=0.0,
            valid=v,
        )
        for i, v in enumerate(valid_flags)
    ]
    return OpticalFlowVideoResult(
        video_id=video_id,
        frames=frames,
        flow_motion_y=np.zeros(len(valid_flags)),
        unstable_periods=[],
        longest_unstable_sec=0.0,
    )


def _estimates(cwt_conf, autocorr_conf, fft_conf, peaks_conf, video_id="v"):
    return AllEstimatesResult(
        video_id=video_id,
        cwt=CwtEstimate(dominant_freq_hz=1.5, cpm=90.0, confidence=cwt_conf),
        autocorrelation=AutocorrelationEstimate(dominant_lag_sec=0.6, cpm=90.0, confidence=autocorr_conf),
        fft=FftEstimate(dominant_freq_hz=1.5, cpm=90.0, confidence=fft_conf),
        peaks=PeakEstimate(
            peak_timestamps_sec=np.array([0.0, 0.6]),
            num_peaks=2,
            median_inter_peak_interval_sec=0.6,
            cpm=90.0,
            confidence=peaks_conf,
        ),
    )


def _repnet_result(confidence, video_id="v"):
    return RepNetResult(
        video_id=video_id,
        cpm=90.0,
        confidence=confidence,
        pred_period_frames=18.0,
        chosen_stride=1,
        fps=30.0,
        num_frames=100,
        reason=None,
    )


def test_collect_confidences_happy_path():
    result = collect_confidences(
        mediapipe_result=_mediapipe_result(0.95),
        cotracker_result=_cotracker_result([0.9, 1.0, 0.8]),
        ego_motion_result=_ego_motion_result([1.0, 0.5, 0.7]),
        optical_flow_result=_optical_flow_result([True, True, False]),
        estimates=_estimates(0.06, 0.9, 0.85, 0.92),
        repnet_result=_repnet_result(0.66),
        video_id="v",
    )

    assert isinstance(result, BranchConfidences)
    assert result.mediapipe == pytest.approx(0.95)
    assert result.cotracker == pytest.approx((0.9 + 1.0 + 0.8) / 3)
    assert result.ego_motion == pytest.approx((1.0 + 0.5 + 0.7) / 3)
    assert result.optical_flow == pytest.approx(2 / 3)
    assert result.cwt == pytest.approx(0.06)
    assert result.autocorrelation == pytest.approx(0.9)
    assert result.fft == pytest.approx(0.85)
    assert result.peaks == pytest.approx(0.92)
    assert result.repnet == pytest.approx(0.66)


def test_as_dict_matches_fields():
    result = collect_confidences(
        mediapipe_result=_mediapipe_result(1.0),
        cotracker_result=_cotracker_result([1.0]),
        ego_motion_result=_ego_motion_result([1.0]),
        optical_flow_result=_optical_flow_result([True]),
        estimates=_estimates(1.0, 1.0, 1.0, 1.0),
        repnet_result=_repnet_result(1.0),
        video_id="v",
    )
    d = result.as_dict()
    assert set(d) == {
        "mediapipe",
        "cotracker",
        "ego_motion",
        "optical_flow",
        "cwt",
        "autocorrelation",
        "fft",
        "peaks",
        "repnet",
    }
    assert all(v == pytest.approx(1.0) for v in d.values())


def test_collect_confidences_raises_on_out_of_range_value():
    with pytest.raises(ConfidenceRangeError):
        collect_confidences(
            mediapipe_result=_mediapipe_result(1.5),  # invalid: > 1.0
            cotracker_result=_cotracker_result([1.0]),
            ego_motion_result=_ego_motion_result([1.0]),
            optical_flow_result=_optical_flow_result([True]),
            estimates=_estimates(1.0, 1.0, 1.0, 1.0),
            repnet_result=_repnet_result(1.0),
            video_id="v",
        )


def test_collect_confidences_raises_on_negative_value():
    with pytest.raises(ConfidenceRangeError):
        collect_confidences(
            mediapipe_result=_mediapipe_result(1.0),
            cotracker_result=_cotracker_result([1.0]),
            ego_motion_result=_ego_motion_result([1.0]),
            optical_flow_result=_optical_flow_result([True]),
            estimates=_estimates(1.0, 1.0, 1.0, 1.0),
            repnet_result=_repnet_result(-0.1),  # invalid: < 0.0
            video_id="v",
        )
