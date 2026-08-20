import numpy as np
import pytest

from hybrid.ablation import (
    AblationVideoResult,
    _cpm_from_signal,
    _mediapipe_wrist_signal,
    _null_repnet_result,
    run_ablations_on_video,
    summarize_ablations,
    wilcoxon_hybrid_vs_best_ablation,
)
from hybrid.caching import CacheManager
from hybrid.config import HybridConfig
from hybrid.cotracker_tracker import CoTrackerVideoResult
from hybrid.dataset import DevVideo, GroundTruth
from hybrid.ego_motion import EgoMotionVideoResult, FrameEgoMotion
from hybrid.evaluation import VideoEvaluationResult
from hybrid.mediapipe_roi import HandDetection, MediaPipeVideoResult, RoiBox
from hybrid.optical_flow import FrameOpticalFlow, OpticalFlowVideoResult
from hybrid.repnet_branch import RepNetResult

FS = 30.0
DURATION_SEC = 30.0
TRUE_CPM = 100.0
TRUE_FREQ_HZ = TRUE_CPM / 60.0


def _sine(amplitude=100.0, offset=0.0):
    t = np.arange(0, DURATION_SEC, 1.0 / FS)
    return t, offset + amplitude * np.sin(2 * np.pi * TRUE_FREQ_HZ * t)


# ---------------------------------------------------------------------------
# Pure-logic tests
# ---------------------------------------------------------------------------


def test_mediapipe_wrist_signal_nan_where_undetected():
    detections = [
        HandDetection(frame_index=0, timestamp_sec=0.0, detected=True, wrist_xy=(10.0, 20.0)),
        HandDetection(frame_index=1, timestamp_sec=0.1, detected=False),
        HandDetection(frame_index=2, timestamp_sec=0.2, detected=True, wrist_xy=(12.0, 25.0)),
    ]
    mp_result = MediaPipeVideoResult(
        video_id="v",
        frame_width=100,
        frame_height=100,
        detections=detections,
        detection_rate=2 / 3,
        longest_gap_frames=1,
        longest_gap_sec=0.1,
        model_sha256="deadbeef",
    )
    t, wrist_y = _mediapipe_wrist_signal(mp_result)
    np.testing.assert_array_equal(t, [0.0, 0.1, 0.2])
    assert wrist_y[0] == pytest.approx(20.0)
    assert np.isnan(wrist_y[1])
    assert wrist_y[2] == pytest.approx(25.0)


def test_null_repnet_result_has_no_cpm():
    result = _null_repnet_result("v")
    assert result.cpm is None
    assert result.confidence == 0.0


def test_cpm_from_signal_recovers_known_frequency():
    t, signal = _sine()
    config = HybridConfig()
    cpm = _cpm_from_signal(t, signal, config, "v")
    assert cpm == pytest.approx(TRUE_CPM, abs=3.0)


# ---------------------------------------------------------------------------
# run_ablations_on_video: monkeypatch cached branch calls with synthetic,
# genuinely-oscillating data so the filter/estimate/fuse chain succeeds
# ---------------------------------------------------------------------------


def _synthetic_mediapipe_result(n_frames, video_id="v"):
    t = np.arange(n_frames) / FS
    roi = RoiBox(0, 0, 50, 50)
    detections = [
        HandDetection(
            frame_index=i,
            timestamp_sec=float(t[i]),
            detected=True,
            confidence=0.9,
            wrist_xy=(25.0, 100.0 + 20.0 * np.sin(2 * np.pi * TRUE_FREQ_HZ * t[i])),
            landmarks_xy=tuple((float(j), float(j)) for j in range(21)),
            bbox=roi,
            roi=roi,
        )
        for i in range(n_frames)
    ]
    return MediaPipeVideoResult(
        video_id=video_id,
        frame_width=200,
        frame_height=200,
        detections=detections,
        detection_rate=1.0,
        longest_gap_frames=0,
        longest_gap_sec=0.0,
        model_sha256="deadbeef",
    )


def _synthetic_cotracker_result(n_frames, video_id="v"):
    t = np.arange(n_frames) / FS
    motion_y = 100.0 + 20.0 * np.sin(2 * np.pi * TRUE_FREQ_HZ * t)
    return CoTrackerVideoResult(
        video_id=video_id,
        num_points=1,
        frame_indices=np.arange(n_frames),
        timestamps_sec=t,
        tracks_x=np.zeros((n_frames, 1)),
        tracks_y=motion_y.reshape(-1, 1),
        visibility=np.ones((n_frames, 1), dtype=bool),
        num_visible_per_frame=np.ones(n_frames, dtype=int),
        valid_ratio_per_frame=np.ones(n_frames),
        reinit_events=[],
        track_loss_periods=[],
        longest_track_loss_sec=0.0,
        tracker_motion_y=motion_y,
    )


def _synthetic_ego_motion_result(n_frames, video_id="v"):
    frames = [
        FrameEgoMotion(
            frame_index=i,
            timestamp_sec=i / FS,
            num_background_features=10,
            num_matched=8,
            num_inliers=8,
            inlier_ratio=1.0,
            translation_x=0.0,
            translation_y=0.0,
            rotation_rad=0.0,
            scale=1.0,
            transform_valid=True,
            confidence=1.0,
        )
        for i in range(n_frames)
    ]
    return EgoMotionVideoResult(
        video_id=video_id, seed=42, frames=frames, unreliable_periods=[], longest_unreliable_sec=0.0
    )


def _synthetic_optical_flow_result(n_frames, video_id="v"):
    t = np.arange(n_frames) / FS
    residual = 10.0 * np.sin(2 * np.pi * TRUE_FREQ_HZ * t)
    frames = [
        FrameOpticalFlow(
            frame_index=i,
            timestamp_sec=float(t[i]),
            foreground_flow_y=float(residual[i]),
            background_flow_y=0.0,
            residual_flow_y=float(residual[i]),
            flow_magnitude=abs(float(residual[i])),
            valid=True,
        )
        for i in range(n_frames)
    ]
    return OpticalFlowVideoResult(
        video_id=video_id, frames=frames, flow_motion_y=residual, unstable_periods=[], longest_unstable_sec=0.0
    )


def _patch_branches(monkeypatch, n_frames=int(DURATION_SEC * FS), repnet_cpm=98.0):
    mp_result = _synthetic_mediapipe_result(n_frames)
    ct_result = _synthetic_cotracker_result(n_frames)
    ego_result = _synthetic_ego_motion_result(n_frames)
    flow_result = _synthetic_optical_flow_result(n_frames)
    repnet_result = RepNetResult(
        video_id="v",
        cpm=repnet_cpm,
        confidence=0.7,
        pred_period_frames=18.0,
        chosen_stride=1,
        fps=FS,
        num_frames=n_frames,
        reason=None,
    )

    monkeypatch.setattr("hybrid.ablation.run_mediapipe_on_video_cached", lambda *a, **k: mp_result)
    monkeypatch.setattr("hybrid.ablation.run_cotracker_on_video_cached", lambda *a, **k: ct_result)
    monkeypatch.setattr("hybrid.ablation.run_ego_motion_on_video_cached", lambda *a, **k: ego_result)
    monkeypatch.setattr("hybrid.ablation.run_optical_flow_on_video_cached", lambda *a, **k: flow_result)
    monkeypatch.setattr("hybrid.ablation.run_repnet_on_video_cached", lambda *a, **k: repnet_result)


def _dev_video(video_id, gt_cpm, tmp_path):
    gt = GroundTruth(
        filename=f"{video_id}.mp4",
        cpr_start_sec=0.0,
        cpr_end_sec=30.0,
        cpr_duration_sec=30.0,
        known_compression_count=50,
        gt_cpm=gt_cpm,
    )
    return DevVideo(video_id=video_id, split_path=tmp_path / f"{video_id}_development.mp4", gt=gt)


def _fake_full_hybrid_result(video_id, gt_cpm, final_cpm):
    return VideoEvaluationResult(
        video_id=video_id,
        gt_cpm=gt_cpm,
        cwt_cpm=final_cpm,
        autocorrelation_cpm=final_cpm,
        fft_cpm=final_cpm,
        peaks_cpm=final_cpm,
        repnet_cpm=final_cpm,
        final_cpm=final_cpm,
        overall_confidence=0.8,
        signed_error=final_cpm - gt_cpm,
        absolute_error=abs(final_cpm - gt_cpm),
        runtime_sec=5.0,
    )


def test_run_ablations_on_video_produces_all_eight(monkeypatch, tmp_path):
    _patch_branches(monkeypatch)
    dev_video = _dev_video("video1", gt_cpm=TRUE_CPM, tmp_path=tmp_path)
    full_hybrid = _fake_full_hybrid_result("video1", TRUE_CPM, 99.0)
    config = HybridConfig()
    cache = CacheManager(tmp_path)

    results = run_ablations_on_video(dev_video, config, cache, full_hybrid)

    names = {r.ablation for r in results}
    assert names == {
        "A_mediapipe_wrist",
        "B_cotracker_raw",
        "C_cotracker_affine",
        "D_flow_raw",
        "E_flow_affine_compensated",
        "F_tracker_flow_fused",
        "G_repnet_alone",
        "H_full_hybrid",
    }
    h = next(r for r in results if r.ablation == "H_full_hybrid")
    assert h.cpm == pytest.approx(99.0)
    g = next(r for r in results if r.ablation == "G_repnet_alone")
    assert g.cpm == pytest.approx(98.0)

    # every signal-based ablation should recover something in the right ballpark
    for r in results:
        if r.ablation not in ("G_repnet_alone", "H_full_hybrid"):
            assert r.cpm is not None
            assert r.cpm == pytest.approx(TRUE_CPM, abs=15.0)


def test_run_ablations_on_video_handles_repnet_none(monkeypatch, tmp_path):
    _patch_branches(monkeypatch, repnet_cpm=None)
    dev_video = _dev_video("video1", gt_cpm=TRUE_CPM, tmp_path=tmp_path)
    full_hybrid = _fake_full_hybrid_result("video1", TRUE_CPM, 99.0)
    config = HybridConfig()
    cache = CacheManager(tmp_path)

    results = run_ablations_on_video(dev_video, config, cache, full_hybrid)
    g = next(r for r in results if r.ablation == "G_repnet_alone")
    assert g.cpm is None
    assert g.absolute_error is None


# ---------------------------------------------------------------------------
# summarize_ablations
# ---------------------------------------------------------------------------


def test_summarize_ablations_aggregates_and_sorts_by_mae():
    all_results = [
        [
            AblationVideoResult(ablation="X", video_id="v0", cpm=92.0, absolute_error=2.0, runtime_sec=1.0),
            AblationVideoResult(ablation="Y", video_id="v0", cpm=80.0, absolute_error=10.0, runtime_sec=1.0),
        ],
        [
            AblationVideoResult(ablation="X", video_id="v1", cpm=94.0, absolute_error=4.0, runtime_sec=1.0),
            AblationVideoResult(ablation="Y", video_id="v1", cpm=70.0, absolute_error=20.0, runtime_sec=1.0),
        ],
    ]
    summaries = summarize_ablations(all_results, seed=42, num_bootstrap_resamples=100)

    assert [s.ablation for s in summaries] == ["X", "Y"]  # sorted by MAE ascending
    x = next(s for s in summaries if s.ablation == "X")
    assert x.mae == pytest.approx(3.0)
    assert x.num_videos == 2
    assert x.notes == ""


def test_summarize_ablations_notes_missing_videos():
    all_results = [
        [AblationVideoResult(ablation="X", video_id="v0", cpm=90.0, absolute_error=0.0, runtime_sec=1.0)],
        [AblationVideoResult(ablation="X", video_id="v1", cpm=None, absolute_error=None, runtime_sec=1.0)],
    ]
    summaries = summarize_ablations(all_results, seed=42, num_bootstrap_resamples=100)
    x = summaries[0]
    assert x.num_videos == 1
    assert "1 video" in x.notes


# ---------------------------------------------------------------------------
# wilcoxon_hybrid_vs_best_ablation
# ---------------------------------------------------------------------------


def test_wilcoxon_picks_best_single_branch_ablation():
    all_results = [
        [
            AblationVideoResult(ablation="H_full_hybrid", video_id="v0", cpm=91.0, absolute_error=1.0, runtime_sec=1.0),
            AblationVideoResult(ablation="X", video_id="v0", cpm=95.0, absolute_error=5.0, runtime_sec=1.0),
            AblationVideoResult(ablation="Y", video_id="v0", cpm=100.0, absolute_error=10.0, runtime_sec=1.0),
        ],
        [
            AblationVideoResult(ablation="H_full_hybrid", video_id="v1", cpm=88.0, absolute_error=2.0, runtime_sec=1.0),
            AblationVideoResult(ablation="X", video_id="v1", cpm=93.0, absolute_error=7.0, runtime_sec=1.0),
            AblationVideoResult(ablation="Y", video_id="v1", cpm=101.0, absolute_error=11.0, runtime_sec=1.0),
        ],
        [
            AblationVideoResult(ablation="H_full_hybrid", video_id="v2", cpm=89.0, absolute_error=1.0, runtime_sec=1.0),
            AblationVideoResult(ablation="X", video_id="v2", cpm=94.0, absolute_error=6.0, runtime_sec=1.0),
            AblationVideoResult(ablation="Y", video_id="v2", cpm=102.0, absolute_error=12.0, runtime_sec=1.0),
        ],
    ]
    summaries = summarize_ablations(all_results, seed=42, num_bootstrap_resamples=100)
    result = wilcoxon_hybrid_vs_best_ablation(all_results, summaries)

    assert result["best_ablation"] == "X"  # X has lower MAE than Y
    assert result["statistic"] is not None
    assert result["p_value"] is not None


def test_wilcoxon_handles_too_few_paired_videos():
    all_results = [
        [
            AblationVideoResult(ablation="H_full_hybrid", video_id="v0", cpm=91.0, absolute_error=1.0, runtime_sec=1.0),
            AblationVideoResult(ablation="X", video_id="v0", cpm=95.0, absolute_error=5.0, runtime_sec=1.0),
        ],
    ]
    summaries = summarize_ablations(all_results, seed=42, num_bootstrap_resamples=100)
    result = wilcoxon_hybrid_vs_best_ablation(all_results, summaries)

    assert result["statistic"] is None
    assert result["p_value"] is None
