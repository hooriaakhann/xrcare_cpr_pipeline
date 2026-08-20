import csv
import json

import numpy as np
import pytest

from hybrid.caching import CacheManager
from hybrid.config import HybridConfig
from hybrid.cotracker_tracker import CoTrackerVideoResult
from hybrid.dataset import DevVideo, GroundTruth
from hybrid.diagnostics import save_diagnostics_for_all_videos, save_diagnostics_for_video
from hybrid.ego_motion import EgoMotionVideoResult, FrameEgoMotion
from hybrid.mediapipe_roi import HandDetection, MediaPipeVideoResult, RoiBox
from hybrid.optical_flow import FrameOpticalFlow, OpticalFlowVideoResult
from hybrid.repnet_branch import RepNetResult

FS = 30.0
DURATION_SEC = 30.0
TRUE_CPM = 100.0
TRUE_FREQ_HZ = TRUE_CPM / 60.0


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
        num_points=2,
        frame_indices=np.arange(n_frames),
        timestamps_sec=t,
        tracks_x=np.zeros((n_frames, 2)),
        tracks_y=np.tile(motion_y.reshape(-1, 1), (1, 2)),
        visibility=np.ones((n_frames, 2), dtype=bool),
        num_visible_per_frame=np.full(n_frames, 2),
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
    monkeypatch.setattr("hybrid.diagnostics.run_mediapipe_on_video_cached", lambda *a, **k: mp_result)
    monkeypatch.setattr("hybrid.diagnostics.run_cotracker_on_video_cached", lambda *a, **k: ct_result)
    monkeypatch.setattr("hybrid.diagnostics.run_ego_motion_on_video_cached", lambda *a, **k: ego_result)
    monkeypatch.setattr("hybrid.diagnostics.run_optical_flow_on_video_cached", lambda *a, **k: flow_result)
    monkeypatch.setattr("hybrid.diagnostics.run_repnet_on_video_cached", lambda *a, **k: repnet_result)


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


def test_save_diagnostics_creates_all_expected_files(monkeypatch, tmp_path):
    _patch_branches(monkeypatch)
    config = HybridConfig()
    config.paths.runs_dir = tmp_path / "runs"
    dev_video = _dev_video("video1", gt_cpm=TRUE_CPM, tmp_path=tmp_path)
    cache = CacheManager(tmp_path / "cache")

    run_dir = save_diagnostics_for_video(dev_video, config, cache)

    expected_files = {
        "summary.json",
        "signals.csv",
        "cotracker_tracks.csv",
        "ego_motion.csv",
        "optical_flow_signal.csv",
        "filtered_signal.csv",
        "estimator_results.json",
        "repnet_result.json",
        "peaks.csv",
    }
    actual_files = {p.name for p in run_dir.iterdir()}
    assert expected_files.issubset(actual_files)
    assert run_dir == config.paths.runs_dir / "development" / "video1"


def test_summary_json_has_correct_content(monkeypatch, tmp_path):
    _patch_branches(monkeypatch)
    config = HybridConfig()
    config.paths.runs_dir = tmp_path / "runs"
    dev_video = _dev_video("video1", gt_cpm=TRUE_CPM, tmp_path=tmp_path)
    cache = CacheManager(tmp_path / "cache")

    run_dir = save_diagnostics_for_video(dev_video, config, cache)

    with open(run_dir / "summary.json") as f:
        summary = json.load(f)

    assert summary["video_id"] == "video1"
    assert summary["gt_cpm"] == pytest.approx(TRUE_CPM)
    assert summary["final_cpm"] == pytest.approx(summary["gt_cpm"] + summary["signed_error"])
    assert summary["absolute_error"] == pytest.approx(abs(summary["signed_error"]))
    assert "config_hash" in summary
    assert "seed" in summary


def test_cotracker_tracks_csv_has_one_row_per_frame_per_point(monkeypatch, tmp_path):
    n_frames = 60
    _patch_branches(monkeypatch, n_frames=n_frames)
    config = HybridConfig()
    config.paths.runs_dir = tmp_path / "runs"
    dev_video = _dev_video("video1", gt_cpm=TRUE_CPM, tmp_path=tmp_path)
    cache = CacheManager(tmp_path / "cache")

    run_dir = save_diagnostics_for_video(dev_video, config, cache)

    with open(run_dir / "cotracker_tracks.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == n_frames * 2  # 2 points per frame in the synthetic fixture


def test_estimator_results_json_is_valid_and_complete(monkeypatch, tmp_path):
    _patch_branches(monkeypatch)
    config = HybridConfig()
    config.paths.runs_dir = tmp_path / "runs"
    dev_video = _dev_video("video1", gt_cpm=TRUE_CPM, tmp_path=tmp_path)
    cache = CacheManager(tmp_path / "cache")

    run_dir = save_diagnostics_for_video(dev_video, config, cache)

    with open(run_dir / "estimator_results.json") as f:
        payload = json.load(f)

    assert set(payload) == {"cwt", "autocorrelation", "fft", "peaks", "fusion"}
    assert "final_cpm" in payload["fusion"]
    assert len(payload["fusion"]["candidates"]) == 5
    assert isinstance(payload["peaks"]["peak_timestamps_sec"], list)  # ndarray -> list, JSON-safe


def test_save_diagnostics_for_all_videos_processes_each(monkeypatch, tmp_path):
    _patch_branches(monkeypatch)
    config = HybridConfig()
    config.paths.runs_dir = tmp_path / "runs"
    dev_videos = [_dev_video(f"video{i}", gt_cpm=TRUE_CPM, tmp_path=tmp_path) for i in range(2)]
    cache = CacheManager(tmp_path / "cache")

    run_dirs = save_diagnostics_for_all_videos(dev_videos, config, cache)

    assert len(run_dirs) == 2
    for d in run_dirs:
        assert (d / "summary.json").exists()
