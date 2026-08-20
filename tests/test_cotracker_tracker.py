import cv2
import numpy as np
import pytest

from hybrid.caching import CacheManager
from hybrid.config import HybridConfig
from hybrid.cotracker_tracker import (
    CoTrackerVideoResult,
    _cotracker_config_hash,
    _longest_run_below_threshold,
    _median_visible_y,
    _resize_for_cotracker,
    run_cotracker_on_video,
    run_cotracker_on_video_cached,
)
from hybrid.exceptions import TrackLostError
from hybrid.mediapipe_roi import HandDetection, MediaPipeVideoResult, RoiBox

# ---------------------------------------------------------------------------
# Pure-logic tests (no model / video I/O) -- fast, run in CI
# ---------------------------------------------------------------------------


def test_resize_for_cotracker_caps_max_dim_preserving_aspect():
    frames = [np.zeros((2000, 1000, 3), dtype=np.uint8)]
    resized, scale_x, scale_y = _resize_for_cotracker(frames, max_dim=1000)
    assert resized.shape[1:3] == (1000, 500)  # (H, W)
    assert scale_x == pytest.approx(500 / 1000)
    assert scale_y == pytest.approx(1000 / 2000)


def test_resize_for_cotracker_noop_when_already_small():
    frames = [np.zeros((100, 80, 3), dtype=np.uint8)]
    resized, scale_x, scale_y = _resize_for_cotracker(frames, max_dim=1000)
    assert resized.shape[1:3] == (100, 80)
    assert scale_x == pytest.approx(1.0)
    assert scale_y == pytest.approx(1.0)


def test_longest_run_below_threshold_none():
    ratio = np.array([1.0, 1.0, 1.0])
    ts = np.array([0.0, 0.1, 0.2])
    frames, seconds, periods = _longest_run_below_threshold(ratio, ts, threshold=0.5)
    assert (frames, seconds, periods) == (0, 0.0, [])


def test_longest_run_below_threshold_picks_longest_of_several():
    ratio = np.array([1.0, 0.2, 1.0, 0.1, 0.1, 0.1, 1.0])
    ts = np.arange(7) * 0.1
    frames, seconds, periods = _longest_run_below_threshold(ratio, ts, threshold=0.5)
    assert frames == 3
    assert periods == [(1, 1), (3, 5)]


def test_longest_run_below_threshold_run_at_end():
    ratio = np.array([1.0, 0.1, 0.1])
    ts = np.arange(3) * 0.1
    frames, seconds, periods = _longest_run_below_threshold(ratio, ts, threshold=0.5)
    assert frames == 2
    assert periods == [(1, 2)]


def test_median_visible_y_uses_only_visible_points():
    tracks_y = np.array([[10.0, 20.0, 30.0]])
    visibility = np.array([[True, False, True]])
    motion = _median_visible_y(tracks_y, visibility)
    assert motion[0] == pytest.approx(20.0)  # median of 10, 30


def test_median_visible_y_nan_when_none_visible():
    tracks_y = np.array([[10.0, 20.0]])
    visibility = np.array([[False, False]])
    motion = _median_visible_y(tracks_y, visibility)
    assert np.isnan(motion[0])


def test_cotracker_config_hash_changes_with_config():
    a = HybridConfig()
    b = HybridConfig()
    assert _cotracker_config_hash(a) == _cotracker_config_hash(b)
    b.cotracker.num_points = 99
    assert _cotracker_config_hash(a) != _cotracker_config_hash(b)


# ---------------------------------------------------------------------------
# Orchestration tests: monkeypatch _run_window so windowing/reinit/track-loss
# logic is tested without any real model inference.
# ---------------------------------------------------------------------------


def _write_synthetic_video(path, num_frames, fps=2.0, width=32, height=32):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    for i in range(num_frames):
        writer.write(np.full((height, width, 3), i % 256, dtype=np.uint8))
    writer.release()


def _detected(frame_index, timestamp_sec):
    landmarks = tuple((float(5 + i), float(5 + i)) for i in range(21))
    return HandDetection(
        frame_index=frame_index,
        timestamp_sec=timestamp_sec,
        detected=True,
        confidence=0.9,
        wrist_xy=landmarks[0],
        landmarks_xy=landmarks,
        bbox=RoiBox(0, 0, 20, 20),
        roi=RoiBox(0, 0, 30, 30),
    )


def _mp_result(detections, video_id="video_x"):
    detected_count = sum(1 for d in detections if d.detected)
    return MediaPipeVideoResult(
        video_id=video_id,
        frame_width=32,
        frame_height=32,
        detections=detections,
        detection_rate=detected_count / len(detections),
        longest_gap_frames=0,
        longest_gap_sec=0.0,
        model_sha256="deadbeef",
    )


def _make_fake_run_window(visible_fractions):
    """visible_fractions[i] = fraction of points visible in window i's last
    (and every) frame; enough for the reinit/track-loss decisions under test.
    """
    calls = []

    def fake(model, frames_bgr, query_frame_local, query_points_xy, ct_config):
        idx = len(calls)
        calls.append({"query_frame_local": query_frame_local, "query_points_xy": query_points_xy.copy()})
        t, n = len(frames_bgr), query_points_xy.shape[0]
        tracks = np.tile(query_points_xy, (t, 1, 1)).astype(np.float64)
        visibility = np.ones((t, n), dtype=bool)
        frac = visible_fractions[idx]
        num_invisible = round((1 - frac) * n)
        if num_invisible > 0:
            visibility[:, :num_invisible] = False
        return tracks, visibility

    return fake, calls


def test_reinit_triggered_on_low_visibility_then_recovers(tmp_path, monkeypatch):
    video_path = tmp_path / "v.mp4"
    _write_synthetic_video(video_path, num_frames=10, fps=2.0)

    detections = [_detected(0, 0.0), _detected(8, 4.0)]
    mp_result = _mp_result(detections)

    config = HybridConfig()
    config.cotracker.window_frames = 4  # windows: [0-3], [4-7], [8-9]
    config.cotracker.reinit_visibility_threshold = 0.4
    config.cotracker.visibility_threshold = 0.6
    config.cotracker.max_track_loss_sec = 5.0  # generous -- shouldn't raise here

    fake_run_window, calls = _make_fake_run_window([1.0, 0.2, 1.0])
    monkeypatch.setattr("hybrid.cotracker_tracker._run_window", fake_run_window)
    monkeypatch.setattr("hybrid.cotracker_tracker._load_cotracker_model", lambda: object())

    result = run_cotracker_on_video(video_path, mp_result, config, "video_x")

    assert len(calls) == 3  # 3 windows processed
    assert len(result.reinit_events) == 1
    assert result.reinit_events[0][0] == 8  # reseeded at frame 8
    assert result.tracker_motion_y.shape[0] == 10
    assert result.frame_indices.tolist() == list(range(10))


def test_track_lost_error_when_sustained_and_reinits_exhausted(tmp_path, monkeypatch):
    video_path = tmp_path / "v.mp4"
    _write_synthetic_video(video_path, num_frames=10, fps=2.0)

    detections = [_detected(0, 0.0)]
    mp_result = _mp_result(detections)

    config = HybridConfig()
    config.cotracker.window_frames = 4
    config.cotracker.max_reinits = 0  # never allowed to reinit
    config.cotracker.visibility_threshold = 0.6
    config.cotracker.max_track_loss_sec = 1.0  # low bar -- sustained loss should exceed it

    fake_run_window, _ = _make_fake_run_window([1.0, 0.1, 0.1])
    monkeypatch.setattr("hybrid.cotracker_tracker._run_window", fake_run_window)
    monkeypatch.setattr("hybrid.cotracker_tracker._load_cotracker_model", lambda: object())

    with pytest.raises(TrackLostError):
        run_cotracker_on_video(video_path, mp_result, config, "video_x")


def test_reinit_needed_but_no_detection_available_raises(tmp_path, monkeypatch):
    video_path = tmp_path / "v.mp4"
    _write_synthetic_video(video_path, num_frames=10, fps=2.0)

    detections = [_detected(0, 0.0)]  # nothing detected later to reseed window 2
    mp_result = _mp_result(detections)

    config = HybridConfig()
    config.cotracker.window_frames = 4
    config.cotracker.reinit_visibility_threshold = 0.9  # window0's full visibility (1.0) still triggers reinit check

    # window0 ends below reinit_visibility_threshold -> forces reinit for window1,
    # but no MediaPipe detection exists at/after frame 4.
    fake_run_window, _ = _make_fake_run_window([0.5, 1.0, 1.0])
    monkeypatch.setattr("hybrid.cotracker_tracker._run_window", fake_run_window)
    monkeypatch.setattr("hybrid.cotracker_tracker._load_cotracker_model", lambda: object())

    with pytest.raises(TrackLostError, match="no MediaPipe detection available"):
        run_cotracker_on_video(video_path, mp_result, config, "video_x")


def test_no_detections_at_all_raises_immediately(tmp_path, monkeypatch):
    video_path = tmp_path / "v.mp4"
    _write_synthetic_video(video_path, num_frames=5, fps=2.0)

    mp_result = _mp_result([HandDetection(frame_index=i, timestamp_sec=i * 0.5, detected=False) for i in range(5)])

    config = HybridConfig()
    monkeypatch.setattr("hybrid.cotracker_tracker._load_cotracker_model", lambda: object())

    with pytest.raises(TrackLostError, match="never detected"):
        run_cotracker_on_video(video_path, mp_result, config, "video_x")


# ---------------------------------------------------------------------------
# Caching wrapper: monkeypatch the expensive call itself
# ---------------------------------------------------------------------------


def _fake_cotracker_result(video_id="video_x"):
    return CoTrackerVideoResult(
        video_id=video_id,
        num_points=1,
        frame_indices=np.array([0]),
        timestamps_sec=np.array([0.0]),
        tracks_x=np.array([[1.0]]),
        tracks_y=np.array([[1.0]]),
        visibility=np.array([[True]]),
        num_visible_per_frame=np.array([1]),
        valid_ratio_per_frame=np.array([1.0]),
        reinit_events=[],
        track_loss_periods=[],
        longest_track_loss_sec=0.0,
        tracker_motion_y=np.array([1.0]),
    )


def test_cached_wrapper_avoids_recompute_on_hit(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        "hybrid.cotracker_tracker.run_cotracker_on_video",
        lambda video_path, mp_result, config, video_id: calls.append(video_id) or _fake_cotracker_result(video_id),
    )

    config = HybridConfig()
    cache = CacheManager(tmp_path)
    mp_result = _mp_result([_detected(0, 0.0)])

    run_cotracker_on_video_cached("dummy.mp4", mp_result, config, "video_x", cache)
    run_cotracker_on_video_cached("dummy.mp4", mp_result, config, "video_x", cache)

    assert calls == ["video_x"]


def test_cached_wrapper_recomputes_on_mediapipe_config_change(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        "hybrid.cotracker_tracker.run_cotracker_on_video",
        lambda video_path, mp_result, config, video_id: calls.append(video_id) or _fake_cotracker_result(video_id),
    )

    cache = CacheManager(tmp_path)
    mp_result = _mp_result([_detected(0, 0.0)])

    config_a = HybridConfig()
    run_cotracker_on_video_cached("dummy.mp4", mp_result, config_a, "video_x", cache)

    config_b = HybridConfig()
    config_b.mediapipe.min_hand_detection_confidence = 0.9  # MediaPipe change must invalidate CoTracker cache too
    run_cotracker_on_video_cached("dummy.mp4", mp_result, config_b, "video_x", cache)

    assert calls == ["video_x", "video_x"]


# ---------------------------------------------------------------------------
# Real model: one small end-to-end smoke test. Slow (loads the real
# checkpoint via torch.hub) -- excluded from `make test`, see `make test-all`.
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_real_cotracker_runs_on_tiny_synthetic_video(tmp_path):
    from hybrid.cotracker_tracker import _load_cotracker_model, _run_window

    width, height, num_frames = 128, 96, 12
    rng = np.random.default_rng(0)
    frames = []
    for i in range(num_frames):
        base = rng.integers(0, 255, size=(height, width, 3), dtype=np.uint8)
        # a moving bright block gives CoTracker's queried points something
        # locally distinctive to actually track, not pure noise
        base[10 : 30 + i, 10 : 30 + i] = 255
        frames.append(base)

    model = _load_cotracker_model()
    query_points = np.array([[15.0, 15.0], [50.0, 50.0], [100.0, 80.0]])
    config = HybridConfig()

    tracks, visibility = _run_window(
        model, frames, query_frame_local=0, query_points_xy=query_points, ct_config=config.cotracker
    )

    assert tracks.shape == (num_frames, 3, 2)
    assert visibility.shape == (num_frames, 3)
    assert visibility.dtype == bool
    # the query frame's own points must be visible and equal to the query
    np.testing.assert_allclose(tracks[0], query_points, atol=1.0)
