import cv2
import numpy as np
import pytest

from hybrid.caching import CacheManager
from hybrid.config import HybridConfig
from hybrid.mediapipe_pose import (
    NUM_POSE_LANDMARKS,
    POSE_CONNECTIONS,
    MediaPipePoseVideoResult,
    PoseDetection,
    _pose_config_hash,
    run_mediapipe_pose_on_video_cached,
)


def _det(frame_index, timestamp_sec, detected, landmarks_xy=None):
    return PoseDetection(
        frame_index=frame_index, timestamp_sec=timestamp_sec, detected=detected, landmarks_xy=landmarks_xy
    )


# ---------------------------------------------------------------------------
# POSE_CONNECTIONS topology
# ---------------------------------------------------------------------------


def test_pose_connections_reference_valid_landmark_indices():
    for i, j in POSE_CONNECTIONS:
        assert 0 <= i < NUM_POSE_LANDMARKS
        assert 0 <= j < NUM_POSE_LANDMARKS
        assert i != j


# ---------------------------------------------------------------------------
# _pose_config_hash
# ---------------------------------------------------------------------------


def test_pose_config_hash_stable_and_sensitive_to_changes():
    a = HybridConfig()
    b = HybridConfig()
    assert _pose_config_hash(a) == _pose_config_hash(b)

    b.mediapipe_pose.min_pose_detection_confidence = 0.9
    assert _pose_config_hash(a) != _pose_config_hash(b)


# ---------------------------------------------------------------------------
# Caching wrapper: monkeypatch the expensive call so this stays fast/offline
# ---------------------------------------------------------------------------


def _fake_result(video_id="video1"):
    return MediaPipePoseVideoResult(
        video_id=video_id,
        frame_width=64,
        frame_height=48,
        detections=[_det(0, 0.0, True, landmarks_xy=tuple((float(i), float(i)) for i in range(NUM_POSE_LANDMARKS)))],
        detection_rate=1.0,
        model_sha256="deadbeef",
    )


def test_cached_wrapper_avoids_recompute_on_hit(tmp_path, monkeypatch):
    calls = []

    def fake_run(video_path, config, video_id):
        calls.append(video_id)
        return _fake_result(video_id)

    monkeypatch.setattr("hybrid.mediapipe_pose.run_mediapipe_pose_on_video", fake_run)

    config = HybridConfig()
    cache = CacheManager(tmp_path)

    first = run_mediapipe_pose_on_video_cached("dummy.mp4", config, "video1", cache)
    second = run_mediapipe_pose_on_video_cached("dummy.mp4", config, "video1", cache)

    assert calls == ["video1"]  # only computed once
    assert first.video_id == second.video_id == "video1"


def test_cached_wrapper_recomputes_when_caching_disabled(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        "hybrid.mediapipe_pose.run_mediapipe_pose_on_video",
        lambda video_path, config, video_id: calls.append(video_id) or _fake_result(video_id),
    )

    config = HybridConfig()
    config.caching.enabled = False
    cache = CacheManager(tmp_path)

    run_mediapipe_pose_on_video_cached("dummy.mp4", config, "video1", cache)
    run_mediapipe_pose_on_video_cached("dummy.mp4", config, "video1", cache)

    assert calls == ["video1", "video1"]


def test_cached_wrapper_recomputes_on_config_change(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        "hybrid.mediapipe_pose.run_mediapipe_pose_on_video",
        lambda video_path, config, video_id: calls.append(video_id) or _fake_result(video_id),
    )

    cache = CacheManager(tmp_path)
    config_a = HybridConfig()
    run_mediapipe_pose_on_video_cached("dummy.mp4", config_a, "video1", cache)

    config_b = HybridConfig()
    config_b.mediapipe_pose.min_pose_detection_confidence = 0.9
    run_mediapipe_pose_on_video_cached("dummy.mp4", config_b, "video1", cache)

    assert calls == ["video1", "video1"]


# ---------------------------------------------------------------------------
# Real-model tests: download + run actual MediaPipe Pose inference. Slow,
# needs network on first run (cached under models/ after that) — excluded
# from `make test` by default, see `make test-all`.
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_no_pose_detected_on_blank_video_does_not_raise(tmp_path):
    from hybrid.mediapipe_pose import run_mediapipe_pose_on_video

    video_path = tmp_path / "blank.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, 10.0, (64, 48))
    for _ in range(5):
        writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
    writer.release()

    config = HybridConfig()
    result = run_mediapipe_pose_on_video(video_path, config, "blank_video")  # must not raise

    assert result.detection_rate == 0.0
    assert all(not d.detected for d in result.detections)


@pytest.mark.slow
def test_detect_frame_finds_pose_on_real_dev_video():
    from pathlib import Path

    from hybrid.mediapipe_pose import MediaPipePoseLocalizer
    from hybrid.video_io import VideoReader

    real_video = Path(__file__).resolve().parents[1] / "data" / "split" / "video1_development.mp4"
    if not real_video.exists():
        pytest.skip("real development video not present locally (data/split is gitignored)")

    config = HybridConfig()
    with VideoReader(real_video) as reader, MediaPipePoseLocalizer(config) as localizer:
        detections = []
        for i, frame in enumerate(reader):
            detections.append(localizer.detect_frame(frame))
            if i >= 4:
                break
    # not asserting every frame detects (real video, model can miss) but at
    # least one of the first 5 frames should show a pose
    assert any(d.detected for d in detections)
    detected = next(d for d in detections if d.detected)
    assert detected.landmarks_xy is not None
    assert len(detected.landmarks_xy) == NUM_POSE_LANDMARKS
