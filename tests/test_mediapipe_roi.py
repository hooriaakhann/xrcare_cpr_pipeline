import cv2
import numpy as np
import pytest

from hybrid.caching import CacheManager
from hybrid.config import HybridConfig
from hybrid.exceptions import HandNotDetectedError
from hybrid.mediapipe_roi import (
    HandDetection,
    MediaPipeVideoResult,
    RoiBox,
    _bbox_from_points,
    _expand_roi,
    _longest_gap,
    _mediapipe_config_hash,
    get_cotracker_init_points,
    run_mediapipe_on_video_cached,
)

# ---------------------------------------------------------------------------
# Pure-logic tests (no model download / inference — fast, run in CI)
# ---------------------------------------------------------------------------


def test_roi_box_clip_to_frame_bounds():
    box = RoiBox(x_min=-10, y_min=-5, x_max=1000, y_max=2000)
    clipped = box.clip(width=640, height=480)
    assert clipped == RoiBox(0, 0, 640, 480)


def test_roi_box_width_height():
    box = RoiBox(10, 20, 110, 70)
    assert box.width == 100
    assert box.height == 50


def test_bbox_from_points():
    points = ((10.0, 20.0), (30.0, 5.0), (15.0, 40.0))
    box = _bbox_from_points(points)
    assert box == RoiBox(10, 5, 30, 40)


def test_expand_roi_pads_beyond_tight_bbox():
    bbox = RoiBox(100, 100, 200, 200)
    mp_config = HybridConfig().mediapipe
    # wrist directly "below" middle_mcp -> forearm direction points further down
    roi = _expand_roi(
        bbox, wrist=(150.0, 200.0), middle_mcp=(150.0, 120.0), mp_config=mp_config, width=1000, height=1000
    )
    assert roi.x_min < bbox.x_min
    assert roi.y_min < bbox.y_min
    assert roi.x_max > bbox.x_max
    # forearm extension continues past the wrist in the +y direction
    assert roi.y_max > bbox.y_max + bbox.height * mp_config.forearm_extension_ratio * 0.5


def test_expand_roi_clips_to_frame_bounds():
    bbox = RoiBox(5, 5, 15, 15)
    mp_config = HybridConfig().mediapipe
    roi = _expand_roi(bbox, wrist=(10.0, 15.0), middle_mcp=(10.0, 5.0), mp_config=mp_config, width=20, height=20)
    assert roi.x_min >= 0
    assert roi.y_min >= 0
    assert roi.x_max <= 20
    assert roi.y_max <= 20


def _det(frame_index, timestamp_sec, detected):
    return HandDetection(frame_index=frame_index, timestamp_sec=timestamp_sec, detected=detected)


def test_longest_gap_no_gaps():
    dets = [_det(i, i * 0.1, True) for i in range(5)]
    assert _longest_gap(dets) == (0, 0.0)


def test_longest_gap_single_gap_in_middle():
    dets = [_det(0, 0.0, True), _det(1, 0.1, False), _det(2, 0.2, False), _det(3, 0.3, True)]
    frames, seconds = _longest_gap(dets)
    assert frames == 2
    # span between the first and last missed-frame timestamps in the run
    assert seconds == pytest.approx(0.1, abs=1e-9)


def test_longest_gap_at_start():
    dets = [_det(0, 0.0, False), _det(1, 0.1, False), _det(2, 0.2, True)]
    frames, _ = _longest_gap(dets)
    assert frames == 2


def test_longest_gap_at_end():
    dets = [_det(0, 0.0, True), _det(1, 0.1, False), _det(2, 0.2, False)]
    frames, _ = _longest_gap(dets)
    assert frames == 2


def test_longest_gap_picks_the_longest_of_several():
    dets = [
        _det(0, 0.0, True),
        _det(1, 0.1, False),
        _det(2, 0.2, True),
        _det(3, 0.3, False),
        _det(4, 0.4, False),
        _det(5, 0.5, False),
        _det(6, 0.6, True),
    ]
    frames, _ = _longest_gap(dets)
    assert frames == 3


def _detected_hand(roi=RoiBox(0, 0, 100, 100), num_landmarks=21):
    landmarks = tuple((float(i), float(i)) for i in range(num_landmarks))
    return HandDetection(
        frame_index=0,
        timestamp_sec=0.0,
        detected=True,
        confidence=0.9,
        wrist_xy=landmarks[0],
        landmarks_xy=landmarks,
        bbox=RoiBox(0, 0, 20, 20),
        roi=roi,
    )


def test_get_cotracker_init_points_uses_landmarks_first():
    det = _detected_hand()
    points = get_cotracker_init_points(det, num_points=5)
    assert points.shape == (5, 2)
    np.testing.assert_array_equal(points, np.array([(float(i), float(i)) for i in range(5)]))


def test_get_cotracker_init_points_fills_remainder_with_grid():
    det = _detected_hand(roi=RoiBox(0, 0, 200, 200))
    points = get_cotracker_init_points(det, num_points=30)
    assert points.shape == (30, 2)
    # first 21 come from landmarks
    np.testing.assert_array_equal(points[:21], np.array([(float(i), float(i)) for i in range(21)]))
    # remaining points fall inside the ROI
    assert np.all(points[21:, 0] >= 0) and np.all(points[21:, 0] <= 200)
    assert np.all(points[21:, 1] >= 0) and np.all(points[21:, 1] <= 200)


def test_get_cotracker_init_points_raises_when_not_detected():
    det = HandDetection(frame_index=0, timestamp_sec=0.0, detected=False)
    with pytest.raises(HandNotDetectedError):
        get_cotracker_init_points(det, num_points=10)


def test_get_cotracker_init_points_raises_on_invalid_num_points():
    det = _detected_hand()
    with pytest.raises(ValueError):
        get_cotracker_init_points(det, num_points=0)


def test_mediapipe_config_hash_changes_with_config():
    a = HybridConfig()
    b = HybridConfig()
    assert _mediapipe_config_hash(a) == _mediapipe_config_hash(b)

    b.mediapipe.min_hand_detection_confidence = 0.9
    assert _mediapipe_config_hash(a) != _mediapipe_config_hash(b)


# ---------------------------------------------------------------------------
# Caching wrapper: monkeypatch the expensive call so this stays fast/offline
# ---------------------------------------------------------------------------


def _fake_result(video_id="video1"):
    return MediaPipeVideoResult(
        video_id=video_id,
        frame_width=64,
        frame_height=48,
        detections=[_det(0, 0.0, True)],
        detection_rate=1.0,
        longest_gap_frames=0,
        longest_gap_sec=0.0,
        model_sha256="deadbeef",
    )


def test_cached_wrapper_avoids_recompute_on_hit(tmp_path, monkeypatch):
    calls = []

    def fake_run(video_path, config, video_id):
        calls.append(video_id)
        return _fake_result(video_id)

    monkeypatch.setattr("hybrid.mediapipe_roi.run_mediapipe_on_video", fake_run)

    config = HybridConfig()
    cache = CacheManager(tmp_path)

    first = run_mediapipe_on_video_cached("dummy.mp4", config, "video1", cache)
    second = run_mediapipe_on_video_cached("dummy.mp4", config, "video1", cache)

    assert calls == ["video1"]  # only computed once
    assert first.video_id == second.video_id == "video1"


def test_cached_wrapper_recomputes_when_caching_disabled(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        "hybrid.mediapipe_roi.run_mediapipe_on_video",
        lambda video_path, config, video_id: calls.append(video_id) or _fake_result(video_id),
    )

    config = HybridConfig()
    config.caching.enabled = False
    cache = CacheManager(tmp_path)

    run_mediapipe_on_video_cached("dummy.mp4", config, "video1", cache)
    run_mediapipe_on_video_cached("dummy.mp4", config, "video1", cache)

    assert calls == ["video1", "video1"]


def test_cached_wrapper_recomputes_on_config_change(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        "hybrid.mediapipe_roi.run_mediapipe_on_video",
        lambda video_path, config, video_id: calls.append(video_id) or _fake_result(video_id),
    )

    cache = CacheManager(tmp_path)
    config_a = HybridConfig()
    run_mediapipe_on_video_cached("dummy.mp4", config_a, "video1", cache)

    config_b = HybridConfig()
    config_b.mediapipe.min_hand_detection_confidence = 0.9
    run_mediapipe_on_video_cached("dummy.mp4", config_b, "video1", cache)

    assert calls == ["video1", "video1"]


# ---------------------------------------------------------------------------
# Real-model tests: download + run actual MediaPipe inference. Slow, needs
# network on first run (cached under models/ after that) — excluded from
# `make test` by default, see `make test-all`.
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_hand_not_detected_on_blank_video_raises(tmp_path):
    from hybrid.mediapipe_roi import run_mediapipe_on_video

    video_path = tmp_path / "blank.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, 10.0, (64, 48))
    for _ in range(5):
        writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
    writer.release()

    config = HybridConfig()
    with pytest.raises(HandNotDetectedError):
        run_mediapipe_on_video(video_path, config, "blank_video")


@pytest.mark.slow
def test_detect_frame_finds_hand_on_real_dev_video():
    from pathlib import Path

    from hybrid.mediapipe_roi import MediaPipeHandLocalizer
    from hybrid.video_io import VideoReader

    real_video = Path(__file__).resolve().parents[1] / "data" / "split" / "video1_development.mp4"
    if not real_video.exists():
        pytest.skip("real development video not present locally (data/split is gitignored)")

    config = HybridConfig()
    with VideoReader(real_video) as reader, MediaPipeHandLocalizer(config) as localizer:
        detections = []
        for i, frame in enumerate(reader):
            detections.append(localizer.detect_frame(frame))
            if i >= 4:
                break
    # not asserting every frame detects (real video, model can miss) but at
    # least one of the first 5 frames should show a hand
    assert any(d.detected for d in detections)
    detected = next(d for d in detections if d.detected)
    assert detected.landmarks_xy is not None
    assert len(detected.landmarks_xy) == 21
    assert detected.roi is not None
    assert detected.roi.width > detected.bbox.width  # ROI is padded beyond the tight bbox
