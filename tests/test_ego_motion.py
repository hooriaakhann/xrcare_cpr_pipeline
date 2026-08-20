import math

import cv2
import numpy as np
import pytest

from hybrid.caching import CacheManager
from hybrid.config import HybridConfig
from hybrid.ego_motion import (
    EgoMotionVideoResult,
    FrameEgoMotion,
    _build_roi_lookup,
    _decompose_similarity,
    _ego_motion_config_hash,
    _estimate_frame_pair,
    _roi_exclusion_mask,
    run_ego_motion_on_video,
    run_ego_motion_on_video_cached,
)
from hybrid.exceptions import EgoMotionUnreliableError
from hybrid.mediapipe_roi import HandDetection, MediaPipeVideoResult, RoiBox

# ---------------------------------------------------------------------------
# Pure-logic tests
# ---------------------------------------------------------------------------


def test_roi_exclusion_mask_no_roi_is_all_included():
    mask = _roi_exclusion_mask(None, width=50, height=40, dilate_px=5)
    assert mask.shape == (40, 50)
    assert np.all(mask == 255)


def test_roi_exclusion_mask_zeroes_dilated_region():
    roi = RoiBox(10, 10, 20, 20)
    mask = _roi_exclusion_mask(roi, width=50, height=50, dilate_px=2)
    assert np.all(mask[8:22, 8:22] == 0)
    assert mask[0, 0] == 255
    assert mask[49, 49] == 255


def test_decompose_similarity_recovers_known_transform():
    theta = math.radians(10.0)
    scale = 1.2
    tx, ty = 7.0, -3.0
    matrix = np.array(
        [
            [scale * math.cos(theta), -scale * math.sin(theta), tx],
            [scale * math.sin(theta), scale * math.cos(theta), ty],
        ]
    )
    out_tx, out_ty, out_rot, out_scale = _decompose_similarity(matrix)
    assert out_tx == pytest.approx(tx)
    assert out_ty == pytest.approx(ty)
    assert out_rot == pytest.approx(theta)
    assert out_scale == pytest.approx(scale)


def _detected(frame_index, timestamp_sec, roi):
    return HandDetection(
        frame_index=frame_index,
        timestamp_sec=timestamp_sec,
        detected=True,
        confidence=0.9,
        wrist_xy=(roi.x_min, roi.y_min),
        landmarks_xy=tuple((float(i), float(i)) for i in range(21)),
        bbox=roi,
        roi=roi,
    )


def _not_detected(frame_index, timestamp_sec):
    return HandDetection(frame_index=frame_index, timestamp_sec=timestamp_sec, detected=False)


def test_build_roi_lookup_forward_fills_across_gaps():
    roi_a = RoiBox(0, 0, 10, 10)
    roi_b = RoiBox(5, 5, 15, 15)
    detections = [
        _not_detected(0, 0.0),  # no ROI known yet
        _detected(1, 0.1, roi_a),
        _not_detected(2, 0.2),  # should forward-fill to roi_a
        _detected(3, 0.3, roi_b),
    ]
    lookup = _build_roi_lookup(detections)
    assert 0 not in lookup
    assert lookup[1] == roi_a
    assert lookup[2] == roi_a
    assert lookup[3] == roi_b


def test_ego_motion_config_hash_changes_with_config():
    a = HybridConfig()
    b = HybridConfig()
    assert _ego_motion_config_hash(a) == _ego_motion_config_hash(b)
    b.ego_motion.min_inlier_ratio = 0.9
    assert _ego_motion_config_hash(a) != _ego_motion_config_hash(b)

    c = HybridConfig()
    c.project.seed = 123
    assert _ego_motion_config_hash(a) != _ego_motion_config_hash(c)


# ---------------------------------------------------------------------------
# _estimate_frame_pair: real cv2 ops on raw arrays (fast, no video/model I/O)
# ---------------------------------------------------------------------------


def test_estimate_frame_pair_recovers_pure_translation():
    rng = np.random.default_rng(0)
    pad, height, width = 30, 80, 80
    base = rng.integers(0, 255, size=(height + 2 * pad, width + 2 * pad), dtype=np.uint8)
    dx, dy = 5, 3
    prev = base[pad : pad + height, pad : pad + width]
    curr = base[pad + dy : pad + dy + height, pad + dx : pad + dx + width]

    em_config = HybridConfig().ego_motion
    num_feat, num_matched, num_inliers, inlier_ratio, matrix = _estimate_frame_pair(prev, curr, None, em_config)

    assert num_feat > 0
    assert matrix is not None
    tx, ty, rotation_rad, scale = _decompose_similarity(matrix)
    assert tx == pytest.approx(-dx, abs=1.5)
    assert ty == pytest.approx(-dy, abs=1.5)
    assert abs(math.degrees(rotation_rad)) < 3.0
    assert scale == pytest.approx(1.0, abs=0.05)
    assert inlier_ratio > 0.8


def test_estimate_frame_pair_excludes_roi_features():
    size = 100
    img = np.zeros((size, size), dtype=np.uint8)
    img[::4, ::4] = 255
    img[2::4, 2::4] = 255  # checkerboard-ish -> strong corners everywhere
    roi = RoiBox(20, 20, 60, 60)
    mask = _roi_exclusion_mask(roi, size, size, dilate_px=0)

    features = cv2.goodFeaturesToTrack(img, maxCorners=500, qualityLevel=0.01, minDistance=2, mask=mask)

    assert features is not None and len(features) > 0
    for f in features:
        x, y = f[0]
        assert not (roi.x_min <= x <= roi.x_max and roi.y_min <= y <= roi.y_max)


def test_estimate_frame_pair_no_texture_returns_no_transform():
    blank_prev = np.full((50, 50), 128, dtype=np.uint8)
    blank_curr = np.full((50, 50), 128, dtype=np.uint8)
    em_config = HybridConfig().ego_motion

    num_feat, num_matched, num_inliers, inlier_ratio, matrix = _estimate_frame_pair(
        blank_prev, blank_curr, None, em_config
    )
    assert matrix is None


# ---------------------------------------------------------------------------
# run_ego_motion_on_video: orchestration, via a monkeypatched _estimate_frame_pair
# (deterministic control over per-pair results, real tiny video for frame/timestamp
# iteration only)
# ---------------------------------------------------------------------------


def _write_synthetic_video(path, num_frames, fps=2.0, width=32, height=32):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    for i in range(num_frames):
        writer.write(np.full((height, width, 3), i % 256, dtype=np.uint8))
    writer.release()


def _mp_result(num_frames, roi=RoiBox(0, 0, 10, 10)):
    detections = [_detected(i, i * 0.5, roi) for i in range(num_frames)]
    return MediaPipeVideoResult(
        video_id="video_x",
        frame_width=32,
        frame_height=32,
        detections=detections,
        detection_rate=1.0,
        longest_gap_frames=0,
        longest_gap_sec=0.0,
        model_sha256="deadbeef",
    )


def test_run_ego_motion_first_frame_is_identity(tmp_path, monkeypatch):
    video_path = tmp_path / "v.mp4"
    _write_synthetic_video(video_path, num_frames=5)
    mp_result = _mp_result(5)

    monkeypatch.setattr(
        "hybrid.ego_motion._estimate_frame_pair",
        lambda prev, curr, roi, em_config: (50, 40, 35, 0.9, np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.5]])),
    )

    config = HybridConfig()
    result = run_ego_motion_on_video(video_path, mp_result, config, "video_x")

    assert len(result.frames) == 5
    first = result.frames[0]
    assert first.transform_valid is True
    assert first.confidence == 1.0
    assert first.scale == 1.0
    assert first.translation_x == 0.0
    assert result.seed == config.project.seed


def test_run_ego_motion_all_valid_no_unreliable_periods(tmp_path, monkeypatch):
    video_path = tmp_path / "v.mp4"
    _write_synthetic_video(video_path, num_frames=5)
    mp_result = _mp_result(5)

    monkeypatch.setattr(
        "hybrid.ego_motion._estimate_frame_pair",
        lambda prev, curr, roi, em_config: (50, 40, 38, 0.95, np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.5]])),
    )

    config = HybridConfig()
    result = run_ego_motion_on_video(video_path, mp_result, config, "video_x")

    assert result.unreliable_periods == []
    assert result.longest_unreliable_sec == 0.0
    assert all(f.transform_valid for f in result.frames)


def test_run_ego_motion_raises_when_sustained_unreliable(tmp_path, monkeypatch):
    video_path = tmp_path / "v.mp4"
    _write_synthetic_video(video_path, num_frames=6, fps=2.0)
    mp_result = _mp_result(6)

    # matrix=None -> transform invalid every pair
    monkeypatch.setattr(
        "hybrid.ego_motion._estimate_frame_pair",
        lambda prev, curr, roi, em_config: (10, 2, 0, 0.0, None),
    )

    config = HybridConfig()
    config.ego_motion.max_unreliable_span_sec = 1.0  # low bar -- 5 bad frames at 2fps easily exceeds it

    with pytest.raises(EgoMotionUnreliableError):
        run_ego_motion_on_video(video_path, mp_result, config, "video_x")


def test_run_ego_motion_passes_roi_from_lookup(tmp_path, monkeypatch):
    video_path = tmp_path / "v.mp4"
    _write_synthetic_video(video_path, num_frames=3)
    roi = RoiBox(1, 2, 3, 4)
    mp_result = _mp_result(3, roi=roi)

    seen_rois = []

    def fake_estimate(prev, curr, roi_arg, em_config):
        seen_rois.append(roi_arg)
        return (10, 8, 8, 1.0, np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))

    monkeypatch.setattr("hybrid.ego_motion._estimate_frame_pair", fake_estimate)

    config = HybridConfig()
    run_ego_motion_on_video(video_path, mp_result, config, "video_x")

    assert len(seen_rois) == 2  # 3 frames -> 2 pairs
    assert all(r == roi for r in seen_rois)


# ---------------------------------------------------------------------------
# Caching wrapper
# ---------------------------------------------------------------------------


def _fake_ego_result(video_id="video_x"):
    return EgoMotionVideoResult(
        video_id=video_id,
        seed=42,
        frames=[
            FrameEgoMotion(
                frame_index=0,
                timestamp_sec=0.0,
                num_background_features=0,
                num_matched=0,
                num_inliers=0,
                inlier_ratio=1.0,
                translation_x=0.0,
                translation_y=0.0,
                rotation_rad=0.0,
                scale=1.0,
                transform_valid=True,
                confidence=1.0,
            )
        ],
        unreliable_periods=[],
        longest_unreliable_sec=0.0,
    )


def test_cached_wrapper_avoids_recompute_on_hit(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        "hybrid.ego_motion.run_ego_motion_on_video",
        lambda video_path, mp_result, config, video_id: calls.append(video_id) or _fake_ego_result(video_id),
    )
    config = HybridConfig()
    cache = CacheManager(tmp_path)
    mp_result = _mp_result(1)

    run_ego_motion_on_video_cached("dummy.mp4", mp_result, config, "video_x", cache)
    run_ego_motion_on_video_cached("dummy.mp4", mp_result, config, "video_x", cache)

    assert calls == ["video_x"]


def test_cached_wrapper_recomputes_on_seed_change(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        "hybrid.ego_motion.run_ego_motion_on_video",
        lambda video_path, mp_result, config, video_id: calls.append(video_id) or _fake_ego_result(video_id),
    )
    cache = CacheManager(tmp_path)
    mp_result = _mp_result(1)

    config_a = HybridConfig()
    run_ego_motion_on_video_cached("dummy.mp4", mp_result, config_a, "video_x", cache)

    config_b = HybridConfig()
    config_b.project.seed = 999
    run_ego_motion_on_video_cached("dummy.mp4", mp_result, config_b, "video_x", cache)

    assert calls == ["video_x", "video_x"]
