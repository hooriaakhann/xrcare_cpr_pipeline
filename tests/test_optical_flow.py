import cv2
import numpy as np
import pytest

from hybrid.caching import CacheManager
from hybrid.config import HybridConfig
from hybrid.exceptions import OpticalFlowUnstableError
from hybrid.mediapipe_roi import HandDetection, MediaPipeVideoResult, RoiBox
from hybrid.optical_flow import (
    OpticalFlowVideoResult,
    _compute_frame_flow,
    _identity_frame,
    _optical_flow_config_hash,
    run_optical_flow_on_video,
    run_optical_flow_on_video_cached,
)

# ---------------------------------------------------------------------------
# Pure-logic / real-cv2-on-arrays tests (fast, no video/model I/O)
# ---------------------------------------------------------------------------


def test_identity_frame_is_zero_and_valid():
    f = _identity_frame(0, 0.0)
    assert f.foreground_flow_y == 0.0
    assert f.background_flow_y == 0.0
    assert f.residual_flow_y == 0.0
    assert f.valid is True


def test_optical_flow_config_hash_changes_with_config():
    a = HybridConfig()
    b = HybridConfig()
    assert _optical_flow_config_hash(a) == _optical_flow_config_hash(b)
    b.optical_flow.winsize = 21
    assert _optical_flow_config_hash(a) != _optical_flow_config_hash(b)


def test_compute_frame_flow_separates_foreground_and_background():
    # Large frame/ROI relative to Farneback's smoothing window (winsize=15 +
    # pyramid levels) so the median over each region isn't dominated by
    # boundary blending between the two differently-moving textures.
    rng = np.random.default_rng(2)
    size, pad = 300, 40
    base = rng.integers(0, 255, size=(size + 2 * pad, size + 2 * pad), dtype=np.uint8)

    dy_bg, dy_fg = 3, -4
    prev = base[pad : pad + size, pad : pad + size].copy()
    curr = base[pad + dy_bg : pad + dy_bg + size, pad : pad + size].copy()

    fg_size = 150
    fg_base = rng.integers(0, 255, size=(fg_size + 2 * pad, fg_size + 2 * pad), dtype=np.uint8)
    roi = RoiBox(75, 75, 75 + fg_size, 75 + fg_size)
    prev[roi.y_min : roi.y_max, roi.x_min : roi.x_max] = fg_base[pad : pad + fg_size, pad : pad + fg_size]
    curr[roi.y_min : roi.y_max, roi.x_min : roi.x_max] = fg_base[
        pad + dy_fg : pad + dy_fg + fg_size, pad : pad + fg_size
    ]

    of_config = HybridConfig().optical_flow
    fg_y, bg_y, residual_y, magnitude, valid = _compute_frame_flow(prev, curr, roi, of_config)

    # curr's crop window moved by +dy in the base image -> content appears to
    # shift by -dy in local coordinates (same convention verified in Phase 4's
    # ego-motion translation-recovery test).
    assert bg_y == pytest.approx(-dy_bg, abs=1.5)
    assert fg_y == pytest.approx(-dy_fg, abs=1.5)
    assert residual_y == pytest.approx(-dy_fg - -dy_bg, abs=2.0)
    assert valid is True


def test_compute_frame_flow_recovers_minority_moving_signal_within_mostly_static_roi():
    # Regression test for the real-data finding during Phase 6 development:
    # the Phase 2 ROI is deliberately generous, so most of its pixels are
    # static at any instant and only a small sub-region (the part of the
    # hand mid-stroke) actually moves. A plain median over the whole ROI is
    # dominated by the static majority and reports ~0 regardless of the real
    # motion; the motion_percentile-filtered median must recover the
    # minority's true signal instead.
    # Proportions (~30% moving / 70% static) match what real footage showed
    # during Phase 6 development for the default motion_percentile=75.
    rng = np.random.default_rng(4)
    frame_size, margin = 260, 30
    prev = rng.integers(0, 255, size=(frame_size, frame_size), dtype=np.uint8)
    curr = prev.copy()  # everywhere static by default

    moving_size = 110
    my0, mx0 = margin + 15, margin + 15
    moving_patch = rng.integers(0, 255, size=(moving_size + 20, moving_size + 20), dtype=np.uint8)
    dy_true = 6
    prev[my0 : my0 + moving_size, mx0 : mx0 + moving_size] = moving_patch[10 : 10 + moving_size, 10 : 10 + moving_size]
    curr[my0 : my0 + moving_size, mx0 : mx0 + moving_size] = moving_patch[
        10 - dy_true : 10 - dy_true + moving_size, 10 : 10 + moving_size
    ]

    # ROI covers most of the frame (generous, per Phase 2) but leaves a
    # margin outside it as background, so bg_mask isn't empty.
    roi = RoiBox(margin, margin, frame_size - margin, frame_size - margin)
    of_config = HybridConfig().optical_flow  # default motion_percentile=75

    fg_y, _bg_y, _residual_y, _magnitude, valid = _compute_frame_flow(prev, curr, roi, of_config)

    assert valid is True
    # a plain median (motion_percentile=0, i.e. no filtering) would read ~0
    # here since most of the ROI is static -- the fix must not.
    assert fg_y == pytest.approx(dy_true, abs=2.0)


def test_compute_frame_flow_no_roi_is_invalid():
    frame = np.zeros((50, 50), dtype=np.uint8)
    of_config = HybridConfig().optical_flow
    _fg, _bg, _res, _mag, valid = _compute_frame_flow(frame, frame, None, of_config)
    assert valid is False


def test_compute_frame_flow_flags_unstable_when_magnitude_exceeds_threshold():
    # A large synthetic shift on pure random-noise texture is adversarial for
    # Farneback (no smooth local structure for its polynomial expansion), so
    # it doesn't reliably recover large true displacements -- instead of
    # relying on that, use a real (small, correctly-recovered) flow and just
    # set the threshold below it to exercise the invalid-flagging logic.
    rng = np.random.default_rng(3)
    size, pad = 100, 30
    base = rng.integers(0, 255, size=(size + 2 * pad, size + 2 * pad), dtype=np.uint8)
    dy = 5
    prev = base[pad : pad + size, pad : pad + size].copy()
    curr = base[pad + dy : pad + dy + size, pad : pad + size].copy()
    roi = RoiBox(20, 20, 80, 80)

    of_config = HybridConfig().optical_flow
    of_config.max_flow_magnitude = 0.5  # below the real, correctly-recovered ~5px flow
    _fg, _bg, _res, magnitude, valid = _compute_frame_flow(prev, curr, roi, of_config)

    assert magnitude > 0.5
    assert valid is False


# ---------------------------------------------------------------------------
# Orchestration: monkeypatch _compute_frame_flow for deterministic control
# ---------------------------------------------------------------------------


def _write_synthetic_video(path, num_frames, fps=2.0, width=32, height=32):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    for i in range(num_frames):
        writer.write(np.full((height, width, 3), i % 256, dtype=np.uint8))
    writer.release()


def _detected(frame_index, timestamp_sec, roi=RoiBox(0, 0, 10, 10)):
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


def _mp_result(num_frames):
    detections = [_detected(i, i * 0.5) for i in range(num_frames)]
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


def test_run_optical_flow_first_frame_identity_and_no_unstable(tmp_path, monkeypatch):
    video_path = tmp_path / "v.mp4"
    _write_synthetic_video(video_path, num_frames=5)
    mp_result = _mp_result(5)

    monkeypatch.setattr(
        "hybrid.optical_flow._compute_frame_flow",
        lambda prev, curr, roi, of_config: (1.0, 0.5, 0.5, 2.0, True),
    )

    config = HybridConfig()
    result = run_optical_flow_on_video(video_path, mp_result, config, "video_x")

    assert len(result.frames) == 5
    assert result.frames[0].valid is True
    assert result.frames[0].residual_flow_y == 0.0
    assert result.unstable_periods == []
    assert result.flow_motion_y.shape[0] == 5
    assert result.flow_motion_y[1] == pytest.approx(0.5)


def test_run_optical_flow_raises_when_sustained_unstable(tmp_path, monkeypatch):
    video_path = tmp_path / "v.mp4"
    _write_synthetic_video(video_path, num_frames=6, fps=2.0)
    mp_result = _mp_result(6)

    monkeypatch.setattr(
        "hybrid.optical_flow._compute_frame_flow",
        lambda prev, curr, roi, of_config: (0.0, 0.0, 0.0, 999.0, False),
    )

    config = HybridConfig()
    config.optical_flow.max_unstable_span_sec = 1.0

    with pytest.raises(OpticalFlowUnstableError):
        run_optical_flow_on_video(video_path, mp_result, config, "video_x")


# ---------------------------------------------------------------------------
# Caching wrapper
# ---------------------------------------------------------------------------


def _fake_flow_result(video_id="video_x"):
    return OpticalFlowVideoResult(
        video_id=video_id,
        frames=[_identity_frame(0, 0.0)],
        flow_motion_y=np.array([0.0]),
        unstable_periods=[],
        longest_unstable_sec=0.0,
    )


def test_cached_wrapper_avoids_recompute_on_hit(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        "hybrid.optical_flow.run_optical_flow_on_video",
        lambda video_path, mp_result, config, video_id: calls.append(video_id) or _fake_flow_result(video_id),
    )
    config = HybridConfig()
    cache = CacheManager(tmp_path)
    mp_result = _mp_result(1)

    run_optical_flow_on_video_cached("dummy.mp4", mp_result, config, "video_x", cache)
    run_optical_flow_on_video_cached("dummy.mp4", mp_result, config, "video_x", cache)

    assert calls == ["video_x"]


def test_cached_wrapper_recomputes_on_config_change(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        "hybrid.optical_flow.run_optical_flow_on_video",
        lambda video_path, mp_result, config, video_id: calls.append(video_id) or _fake_flow_result(video_id),
    )
    cache = CacheManager(tmp_path)
    mp_result = _mp_result(1)

    config_a = HybridConfig()
    run_optical_flow_on_video_cached("dummy.mp4", mp_result, config_a, "video_x", cache)

    config_b = HybridConfig()
    config_b.optical_flow.winsize = 21
    run_optical_flow_on_video_cached("dummy.mp4", mp_result, config_b, "video_x", cache)

    assert calls == ["video_x", "video_x"]
