from pathlib import Path

import cv2
import numpy as np
import pytest

from hybrid.caching import CacheManager
from hybrid.config import load_config
from hybrid.dataset import discover_development_videos
from hybrid.mediapipe_roi import HandDetection
from hybrid.mediapipe_roi import RoiBox as _RoiBox
from hybrid.overlay_video import (
    _draw_cotracker_overlay,
    _draw_mediapipe_overlay,
    _polyline_segments,
    _render_motion_strip_base,
    save_overlay_video_for_video,
)

# ---------------------------------------------------------------------------
# _draw_mediapipe_overlay
# ---------------------------------------------------------------------------


def test_draw_mediapipe_overlay_draws_roi_when_detected():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    detection = HandDetection(
        frame_index=0,
        timestamp_sec=0.0,
        detected=True,
        roi=_RoiBox(x_min=10, y_min=10, x_max=50, y_max=50),
    )

    _draw_mediapipe_overlay(image, detection, point_radius_px=4, box_thickness_px=2)

    assert image.any()  # something was drawn


def test_draw_mediapipe_overlay_no_op_when_not_detected():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    detection = HandDetection(frame_index=0, timestamp_sec=0.0, detected=False)

    _draw_mediapipe_overlay(image, detection, point_radius_px=4, box_thickness_px=2)

    assert not image.any()  # untouched


def test_draw_mediapipe_overlay_draws_landmarks():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    detection = HandDetection(
        frame_index=0,
        timestamp_sec=0.0,
        detected=True,
        landmarks_xy=((50.0, 50.0),),
    )

    _draw_mediapipe_overlay(image, detection, point_radius_px=4, box_thickness_px=2)

    assert image.any()


# ---------------------------------------------------------------------------
# _draw_cotracker_overlay
# ---------------------------------------------------------------------------


def test_draw_cotracker_overlay_colors_visible_and_occluded_differently():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    xs = np.array([20.0, 80.0])
    ys = np.array([20.0, 80.0])
    visible = np.array([True, False])

    _draw_cotracker_overlay(image, xs, ys, visible, point_radius_px=4)

    visible_pixel = image[20, 20]
    occluded_pixel = image[80, 80]
    assert visible_pixel.any()
    assert occluded_pixel.any()
    assert not np.array_equal(visible_pixel, occluded_pixel)  # different colors


def test_draw_cotracker_overlay_skips_nonfinite_points():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    xs = np.array([np.nan])
    ys = np.array([np.nan])
    visible = np.array([True])

    _draw_cotracker_overlay(image, xs, ys, visible, point_radius_px=4)  # must not raise

    assert not image.any()


# ---------------------------------------------------------------------------
# _polyline_segments
# ---------------------------------------------------------------------------


def test_polyline_segments_single_run_when_all_finite():
    timestamps = np.array([0.0, 1.0, 2.0, 3.0])
    values = np.array([0.0, 1.0, 0.5, 1.0])

    segments = _polyline_segments(
        timestamps, values, duration_sec=3.0, width=100, y_top=0, y_bottom=50, v_min=0.0, v_max=1.0
    )

    assert len(segments) == 1
    assert len(segments[0]) == 4


def test_polyline_segments_splits_at_nan_gap():
    timestamps = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    values = np.array([0.0, 1.0, np.nan, 0.5, 1.0])

    segments = _polyline_segments(
        timestamps, values, duration_sec=4.0, width=100, y_top=0, y_bottom=50, v_min=0.0, v_max=1.0
    )

    assert len(segments) == 2
    assert len(segments[0]) == 2
    assert len(segments[1]) == 2


def test_polyline_segments_empty_when_all_nan():
    timestamps = np.array([0.0, 1.0])
    values = np.array([np.nan, np.nan])

    segments = _polyline_segments(
        timestamps, values, duration_sec=1.0, width=100, y_top=0, y_bottom=50, v_min=0.0, v_max=1.0
    )

    assert segments == []


# ---------------------------------------------------------------------------
# _render_motion_strip_base
# ---------------------------------------------------------------------------


def test_render_motion_strip_base_shape_and_dtype():
    timestamps = np.linspace(0, 10, 50)
    raw = np.sin(timestamps)
    corrected = np.cos(timestamps)

    strip = _render_motion_strip_base(200, 150, timestamps, raw, corrected, duration_sec=10.0)

    assert strip.shape == (150, 200, 3)
    assert strip.dtype == np.uint8
    assert strip.any()  # curves were actually drawn


def test_render_motion_strip_base_handles_all_nan_gracefully():
    timestamps = np.linspace(0, 10, 50)
    raw = np.full(50, np.nan)
    corrected = np.full(50, np.nan)

    strip = _render_motion_strip_base(200, 150, timestamps, raw, corrected, duration_sec=10.0)  # must not raise

    assert strip.shape == (150, 200, 3)


# ---------------------------------------------------------------------------
# save_overlay_video_for_video -- real video, real cached branches (Phase 19 pattern)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_save_overlay_video_end_to_end_on_real_video():
    split_dir = Path(__file__).resolve().parents[1] / "data" / "split"
    if not split_dir.exists():
        pytest.skip("real split videos not present locally (data/split is gitignored)")

    config = load_config()
    cache_manager = CacheManager(config.paths.cache_dir)
    dev_videos = discover_development_videos(config)
    dev_video = next((v for v in dev_videos if v.video_id == "video3"), dev_videos[0])

    output_path = save_overlay_video_for_video(dev_video, config, cache_manager)

    assert output_path.exists()
    assert output_path.suffix == ".mp4"

    cap = cv2.VideoCapture(str(output_path))
    try:
        assert cap.isOpened()
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        assert width > 0
        assert height > config.overlay_video.motion_strip_height_px  # frame height + strip
        assert frame_count > 0
    finally:
        cap.release()
