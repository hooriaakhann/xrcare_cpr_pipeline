import pytest

from hybrid.config import HybridConfig
from hybrid.dataset import DevVideo, GroundTruth
from hybrid.evaluation import VideoEvaluationResult
from hybrid.exceptions import HeldOutVideoError, HybridError
from hybrid.run_test import _aggregate_metrics, _test_video_for_path, guard_test_only, main

GT_CSV_HEADER = "filename,cpr_start_sec,cpr_end_sec,cpr_duration_sec,known_compression_count,gt_cpm,notes\n"


def _write_gt_csv(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        f.write(GT_CSV_HEADER)
        for row in rows:
            f.write(row + "\n")


# ---------------------------------------------------------------------------
# guard_test_only
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "filename",
    ["video5_test.mp4", "personA_session2_test.mp4"],
)
def test_guard_accepts_test_files(filename):
    guard_test_only(filename)  # should not raise


@pytest.mark.parametrize(
    "filename",
    ["video5_development.mp4", "video5.mp4", "video5_test.mov", "random_file.txt"],
)
def test_guard_rejects_non_test_files(filename):
    with pytest.raises(HeldOutVideoError):
        guard_test_only(filename)


# ---------------------------------------------------------------------------
# _test_video_for_path
# ---------------------------------------------------------------------------


def _make_config(tmp_path):
    metadata_dir = tmp_path / "metadata"
    metadata_dir.mkdir()
    gt_csv = metadata_dir / "ground_truth_summary.csv"
    config = HybridConfig()
    config.paths.ground_truth_csv = gt_csv
    return config, gt_csv


def test_test_video_for_path_happy_path(tmp_path):
    config, gt_csv = _make_config(tmp_path)
    _write_gt_csv(gt_csv, ["video5.mp4,4.0,24.0,20.0,30,90.0,"])

    test_video = _test_video_for_path(tmp_path / "video5_test.mp4", config)

    assert test_video.video_id == "video5"
    assert test_video.gt.gt_cpm == pytest.approx(90.0)


def test_test_video_for_path_rejects_development_video(tmp_path):
    config, gt_csv = _make_config(tmp_path)
    _write_gt_csv(gt_csv, ["video5.mp4,4.0,24.0,20.0,30,90.0,"])

    with pytest.raises(HeldOutVideoError):
        _test_video_for_path(tmp_path / "video5_development.mp4", config)


def test_test_video_for_path_missing_gt_row_raises(tmp_path):
    config, gt_csv = _make_config(tmp_path)
    _write_gt_csv(gt_csv, ["video5.mp4,4.0,24.0,20.0,30,90.0,"])

    with pytest.raises(HeldOutVideoError):
        _test_video_for_path(tmp_path / "video99_test.mp4", config)


# ---------------------------------------------------------------------------
# _aggregate_metrics
# ---------------------------------------------------------------------------


def _fake_result(video_id, gt_cpm=90.0, final_cpm=91.0):
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
        runtime_sec=1.0,
    )


def test_aggregate_metrics_matches_hand_computation():
    per_video = [
        _fake_result("video5", gt_cpm=90.0, final_cpm=92.0),  # signed +2, abs 2
        _fake_result("video6", gt_cpm=80.0, final_cpm=76.0),  # signed -4, abs 4
    ]

    m = _aggregate_metrics(per_video, seed=42, num_bootstrap_resamples=200)

    assert m["mae"] == pytest.approx(3.0)
    assert m["rmse"] == pytest.approx(((2.0**2 + 4.0**2) / 2) ** 0.5)
    assert m["mean_signed_error"] == pytest.approx(-1.0)
    assert m["median_absolute_error"] == pytest.approx(3.0)
    assert m["max_absolute_error"] == pytest.approx(4.0)
    # bootstrap-resampled from {2.0, 4.0} with replacement -- CI must sit within that range
    assert 2.0 <= m["mae_bootstrap_ci_95_lower"] <= m["mae_bootstrap_ci_95_upper"] <= 4.0


def test_aggregate_metrics_ci_is_deterministic_for_a_given_seed():
    per_video = [
        _fake_result("video5", gt_cpm=90.0, final_cpm=92.0),
        _fake_result("video6", gt_cpm=80.0, final_cpm=76.0),
        _fake_result("video7", gt_cpm=100.0, final_cpm=101.0),
    ]

    m1 = _aggregate_metrics(per_video, seed=42, num_bootstrap_resamples=500)
    m2 = _aggregate_metrics(per_video, seed=42, num_bootstrap_resamples=500)

    assert m1["mae_bootstrap_ci_95_lower"] == m2["mae_bootstrap_ci_95_lower"]
    assert m1["mae_bootstrap_ci_95_upper"] == m2["mae_bootstrap_ci_95_upper"]


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------


def test_main_rejects_development_video_without_running_pipeline(tmp_path, monkeypatch):
    config, gt_csv = _make_config(tmp_path)
    _write_gt_csv(gt_csv, ["video5.mp4,4.0,24.0,20.0,30,90.0,"])
    monkeypatch.setattr("hybrid.run_test.load_config", lambda *a, **k: config)
    monkeypatch.setattr("hybrid.run_test.setup_logging", lambda *a, **k: None)

    calls = []
    monkeypatch.setattr(
        "hybrid.run_test.run_full_pipeline_on_video", lambda *a, **k: calls.append(1) or _fake_result("x")
    )

    dev_video_path = str(tmp_path / "video5_development.mp4")
    with pytest.raises(HeldOutVideoError):
        main([dev_video_path])

    assert calls == []  # pipeline must never have been invoked


def test_main_runs_full_test_set_and_returns_zero_on_success(tmp_path, monkeypatch):
    config, gt_csv = _make_config(tmp_path)
    monkeypatch.setattr("hybrid.run_test.load_config", lambda *a, **k: config)
    monkeypatch.setattr("hybrid.run_test.setup_logging", lambda *a, **k: None)
    monkeypatch.setattr("hybrid.run_test.log_run", lambda **k: None)

    fake_videos = [
        DevVideo(
            video_id="video5",
            split_path=tmp_path / "video5_test.mp4",
            gt=GroundTruth(
                filename="video5.mp4",
                cpr_start_sec=0.0,
                cpr_end_sec=20.0,
                cpr_duration_sec=20.0,
                known_compression_count=30,
                gt_cpm=90.0,
            ),
        ),
        DevVideo(
            video_id="video6",
            split_path=tmp_path / "video6_test.mp4",
            gt=GroundTruth(
                filename="video6.mp4",
                cpr_start_sec=0.0,
                cpr_end_sec=20.0,
                cpr_duration_sec=20.0,
                known_compression_count=30,
                gt_cpm=80.0,
            ),
        ),
    ]
    monkeypatch.setattr("hybrid.run_test.discover_test_videos", lambda config: fake_videos)
    monkeypatch.setattr(
        "hybrid.run_test.run_full_pipeline_on_video",
        lambda dev_video, config, cache: _fake_result(dev_video.video_id, gt_cpm=dev_video.gt.gt_cpm),
    )

    exit_code = main([])

    assert exit_code == 0


def test_main_returns_nonzero_when_a_video_fails_but_continues_others(tmp_path, monkeypatch):
    config, gt_csv = _make_config(tmp_path)
    monkeypatch.setattr("hybrid.run_test.load_config", lambda *a, **k: config)
    monkeypatch.setattr("hybrid.run_test.setup_logging", lambda *a, **k: None)
    monkeypatch.setattr("hybrid.run_test.log_run", lambda **k: None)

    fake_videos = [
        DevVideo(
            video_id="video5",
            split_path=tmp_path / "video5_test.mp4",
            gt=GroundTruth(
                filename="video5.mp4",
                cpr_start_sec=0.0,
                cpr_end_sec=20.0,
                cpr_duration_sec=20.0,
                known_compression_count=30,
                gt_cpm=90.0,
            ),
        ),
        DevVideo(
            video_id="video6",
            split_path=tmp_path / "video6_test.mp4",
            gt=GroundTruth(
                filename="video6.mp4",
                cpr_start_sec=0.0,
                cpr_end_sec=20.0,
                cpr_duration_sec=20.0,
                known_compression_count=30,
                gt_cpm=80.0,
            ),
        ),
    ]
    monkeypatch.setattr("hybrid.run_test.discover_test_videos", lambda config: fake_videos)

    processed = []

    def fake_pipeline(dev_video, config, cache):
        processed.append(dev_video.video_id)
        if dev_video.video_id == "video5":
            raise HybridError("simulated failure")
        return _fake_result(dev_video.video_id, gt_cpm=dev_video.gt.gt_cpm)

    monkeypatch.setattr("hybrid.run_test.run_full_pipeline_on_video", fake_pipeline)

    exit_code = main([])

    assert exit_code == 1
    assert processed == ["video5", "video6"]  # video6 still ran despite video5's failure


def test_main_loads_frozen_config_path_explicitly(tmp_path, monkeypatch):
    """The one property this script exists to guarantee: it must always
    load config/frozen.yaml, never config/default.yaml, regardless of
    what default.yaml contains by the time this runs."""
    from hybrid.run_test import FROZEN_CONFIG_PATH

    config, gt_csv = _make_config(tmp_path)
    seen_paths = []

    def fake_load_config(path=None, **k):
        seen_paths.append(path)
        return config

    monkeypatch.setattr("hybrid.run_test.load_config", fake_load_config)
    monkeypatch.setattr("hybrid.run_test.setup_logging", lambda *a, **k: None)
    monkeypatch.setattr("hybrid.run_test.discover_test_videos", lambda config: [])

    main([])

    assert seen_paths == [FROZEN_CONFIG_PATH]
    assert FROZEN_CONFIG_PATH.name == "frozen.yaml"
