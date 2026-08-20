import pytest

from hybrid.config import HybridConfig
from hybrid.dataset import DevVideo, GroundTruth
from hybrid.evaluation import VideoEvaluationResult
from hybrid.exceptions import HeldOutVideoError, HybridError
from hybrid.run_development import _dev_video_for_path, guard_development_only, main

GT_CSV_HEADER = "filename,cpr_start_sec,cpr_end_sec,cpr_duration_sec,known_compression_count,gt_cpm,notes\n"


def _write_gt_csv(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        f.write(GT_CSV_HEADER)
        for row in rows:
            f.write(row + "\n")


# ---------------------------------------------------------------------------
# guard_development_only
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "filename",
    ["video1_development.mp4", "personA_session2_development.mp4"],
)
def test_guard_accepts_development_files(filename):
    guard_development_only(filename)  # should not raise


@pytest.mark.parametrize(
    "filename",
    ["video1_test.mp4", "video1.mp4", "video1_development.mov", "random_file.txt"],
)
def test_guard_rejects_non_development_files(filename):
    with pytest.raises(HeldOutVideoError):
        guard_development_only(filename)


# ---------------------------------------------------------------------------
# _dev_video_for_path
# ---------------------------------------------------------------------------


def _make_config(tmp_path):
    metadata_dir = tmp_path / "metadata"
    metadata_dir.mkdir()
    gt_csv = metadata_dir / "ground_truth_summary.csv"
    config = HybridConfig()
    config.paths.ground_truth_csv = gt_csv
    return config, gt_csv


def test_dev_video_for_path_happy_path(tmp_path):
    config, gt_csv = _make_config(tmp_path)
    _write_gt_csv(gt_csv, ["video1.mp4,4.0,24.0,20.0,30,90.0,"])

    dev_video = _dev_video_for_path(tmp_path / "video1_development.mp4", config)

    assert dev_video.video_id == "video1"
    assert dev_video.gt.gt_cpm == pytest.approx(90.0)


def test_dev_video_for_path_rejects_test_video(tmp_path):
    config, gt_csv = _make_config(tmp_path)
    _write_gt_csv(gt_csv, ["video1.mp4,4.0,24.0,20.0,30,90.0,"])

    with pytest.raises(HeldOutVideoError):
        _dev_video_for_path(tmp_path / "video1_test.mp4", config)


def test_dev_video_for_path_missing_gt_row_raises(tmp_path):
    config, gt_csv = _make_config(tmp_path)
    _write_gt_csv(gt_csv, ["video1.mp4,4.0,24.0,20.0,30,90.0,"])

    with pytest.raises(HeldOutVideoError):
        _dev_video_for_path(tmp_path / "video99_development.mp4", config)


# ---------------------------------------------------------------------------
# main()
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


def test_main_rejects_test_video_without_running_pipeline(tmp_path, monkeypatch):
    config, gt_csv = _make_config(tmp_path)
    _write_gt_csv(gt_csv, ["video1.mp4,4.0,24.0,20.0,30,90.0,"])
    monkeypatch.setattr("hybrid.run_development.load_config", lambda: config)
    monkeypatch.setattr("hybrid.run_development.setup_logging", lambda *a, **k: None)

    calls = []
    monkeypatch.setattr(
        "hybrid.run_development.run_full_pipeline_on_video", lambda *a, **k: calls.append(1) or _fake_result("x")
    )

    test_video_path = str(tmp_path / "video1_test.mp4")
    with pytest.raises(HeldOutVideoError):
        main([test_video_path])

    assert calls == []  # pipeline must never have been invoked


def test_main_runs_full_dev_set_and_returns_zero_on_success(tmp_path, monkeypatch):
    config, gt_csv = _make_config(tmp_path)
    monkeypatch.setattr("hybrid.run_development.load_config", lambda: config)
    monkeypatch.setattr("hybrid.run_development.setup_logging", lambda *a, **k: None)

    fake_videos = [
        DevVideo(
            video_id="video1",
            split_path=tmp_path / "video1_development.mp4",
            gt=GroundTruth(
                filename="video1.mp4",
                cpr_start_sec=0.0,
                cpr_end_sec=20.0,
                cpr_duration_sec=20.0,
                known_compression_count=30,
                gt_cpm=90.0,
            ),
        ),
        DevVideo(
            video_id="video2",
            split_path=tmp_path / "video2_development.mp4",
            gt=GroundTruth(
                filename="video2.mp4",
                cpr_start_sec=0.0,
                cpr_end_sec=20.0,
                cpr_duration_sec=20.0,
                known_compression_count=30,
                gt_cpm=80.0,
            ),
        ),
    ]
    monkeypatch.setattr("hybrid.run_development.discover_development_videos", lambda config: fake_videos)
    monkeypatch.setattr(
        "hybrid.run_development.run_full_pipeline_on_video",
        lambda dev_video, config, cache: _fake_result(dev_video.video_id, gt_cpm=dev_video.gt.gt_cpm),
    )

    exit_code = main([])

    assert exit_code == 0


def test_main_returns_nonzero_when_a_video_fails_but_continues_others(tmp_path, monkeypatch):
    config, gt_csv = _make_config(tmp_path)
    monkeypatch.setattr("hybrid.run_development.load_config", lambda: config)
    monkeypatch.setattr("hybrid.run_development.setup_logging", lambda *a, **k: None)

    fake_videos = [
        DevVideo(
            video_id="video1",
            split_path=tmp_path / "video1_development.mp4",
            gt=GroundTruth(
                filename="video1.mp4",
                cpr_start_sec=0.0,
                cpr_end_sec=20.0,
                cpr_duration_sec=20.0,
                known_compression_count=30,
                gt_cpm=90.0,
            ),
        ),
        DevVideo(
            video_id="video2",
            split_path=tmp_path / "video2_development.mp4",
            gt=GroundTruth(
                filename="video2.mp4",
                cpr_start_sec=0.0,
                cpr_end_sec=20.0,
                cpr_duration_sec=20.0,
                known_compression_count=30,
                gt_cpm=80.0,
            ),
        ),
    ]
    monkeypatch.setattr("hybrid.run_development.discover_development_videos", lambda config: fake_videos)

    processed = []

    def fake_pipeline(dev_video, config, cache):
        processed.append(dev_video.video_id)
        if dev_video.video_id == "video1":
            raise HybridError("simulated failure")
        return _fake_result(dev_video.video_id, gt_cpm=dev_video.gt.gt_cpm)

    monkeypatch.setattr("hybrid.run_development.run_full_pipeline_on_video", fake_pipeline)

    exit_code = main([])

    assert exit_code == 1
    assert processed == ["video1", "video2"]  # video2 still ran despite video1's failure
