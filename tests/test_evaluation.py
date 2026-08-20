from types import SimpleNamespace

import numpy as np
import pytest

from hybrid.caching import CacheManager
from hybrid.config import HybridConfig
from hybrid.dataset import DevVideo, GroundTruth
from hybrid.estimators import AllEstimatesResult, AutocorrelationEstimate, CwtEstimate, FftEstimate, PeakEstimate
from hybrid.evaluation import (
    VideoEvaluationResult,
    _bootstrap_mae_ci,
    run_development_evaluation,
    run_full_pipeline_on_video,
)
from hybrid.fusion import FusionResult
from hybrid.repnet_branch import RepNetResult

# ---------------------------------------------------------------------------
# _bootstrap_mae_ci
# ---------------------------------------------------------------------------


def test_bootstrap_mae_ci_constant_errors_gives_tight_ci():
    errors = np.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0])
    lower, upper = _bootstrap_mae_ci(errors, num_resamples=500, seed=42)
    assert lower == pytest.approx(5.0)
    assert upper == pytest.approx(5.0)


def test_bootstrap_mae_ci_brackets_true_mean_for_varied_errors():
    errors = np.array([1.0, 2.0, 3.0, 10.0, 15.0, 2.0])
    lower, upper = _bootstrap_mae_ci(errors, num_resamples=2000, seed=42)
    true_mean = errors.mean()
    assert lower <= true_mean <= upper
    assert lower < upper


def test_bootstrap_mae_ci_deterministic_given_seed():
    errors = np.array([1.0, 4.0, 9.0, 2.0])
    a = _bootstrap_mae_ci(errors, num_resamples=500, seed=7)
    b = _bootstrap_mae_ci(errors, num_resamples=500, seed=7)
    assert a == b


# ---------------------------------------------------------------------------
# run_full_pipeline_on_video: monkeypatch every stage so this stays fast
# ---------------------------------------------------------------------------


def _dev_video(video_id, gt_cpm, tmp_path):
    gt = GroundTruth(
        filename=f"{video_id}.mp4",
        cpr_start_sec=0.0,
        cpr_end_sec=20.0,
        cpr_duration_sec=20.0,
        known_compression_count=30,
        gt_cpm=gt_cpm,
    )
    return DevVideo(video_id=video_id, split_path=tmp_path / f"{video_id}_development.mp4", gt=gt)


def _patch_pipeline_stages(monkeypatch, final_cpm, repnet_cpm, overall_confidence=0.7):
    monkeypatch.setattr("hybrid.evaluation.run_mediapipe_on_video_cached", lambda *a, **k: "mp")
    monkeypatch.setattr("hybrid.evaluation.run_cotracker_on_video_cached", lambda *a, **k: "ct")
    monkeypatch.setattr("hybrid.evaluation.run_ego_motion_on_video_cached", lambda *a, **k: "ego")
    monkeypatch.setattr("hybrid.evaluation.correct_tracker_trajectory", lambda *a, **k: "corrected")
    monkeypatch.setattr("hybrid.evaluation.run_optical_flow_on_video_cached", lambda *a, **k: "flow")

    fake_wave = SimpleNamespace(timestamps_sec=np.arange(10) * 0.1, motion_wave=np.zeros(10))
    monkeypatch.setattr("hybrid.evaluation.generate_motion_wave", lambda *a, **k: fake_wave)
    monkeypatch.setattr("hybrid.evaluation.apply_butterworth_filter", lambda *a, **k: "filt")

    fake_estimates = AllEstimatesResult(
        video_id="v",
        cwt=CwtEstimate(dominant_freq_hz=1.5, cpm=90.0, confidence=0.5),
        autocorrelation=AutocorrelationEstimate(dominant_lag_sec=0.6, cpm=91.0, confidence=0.6),
        fft=FftEstimate(dominant_freq_hz=1.5, cpm=89.0, confidence=0.7),
        peaks=PeakEstimate(
            peak_timestamps_sec=np.array([0.0, 1.0]),
            num_peaks=2,
            median_inter_peak_interval_sec=0.6,
            cpm=92.0,
            confidence=0.8,
        ),
    )
    monkeypatch.setattr("hybrid.evaluation.run_estimators_on_video", lambda *a, **k: fake_estimates)

    fake_repnet = RepNetResult(
        video_id="v",
        cpm=repnet_cpm,
        confidence=0.4,
        pred_period_frames=20.0,
        chosen_stride=1,
        fps=30.0,
        num_frames=500,
        reason=None if repnet_cpm is not None else "no periodicity",
    )
    monkeypatch.setattr("hybrid.evaluation.run_repnet_on_video_cached", lambda *a, **k: fake_repnet)

    fake_fusion = FusionResult(
        video_id="v",
        candidates=[],
        center_cpm=90.0,
        final_cpm=final_cpm,
        overall_confidence=overall_confidence,
        spread_cpm=7.0,
        std_cpm=2.5,
    )
    monkeypatch.setattr("hybrid.evaluation.fuse_estimates", lambda *a, **k: fake_fusion)

    return fake_estimates, fake_repnet, fake_fusion


def test_run_full_pipeline_on_video_assembles_result_correctly(monkeypatch, tmp_path):
    _patch_pipeline_stages(monkeypatch, final_cpm=91.0, repnet_cpm=85.0)
    dev_video = _dev_video("video1", gt_cpm=90.0, tmp_path=tmp_path)
    config = HybridConfig()
    cache = CacheManager(tmp_path)

    result = run_full_pipeline_on_video(dev_video, config, cache)

    assert isinstance(result, VideoEvaluationResult)
    assert result.video_id == "video1"
    assert result.gt_cpm == pytest.approx(90.0)
    assert result.final_cpm == pytest.approx(91.0)
    assert result.signed_error == pytest.approx(1.0)
    assert result.absolute_error == pytest.approx(1.0)
    assert result.repnet_cpm == pytest.approx(85.0)
    assert result.cwt_cpm == pytest.approx(90.0)
    assert result.runtime_sec >= 0.0


def test_run_full_pipeline_on_video_negative_signed_error(monkeypatch, tmp_path):
    _patch_pipeline_stages(monkeypatch, final_cpm=80.0, repnet_cpm=None)
    dev_video = _dev_video("video2", gt_cpm=90.0, tmp_path=tmp_path)
    config = HybridConfig()
    cache = CacheManager(tmp_path)

    result = run_full_pipeline_on_video(dev_video, config, cache)

    assert result.signed_error == pytest.approx(-10.0)
    assert result.absolute_error == pytest.approx(10.0)
    assert result.repnet_cpm is None


# ---------------------------------------------------------------------------
# run_development_evaluation
# ---------------------------------------------------------------------------


def test_run_development_evaluation_aggregates_correctly(monkeypatch, tmp_path):
    fake_videos = [_dev_video(f"video{i}", gt_cpm=90.0, tmp_path=tmp_path) for i in range(3)]
    monkeypatch.setattr("hybrid.evaluation.discover_development_videos", lambda config: fake_videos)

    fake_results = [
        VideoEvaluationResult(
            video_id="video0",
            gt_cpm=90.0,
            cwt_cpm=90.0,
            autocorrelation_cpm=90.0,
            fft_cpm=90.0,
            peaks_cpm=90.0,
            repnet_cpm=90.0,
            final_cpm=92.0,
            overall_confidence=0.8,
            signed_error=2.0,
            absolute_error=2.0,
            runtime_sec=1.0,
        ),
        VideoEvaluationResult(
            video_id="video1",
            gt_cpm=90.0,
            cwt_cpm=90.0,
            autocorrelation_cpm=90.0,
            fft_cpm=90.0,
            peaks_cpm=90.0,
            repnet_cpm=90.0,
            final_cpm=86.0,
            overall_confidence=0.7,
            signed_error=-4.0,
            absolute_error=4.0,
            runtime_sec=1.0,
        ),
        VideoEvaluationResult(
            video_id="video2",
            gt_cpm=90.0,
            cwt_cpm=90.0,
            autocorrelation_cpm=90.0,
            fft_cpm=90.0,
            peaks_cpm=90.0,
            repnet_cpm=90.0,
            final_cpm=96.0,
            overall_confidence=0.6,
            signed_error=6.0,
            absolute_error=6.0,
            runtime_sec=1.0,
        ),
    ]
    results_iter = iter(fake_results)
    monkeypatch.setattr(
        "hybrid.evaluation.run_full_pipeline_on_video", lambda dev_video, config, cache: next(results_iter)
    )

    logged = []
    monkeypatch.setattr("hybrid.evaluation.log_run", lambda **kwargs: logged.append(kwargs))

    config = HybridConfig()
    cache = CacheManager(tmp_path)
    result = run_development_evaluation(config, cache, num_bootstrap_resamples=200)

    assert result.mae == pytest.approx((2.0 + 4.0 + 6.0) / 3)
    assert result.rmse == pytest.approx(np.sqrt((4.0 + 16.0 + 36.0) / 3))
    assert result.mean_signed_error == pytest.approx((2.0 - 4.0 + 6.0) / 3)
    assert result.median_absolute_error == pytest.approx(4.0)
    assert result.max_absolute_error == pytest.approx(6.0)
    assert len(result.per_video) == 3

    assert len(logged) == 1
    assert logged[0]["metrics"]["dev_mae"] == pytest.approx(result.mae)
    assert logged[0]["metrics"]["num_videos"] == 3
    assert logged[0]["extra"]["video_ids"] == ["video0", "video1", "video2"]
