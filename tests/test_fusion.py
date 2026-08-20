import numpy as np
import pytest

from hybrid.config import HybridConfig
from hybrid.estimators import AllEstimatesResult, AutocorrelationEstimate, CwtEstimate, FftEstimate, PeakEstimate
from hybrid.exceptions import FusionError
from hybrid.fusion import _weighted_median, fuse_estimates
from hybrid.repnet_branch import RepNetResult


def test_weighted_median_odd_count_uniform_weights():
    values = np.array([10.0, 30.0, 20.0])
    weights = np.array([1.0, 1.0, 1.0])
    assert _weighted_median(values, weights) == pytest.approx(20.0)


def test_weighted_median_skews_toward_heavier_weight():
    values = np.array([10.0, 20.0, 100.0])
    weights = np.array([1.0, 1.0, 10.0])
    # heavy weight on 100 should pull the weighted median there
    assert _weighted_median(values, weights) == pytest.approx(100.0)


def _estimates(
    cwt_cpm, autocorr_cpm, fft_cpm, peaks_cpm, cwt_conf=0.9, autocorr_conf=0.9, fft_conf=0.9, peaks_conf=0.9
):
    return AllEstimatesResult(
        video_id="v",
        cwt=CwtEstimate(dominant_freq_hz=cwt_cpm / 60, cpm=cwt_cpm, confidence=cwt_conf),
        autocorrelation=AutocorrelationEstimate(
            dominant_lag_sec=60 / autocorr_cpm, cpm=autocorr_cpm, confidence=autocorr_conf
        ),
        fft=FftEstimate(dominant_freq_hz=fft_cpm / 60, cpm=fft_cpm, confidence=fft_conf),
        peaks=PeakEstimate(
            peak_timestamps_sec=np.array([0.0, 1.0]),
            num_peaks=2,
            median_inter_peak_interval_sec=1.0,
            cpm=peaks_cpm,
            confidence=peaks_conf,
        ),
    )


def _repnet(cpm, confidence, video_id="v"):
    return RepNetResult(
        video_id=video_id,
        cpm=cpm,
        confidence=confidence,
        pred_period_frames=20.0,
        chosen_stride=1,
        fps=30.0,
        num_frames=500,
        reason=None if cpm is not None else "no periodicity",
    )


def test_fuse_estimates_full_agreement_high_confidence():
    estimates = _estimates(100.0, 100.0, 100.0, 100.0)
    repnet = _repnet(100.0, 0.9)
    config = HybridConfig()

    result = fuse_estimates(estimates, repnet, config, "v")

    assert result.final_cpm == pytest.approx(100.0, abs=0.5)
    assert result.overall_confidence > 0.8
    assert result.spread_cpm == pytest.approx(0.0)


def test_fuse_estimates_low_confidence_outlier_does_not_pull_final_value():
    # spec's own worked example: CWT~100, autocorr~101, FFT~100, peaks~102,
    # RepNet~65 with low confidence -- 65 should not pull final strongly down
    estimates = _estimates(100.0, 101.0, 100.0, 102.0, cwt_conf=0.9, autocorr_conf=0.9, fft_conf=0.9, peaks_conf=0.9)
    repnet = _repnet(65.0, 0.15)
    config = HybridConfig()

    result = fuse_estimates(estimates, repnet, config, "v")

    assert result.final_cpm > 95.0  # nowhere near being dragged toward 65
    repnet_candidate = next(c for c in result.candidates if c.name == "repnet")
    assert repnet_candidate.final_weight < 0.1  # discounted by both confidence and disagreement


def test_fuse_estimates_repnet_none_excluded_but_recorded():
    estimates = _estimates(100.0, 100.0, 100.0, 100.0)
    repnet = _repnet(None, 0.0)
    config = HybridConfig()

    result = fuse_estimates(estimates, repnet, config, "v")

    repnet_candidate = next(c for c in result.candidates if c.name == "repnet")
    assert repnet_candidate.cpm is None
    assert repnet_candidate.final_weight == 0.0
    assert result.final_cpm == pytest.approx(100.0, abs=0.5)
    assert len(result.candidates) == 5  # still recorded, just excluded from the average


def test_fuse_estimates_all_zero_confidence_falls_back_to_unweighted_mean():
    estimates = _estimates(90.0, 100.0, 110.0, 100.0, cwt_conf=0.0, autocorr_conf=0.0, fft_conf=0.0, peaks_conf=0.0)
    repnet = _repnet(100.0, 0.0)
    config = HybridConfig()

    result = fuse_estimates(estimates, repnet, config, "v")

    expected_mean = np.mean([90.0, 100.0, 110.0, 100.0, 100.0])
    assert result.final_cpm == pytest.approx(expected_mean)
    assert result.overall_confidence == 0.0


def test_fuse_estimates_spread_and_std_computed_correctly():
    estimates = _estimates(90.0, 100.0, 110.0, 100.0)
    repnet = _repnet(100.0, 0.5)
    config = HybridConfig()

    result = fuse_estimates(estimates, repnet, config, "v")

    all_cpms = [90.0, 100.0, 110.0, 100.0, 100.0]
    assert result.spread_cpm == pytest.approx(max(all_cpms) - min(all_cpms))
    assert result.std_cpm == pytest.approx(np.std(all_cpms))


def test_fuse_estimates_raises_when_no_usable_candidates():
    estimates = _estimates(100.0, 100.0, 100.0, 100.0)
    # simulate every candidate being unusable, including the normally-always-valid classical ones
    object.__setattr__(estimates.cwt, "cpm", None)
    object.__setattr__(estimates.autocorrelation, "cpm", None)
    object.__setattr__(estimates.fft, "cpm", None)
    object.__setattr__(estimates.peaks, "cpm", None)
    repnet = _repnet(None, 0.0)
    config = HybridConfig()

    with pytest.raises(FusionError):
        fuse_estimates(estimates, repnet, config, "v")
