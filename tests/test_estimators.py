import numpy as np
import pytest

from hybrid.config import HybridConfig
from hybrid.estimators import (
    AllEstimatesResult,
    estimate_autocorrelation,
    estimate_cwt,
    estimate_fft,
    estimate_peaks,
    run_estimators_on_video,
)
from hybrid.exceptions import EstimatorError
from hybrid.filters import FilterResult

# ---------------------------------------------------------------------------
# Synthetic-signal fixtures (spec: known-frequency sine, noisy version,
# version with dropped cycles -- one per estimator, all four required
# before any estimator is trusted on real video).
# ---------------------------------------------------------------------------

FS = 30.0
DURATION_SEC = 30.0
TRUE_CPM = 100.0
TRUE_FREQ_HZ = TRUE_CPM / 60.0  # 1.6667 Hz
TOLERANCE_CPM = 2.0
DEGRADED_TOLERANCE_CPM = 5.0


def _synthetic_sine(freq_hz=TRUE_FREQ_HZ, fs=FS, duration_sec=DURATION_SEC, amplitude=1.0):
    t = np.arange(0, duration_sec, 1.0 / fs)
    return t, amplitude * np.sin(2 * np.pi * freq_hz * t)


def _noisy(signal, rng, std=0.3):
    return signal + rng.normal(0, std, size=signal.shape)


def _drop_cycles(t, signal, freq_hz=TRUE_FREQ_HZ, num_drops=2, seed=1):
    rng = np.random.default_rng(seed)
    period = 1.0 / freq_hz
    out = signal.copy()
    duration = t[-1]
    starts = rng.uniform(period * 3, duration - period * 3, size=num_drops)
    for start in sorted(starts):
        mask = (t >= start) & (t < start + period)
        out[mask] = 0.0
    return out


@pytest.fixture
def e_config():
    return HybridConfig().estimators


# ---------------------------------------------------------------------------
# CWT
# ---------------------------------------------------------------------------


def test_cwt_recovers_clean_sine_cpm(e_config):
    _t, sig = _synthetic_sine()
    est = estimate_cwt(sig, FS, e_config)
    assert est.cpm == pytest.approx(TRUE_CPM, abs=TOLERANCE_CPM)
    assert 0.0 <= est.confidence <= 1.0


def test_cwt_recovers_noisy_sine_cpm(e_config):
    rng = np.random.default_rng(0)
    _t, sig = _synthetic_sine()
    sig = _noisy(sig, rng)
    est = estimate_cwt(sig, FS, e_config)
    assert est.cpm == pytest.approx(TRUE_CPM, abs=DEGRADED_TOLERANCE_CPM)


def test_cwt_recovers_dropped_cycles_cpm(e_config):
    t, sig = _synthetic_sine()
    sig = _drop_cycles(t, sig)
    est = estimate_cwt(sig, FS, e_config)
    assert est.cpm == pytest.approx(TRUE_CPM, abs=DEGRADED_TOLERANCE_CPM)


def test_cwt_raises_when_band_out_of_range(e_config):
    # FS=30Hz -> Nyquist=15Hz; a band at/beyond that is unreachable
    _t, sig = _synthetic_sine()
    with pytest.raises(EstimatorError):
        estimate_cwt(sig, FS, e_config.model_copy(update={"cwt_freq_range_hz": (16.0, 20.0)}))


# ---------------------------------------------------------------------------
# Autocorrelation
# ---------------------------------------------------------------------------


def test_autocorrelation_recovers_clean_sine_cpm(e_config):
    _t, sig = _synthetic_sine()
    est = estimate_autocorrelation(sig, FS, e_config)
    assert est.cpm == pytest.approx(TRUE_CPM, abs=TOLERANCE_CPM)
    assert 0.0 <= est.confidence <= 1.0


def test_autocorrelation_recovers_noisy_sine_cpm(e_config):
    rng = np.random.default_rng(0)
    _t, sig = _synthetic_sine()
    sig = _noisy(sig, rng)
    est = estimate_autocorrelation(sig, FS, e_config)
    assert est.cpm == pytest.approx(TRUE_CPM, abs=DEGRADED_TOLERANCE_CPM)


def test_autocorrelation_recovers_dropped_cycles_cpm(e_config):
    t, sig = _synthetic_sine()
    sig = _drop_cycles(t, sig)
    est = estimate_autocorrelation(sig, FS, e_config)
    assert est.cpm == pytest.approx(TRUE_CPM, abs=DEGRADED_TOLERANCE_CPM)


def test_autocorrelation_raises_on_degenerate_signal(e_config):
    sig = np.zeros(300)
    with pytest.raises(EstimatorError):
        estimate_autocorrelation(sig, FS, e_config)


# ---------------------------------------------------------------------------
# FFT
# ---------------------------------------------------------------------------


def test_fft_recovers_clean_sine_cpm(e_config):
    _t, sig = _synthetic_sine()
    est = estimate_fft(sig, FS, e_config)
    assert est.cpm == pytest.approx(TRUE_CPM, abs=TOLERANCE_CPM)
    assert 0.0 <= est.confidence <= 1.0


def test_fft_recovers_noisy_sine_cpm(e_config):
    rng = np.random.default_rng(0)
    _t, sig = _synthetic_sine()
    sig = _noisy(sig, rng)
    est = estimate_fft(sig, FS, e_config)
    assert est.cpm == pytest.approx(TRUE_CPM, abs=DEGRADED_TOLERANCE_CPM)


def test_fft_recovers_dropped_cycles_cpm(e_config):
    t, sig = _synthetic_sine()
    sig = _drop_cycles(t, sig)
    est = estimate_fft(sig, FS, e_config)
    assert est.cpm == pytest.approx(TRUE_CPM, abs=DEGRADED_TOLERANCE_CPM)


def test_fft_raises_when_band_out_of_range(e_config):
    # FS=30Hz -> Nyquist=15Hz; a band entirely above that has no FFT bins at all
    _t, sig = _synthetic_sine()
    with pytest.raises(EstimatorError):
        estimate_fft(sig, FS, e_config.model_copy(update={"fft_freq_range_hz": (20.0, 25.0)}))


# ---------------------------------------------------------------------------
# Peak detection
# ---------------------------------------------------------------------------


def test_peaks_recovers_clean_sine_cpm(e_config):
    t, sig = _synthetic_sine()
    est = estimate_peaks(sig, t, FS, e_config)
    assert est.cpm == pytest.approx(TRUE_CPM, abs=TOLERANCE_CPM)
    assert 0.0 <= est.confidence <= 1.0
    # not assuming any fixed compression count -- just checking peaks were
    # found in a sane ballpark for this ~30s / 1.667Hz synthetic signal
    assert est.num_peaks > 10


def test_peaks_recovers_noisy_sine_cpm(e_config):
    rng = np.random.default_rng(0)
    t, sig = _synthetic_sine()
    sig = _noisy(sig, rng)
    est = estimate_peaks(sig, t, FS, e_config)
    assert est.cpm == pytest.approx(TRUE_CPM, abs=DEGRADED_TOLERANCE_CPM)


def test_peaks_recovers_dropped_cycles_cpm(e_config):
    t, sig = _synthetic_sine()
    sig = _drop_cycles(t, sig)
    est = estimate_peaks(sig, t, FS, e_config)
    assert est.cpm == pytest.approx(TRUE_CPM, abs=DEGRADED_TOLERANCE_CPM)


def test_peaks_raises_when_fewer_than_two_peaks(e_config):
    t = np.arange(0, 5, 1.0 / FS)
    flat = np.zeros_like(t)
    with pytest.raises(EstimatorError):
        estimate_peaks(flat, t, FS, e_config)


# ---------------------------------------------------------------------------
# run_estimators_on_video orchestration
# ---------------------------------------------------------------------------


def test_run_estimators_on_video_returns_all_four():
    t, sig = _synthetic_sine()
    filter_result = FilterResult(
        video_id="v",
        sample_rate_hz=FS,
        uniform_timestamps_sec=t,
        unfiltered_signal=sig,
        filtered_signal=sig,
    )
    config = HybridConfig()

    result = run_estimators_on_video(filter_result, config, "v")

    assert isinstance(result, AllEstimatesResult)
    for est in (result.cwt, result.autocorrelation, result.fft, result.peaks):
        assert est.cpm == pytest.approx(TRUE_CPM, abs=TOLERANCE_CPM)
