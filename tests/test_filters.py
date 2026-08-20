import numpy as np
import pytest

from hybrid.config import HybridConfig
from hybrid.exceptions import FilteringError
from hybrid.filters import (
    _design_butterworth,
    _nominal_sample_rate,
    _resample_to_uniform_grid,
    apply_butterworth_filter,
)

# ---------------------------------------------------------------------------
# _nominal_sample_rate
# ---------------------------------------------------------------------------


def test_nominal_sample_rate_uniform_timestamps():
    ts = np.arange(100) * (1.0 / 30.0)
    assert _nominal_sample_rate(ts) == pytest.approx(30.0, rel=1e-6)


def test_nominal_sample_rate_robust_to_vfr_jitter():
    rng = np.random.default_rng(0)
    base = np.arange(200) * (1.0 / 30.0)
    jitter = rng.uniform(-0.002, 0.002, size=200)
    ts = np.sort(base + jitter)
    assert _nominal_sample_rate(ts) == pytest.approx(30.0, rel=0.05)


def test_nominal_sample_rate_too_few_timestamps_raises():
    with pytest.raises(FilteringError):
        _nominal_sample_rate(np.array([0.0]))


# ---------------------------------------------------------------------------
# _resample_to_uniform_grid
# ---------------------------------------------------------------------------


def test_resample_to_uniform_grid_recovers_linear_signal():
    rng = np.random.default_rng(1)
    t = np.sort(rng.uniform(0, 10, size=200))
    v = 3.0 * t + 1.0  # a line -- linear interpolation should recover it near-exactly

    uniform_t, uniform_v = _resample_to_uniform_grid(t, v, sample_rate_hz=30.0)

    expected = 3.0 * uniform_t + 1.0
    np.testing.assert_allclose(uniform_v, expected, atol=1e-6)


def test_resample_to_uniform_grid_skips_nan_samples():
    t = np.arange(20, dtype=float)
    v = t.copy()
    v[5:8] = np.nan
    uniform_t, uniform_v = _resample_to_uniform_grid(t, v, sample_rate_hz=1.0)
    assert not np.any(np.isnan(uniform_v))


def test_resample_to_uniform_grid_all_nan_raises():
    t = np.arange(10, dtype=float)
    v = np.full(10, np.nan)
    with pytest.raises(FilteringError):
        _resample_to_uniform_grid(t, v, sample_rate_hz=1.0)


# ---------------------------------------------------------------------------
# _design_butterworth
# ---------------------------------------------------------------------------


def test_design_butterworth_valid_band_returns_coefficients():
    b, a = _design_butterworth(1.0, 3.0, order=4, sample_rate_hz=30.0)
    assert len(b) > 0 and len(a) > 0


def test_design_butterworth_violates_nyquist_raises():
    with pytest.raises(FilteringError):
        _design_butterworth(1.0, 3.0, order=4, sample_rate_hz=4.0)  # Nyquist=2Hz < high_hz=3Hz


def test_design_butterworth_low_not_less_than_high_raises():
    with pytest.raises(FilteringError):
        _design_butterworth(3.0, 1.0, order=4, sample_rate_hz=30.0)


# ---------------------------------------------------------------------------
# apply_butterworth_filter: synthetic-signal correctness (in-band preserved,
# out-of-band attenuated, zero-phase)
# ---------------------------------------------------------------------------


def test_apply_butterworth_filter_preserves_in_band_attenuates_out_of_band():
    fs = 30.0
    duration_sec = 20.0
    t = np.arange(0, duration_sec, 1.0 / fs)

    in_band_freq = 1.5  # inside default [1.0, 3.0] Hz band
    in_band = 1.0 * np.sin(2 * np.pi * in_band_freq * t)
    drift = 5.0 * np.sin(2 * np.pi * 0.05 * t)  # well below the band
    noise = 0.5 * np.sin(2 * np.pi * 10.0 * t)  # well above the band
    composite = in_band + drift + noise

    config = HybridConfig()

    result = apply_butterworth_filter(t, composite, config, "v")

    # discard filtfilt edge-transient region for amplitude/correlation checks
    edge = int(fs * 2)
    filtered_interior = result.filtered_signal[edge:-edge]
    reference_interior = in_band[edge : len(in_band) - edge]

    # amplitude close to the in-band component's, not the much larger drift
    assert np.std(filtered_interior) == pytest.approx(np.std(reference_interior), rel=0.25)
    # strong correlation with the pure in-band reference signal
    corr = np.corrcoef(filtered_interior, reference_interior)[0, 1]
    assert corr > 0.9


def test_apply_butterworth_filter_is_zero_phase():
    fs = 30.0
    duration_sec = 10.0
    t = np.arange(0, duration_sec, 1.0 / fs)
    freq = 1.5
    signal = np.sin(2 * np.pi * freq * t)  # already in-band, minimal edge distortion of its own peaks

    config = HybridConfig()
    result = apply_butterworth_filter(t, signal, config, "v")

    # find a peak comfortably away from the filtfilt edge-transient region
    edge = int(fs * 2)
    interior = result.filtered_signal[edge:-edge]
    interior_t = result.uniform_timestamps_sec[edge:-edge]
    peak_idx = np.argmax(interior)
    peak_t = interior_t[peak_idx]

    # nearest true peak of the original sine (period = 1/freq)
    period = 1.0 / freq
    nearest_true_peak = round((peak_t - period / 4) / period) * period + period / 4

    assert abs(peak_t - nearest_true_peak) < (period * 0.1)  # well under 10% of one cycle -- no lag


def test_apply_butterworth_filter_violates_nyquist_raises():
    fs = 4.0  # Nyquist = 2Hz, below the default 3.0Hz upper band
    t = np.arange(0, 10.0, 1.0 / fs)
    signal = np.sin(2 * np.pi * 1.0 * t)
    config = HybridConfig()

    with pytest.raises(FilteringError):
        apply_butterworth_filter(t, signal, config, "v")


def test_apply_butterworth_filter_too_short_signal_raises():
    t = np.arange(0, 0.2, 1.0 / 30.0)  # only a handful of samples
    signal = np.sin(2 * np.pi * 1.5 * t)
    config = HybridConfig()

    with pytest.raises(FilteringError):
        apply_butterworth_filter(t, signal, config, "v")
