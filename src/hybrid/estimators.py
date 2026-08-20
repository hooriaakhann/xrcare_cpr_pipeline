"""Four Classical Rate Estimators (Phase 9).

Each estimator computes CPM independently from the Butterworth-filtered,
uniformly-resampled motion waveform (Phase 8), using a different signal-
processing technique -- CWT, autocorrelation, FFT, and peak detection. None
of them assumes or nudges toward the known 30-compression count (CLAUDE.md
rules 11/13); each reports whatever periodicity it actually finds in its
own search band.

Per spec, every estimator has a synthetic-signal unit test (a known-
frequency sine, a noisy version, and a version with dropped cycles),
verifying the recovered CPM is within tolerance -- see
tests/test_estimators.py -- before any of them is trusted on real video.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pywt
from scipy.signal import find_peaks

from hybrid.config import EstimatorsConfig, HybridConfig
from hybrid.exceptions import EstimatorError
from hybrid.filters import FilterResult
from hybrid.logging_config import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class CwtEstimate:
    dominant_freq_hz: float
    cpm: float
    confidence: float  # dominant scale's power / total in-band power


@dataclass(frozen=True)
class AutocorrelationEstimate:
    dominant_lag_sec: float
    cpm: float
    confidence: float  # normalized autocorrelation value at the dominant lag


@dataclass(frozen=True)
class FftEstimate:
    dominant_freq_hz: float
    cpm: float
    confidence: float  # squashed spectral SNR (peak vs. rest-of-band mean power)


@dataclass(frozen=True)
class PeakEstimate:
    peak_timestamps_sec: np.ndarray
    num_peaks: int
    median_inter_peak_interval_sec: float
    cpm: float
    confidence: float  # 1 - coefficient of variation of inter-peak intervals


@dataclass
class AllEstimatesResult:
    video_id: str
    cwt: CwtEstimate
    autocorrelation: AutocorrelationEstimate
    fft: FftEstimate
    peaks: PeakEstimate


def _scales_for_target_frequencies(wavelet: str, sampling_period: float, target_freqs_hz: np.ndarray) -> np.ndarray:
    """Scales whose CWT center frequencies land on `target_freqs_hz`. A plain
    linear scale grid (1, 2, 3, ...) gives very coarse, nonlinear frequency
    resolution right where the CPR band lives (freq(scale) ~ 1/scale), which
    isn't precise enough to recover a known-frequency test signal within a
    couple of CPM -- targeting frequencies directly instead gives a uniform,
    fit-for-purpose grid across the actual search band.
    """
    freq_per_unit_scale = pywt.scale2frequency(wavelet, 1.0) / sampling_period
    return freq_per_unit_scale / target_freqs_hz


def estimate_cwt(signal: np.ndarray, sample_rate_hz: float, config: EstimatorsConfig) -> CwtEstimate:
    low_hz, high_hz = config.cwt_freq_range_hz
    nyquist_hz = sample_rate_hz / 2.0
    if high_hz >= nyquist_hz:
        raise EstimatorError(
            f"CWT: search band high edge ({high_hz}Hz) is at/beyond Nyquist ({nyquist_hz}Hz) "
            f"for sample_rate={sample_rate_hz}Hz"
        )

    # search a bit wider than the nominal band so the dominant-frequency
    # search isn't clipped right at its own edge, then still only report
    # frequencies within [low_hz, high_hz] below
    margin_low_hz, margin_high_hz = low_hz * 0.7, min(high_hz * 1.3, nyquist_hz * 0.99)
    target_freqs = np.linspace(margin_low_hz, margin_high_hz, config.cwt_num_scales)
    scales = _scales_for_target_frequencies(config.cwt_wavelet, 1.0 / sample_rate_hz, target_freqs)
    coeffs, freqs = pywt.cwt(signal, scales, config.cwt_wavelet, sampling_period=1.0 / sample_rate_hz)

    # target_freqs was built to span [low_hz, high_hz] (with margin) whenever
    # high_hz < Nyquist, which was just checked above, so this always has at
    # least one match.
    in_band = (freqs >= low_hz) & (freqs <= high_hz)
    power = np.mean(np.abs(coeffs) ** 2, axis=1)
    band_power = power[in_band]
    band_freqs = freqs[in_band]

    dominant_idx = int(np.argmax(band_power))
    dominant_freq = float(band_freqs[dominant_idx])
    total = float(band_power.sum())
    confidence = float(band_power[dominant_idx] / total) if total > 0 else 0.0

    return CwtEstimate(
        dominant_freq_hz=dominant_freq, cpm=dominant_freq * 60.0, confidence=float(np.clip(confidence, 0.0, 1.0))
    )


def estimate_autocorrelation(
    signal: np.ndarray, sample_rate_hz: float, config: EstimatorsConfig
) -> AutocorrelationEstimate:
    centered = signal - np.mean(signal)
    n = len(centered)
    autocorr_full = np.correlate(centered, centered, mode="full")
    autocorr = autocorr_full[n - 1 :]  # lag 0 onward

    if autocorr[0] <= 0:
        raise EstimatorError("Autocorrelation: zero-lag autocorrelation is non-positive (degenerate signal)")
    autocorr_norm = autocorr / autocorr[0]

    low_sec, high_sec = config.autocorr_lag_range_sec
    low_lag = max(1, int(round(low_sec * sample_rate_hz)))
    high_lag = min(len(autocorr_norm) - 1, int(round(high_sec * sample_rate_hz)))
    if low_lag >= high_lag:
        raise EstimatorError(
            f"Autocorrelation: lag search range [{low_sec}, {high_sec}]s exceeds the signal's length "
            f"at sample_rate={sample_rate_hz}Hz"
        )

    window = autocorr_norm[low_lag : high_lag + 1]
    dominant_idx = int(np.argmax(window))
    dominant_lag_sec = (low_lag + dominant_idx) / sample_rate_hz
    confidence = float(np.clip(window[dominant_idx], 0.0, 1.0))

    return AutocorrelationEstimate(
        dominant_lag_sec=dominant_lag_sec, cpm=60.0 / dominant_lag_sec, confidence=confidence
    )


def estimate_fft(signal: np.ndarray, sample_rate_hz: float, config: EstimatorsConfig) -> FftEstimate:
    n = len(signal)
    windowed = signal * np.hanning(n)  # reduce spectral leakage
    spectrum = np.fft.rfft(windowed)
    power = np.abs(spectrum) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate_hz)

    low_hz, high_hz = config.fft_freq_range_hz
    in_band = (freqs >= low_hz) & (freqs <= high_hz)
    if not in_band.any():
        raise EstimatorError(f"FFT: no bin falls within [{low_hz}, {high_hz}] Hz at sample_rate={sample_rate_hz}Hz")

    band_power = power[in_band]
    band_freqs = freqs[in_band]
    dominant_idx = int(np.argmax(band_power))
    dominant_freq = float(band_freqs[dominant_idx])

    other_power = np.delete(band_power, dominant_idx)
    mean_other = float(np.mean(other_power)) if other_power.size > 0 else 0.0
    snr = float(band_power[dominant_idx]) / (mean_other + 1e-12)
    confidence = float(np.clip(snr / (snr + 1.0), 0.0, 1.0))  # squash unbounded SNR into [0, 1)

    return FftEstimate(dominant_freq_hz=dominant_freq, cpm=dominant_freq * 60.0, confidence=confidence)


def estimate_peaks(
    signal: np.ndarray, timestamps_sec: np.ndarray, sample_rate_hz: float, config: EstimatorsConfig
) -> PeakEstimate:
    min_distance_samples = max(1, int(round(config.peak_min_distance_sec * sample_rate_hz)))
    peak_indices, _properties = find_peaks(signal, distance=min_distance_samples, prominence=config.peak_prominence)

    if len(peak_indices) < 2:
        raise EstimatorError(
            f"Peaks: only {len(peak_indices)} peak(s) found -- need >= 2 to compute an inter-peak interval"
        )

    peak_times = timestamps_sec[peak_indices]
    intervals = np.diff(peak_times)
    median_interval = float(np.median(intervals))
    if median_interval <= 0:
        raise EstimatorError("Peaks: non-positive median inter-peak interval (degenerate/duplicate peaks)")

    mean_interval = float(np.mean(intervals))
    cv = float(np.std(intervals)) / (mean_interval + 1e-12)
    confidence = float(np.clip(1.0 - cv, 0.0, 1.0))

    return PeakEstimate(
        peak_timestamps_sec=peak_times,
        num_peaks=len(peak_indices),
        median_inter_peak_interval_sec=median_interval,
        cpm=60.0 / median_interval,
        confidence=confidence,
    )


def run_estimators_on_video(filter_result: FilterResult, config: HybridConfig, video_id: str) -> AllEstimatesResult:
    e_config = config.estimators
    signal = filter_result.filtered_signal
    sample_rate_hz = filter_result.sample_rate_hz
    timestamps = filter_result.uniform_timestamps_sec

    cwt = estimate_cwt(signal, sample_rate_hz, e_config)
    autocorrelation = estimate_autocorrelation(signal, sample_rate_hz, e_config)
    fft = estimate_fft(signal, sample_rate_hz, e_config)
    peaks = estimate_peaks(signal, timestamps, sample_rate_hz, e_config)

    logger.info(
        "%s: estimators -> CWT=%.1fCPM(conf=%.2f) Autocorr=%.1fCPM(conf=%.2f) FFT=%.1fCPM(conf=%.2f) "
        "Peaks=%.1fCPM(conf=%.2f, n=%d)",
        video_id,
        cwt.cpm,
        cwt.confidence,
        autocorrelation.cpm,
        autocorrelation.confidence,
        fft.cpm,
        fft.confidence,
        peaks.cpm,
        peaks.confidence,
        peaks.num_peaks,
    )

    return AllEstimatesResult(video_id=video_id, cwt=cwt, autocorrelation=autocorrelation, fft=fft, peaks=peaks)
