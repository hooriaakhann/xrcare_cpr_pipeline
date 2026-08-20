"""Butterworth Band-Pass Filtering (Phase 8).

The fused `motion_wave(t)` (Phase 7) is irregularly sampled -- the source
videos are VFR (CLAUDE.md rule 6) -- but a Butterworth filter's frequency
response is only well-defined for a uniformly-sampled signal. So the signal
is first resampled onto a uniform time grid (linear interpolation, at the
video's median inter-frame interval as the nominal sample rate) purely for
the filtering step. This module never mutates Phase 1-7's outputs or
pretends the original frames were uniformly spaced -- it returns its own
`uniform_timestamps_sec` alongside the (un)filtered signal, distinct from
the original per-frame timestamps used everywhere else.

Filtered offline with zero-phase `scipy.signal.filtfilt` (no time lag/shift)
since the whole video is available at once -- appropriate here, unlike an
online/streaming causal filter which would need `lfilter` and accept phase
lag instead.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, filtfilt

from hybrid.config import FilteringConfig, HybridConfig
from hybrid.exceptions import FilteringError
from hybrid.logging_config import get_logger
from hybrid.motion_wave import MotionWaveResult

logger = get_logger(__name__)


@dataclass
class FilterResult:
    video_id: str
    sample_rate_hz: float
    uniform_timestamps_sec: np.ndarray
    unfiltered_signal: np.ndarray  # resampled motion_wave, before filtering
    filtered_signal: np.ndarray  # zero-phase Butterworth band-pass output


def _nominal_sample_rate(timestamps: np.ndarray) -> float:
    """Median inter-frame interval -> Hz. Median (not mean) so an occasional
    dropped/duplicated frame doesn't skew the estimate.
    """
    diffs = np.diff(timestamps)
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        raise FilteringError("Cannot compute a sample rate: fewer than 2 distinct increasing timestamps")
    return 1.0 / float(np.median(diffs))


def _resample_to_uniform_grid(
    timestamps: np.ndarray, values: np.ndarray, sample_rate_hz: float
) -> tuple[np.ndarray, np.ndarray]:
    """Linear interpolation onto a uniform grid at `sample_rate_hz`, skipping
    NaN samples in the source signal. Raises FilteringError if fewer than 2
    non-NaN samples remain.
    """
    valid = ~np.isnan(values)
    if valid.sum() < 2:
        raise FilteringError(f"Not enough non-NaN samples to resample/filter ({valid.sum()} available)")

    t_valid, v_valid = timestamps[valid], values[valid]
    order = np.argsort(t_valid)
    t_valid, v_valid = t_valid[order], v_valid[order]

    n_samples = int(np.floor((t_valid[-1] - t_valid[0]) * sample_rate_hz)) + 1
    uniform_t = t_valid[0] + np.arange(n_samples) / sample_rate_hz
    uniform_t = uniform_t[uniform_t <= t_valid[-1]]
    uniform_v = np.interp(uniform_t, t_valid, v_valid)
    return uniform_t, uniform_v


def _design_butterworth(
    low_hz: float, high_hz: float, order: int, sample_rate_hz: float
) -> tuple[np.ndarray, np.ndarray]:
    nyquist_hz = sample_rate_hz / 2.0
    if not (0.0 < low_hz < high_hz < nyquist_hz):
        raise FilteringError(
            f"Invalid band [{low_hz}, {high_hz}] Hz for sample_rate={sample_rate_hz:.2f}Hz "
            f"(Nyquist={nyquist_hz:.2f}Hz) -- must satisfy 0 < low < high < Nyquist"
        )
    b, a = butter(order, [low_hz, high_hz], btype="band", fs=sample_rate_hz)
    return b, a


def apply_butterworth_filter(motion_wave_result: MotionWaveResult, config: HybridConfig, video_id: str) -> FilterResult:
    f_config: FilteringConfig = config.filtering
    timestamps = motion_wave_result.timestamps_sec
    values = motion_wave_result.motion_wave

    sample_rate_hz = _nominal_sample_rate(timestamps)
    uniform_t, uniform_v = _resample_to_uniform_grid(timestamps, values, sample_rate_hz)

    b, a = _design_butterworth(
        f_config.butterworth_low_hz, f_config.butterworth_high_hz, f_config.butterworth_order, sample_rate_hz
    )

    # filtfilt's default odd-symmetric padding needs the signal longer than
    # 3 * max(len(a), len(b)); a short/aggressively-trimmed video could
    # violate this, so fail loudly rather than let scipy raise an opaque error.
    min_len = 3 * max(len(a), len(b))
    if len(uniform_v) <= min_len:
        raise FilteringError(
            f"{video_id}: resampled signal too short ({len(uniform_v)} samples) for filtfilt with "
            f"butterworth_order={f_config.butterworth_order} (needs > {min_len} samples)"
        )

    filtered_v = filtfilt(b, a, uniform_v)

    logger.info(
        "%s: Butterworth band-pass [%.2f, %.2f] Hz (order=%d) applied over %d resampled frames "
        "(sample_rate=%.2fHz, from %d original frames)",
        video_id,
        f_config.butterworth_low_hz,
        f_config.butterworth_high_hz,
        f_config.butterworth_order,
        len(uniform_v),
        sample_rate_hz,
        len(timestamps),
    )

    return FilterResult(
        video_id=video_id,
        sample_rate_hz=sample_rate_hz,
        uniform_timestamps_sec=uniform_t,
        unfiltered_signal=uniform_v,
        filtered_signal=filtered_v,
    )
