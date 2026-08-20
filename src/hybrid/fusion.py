"""Final Hybrid Fusion (Phase 12).

Combines the four classical estimators (Phase 9, all derived from the
tracker + optical-flow fused motion signal) and RepNet (Phase 10, an
independent learned-periodicity estimate) into one `final_cpm`. Not a plain
confidence-weighted average alone -- spec explicitly asks for "basic
disagreement handling" beyond that, so each candidate is also down-weighted
the further it sits from the group's confidence-weighted-median "center":

    final_weight_i = confidence_i * agreement_weight_i
    agreement_weight_i = 1 / (1 + |cpm_i - center| / disagreement_scale_cpm)

This means a low-confidence outlier (e.g. RepNet reporting ~65 CPM while
the other four cluster around 100) gets discounted twice over -- once for
its own low confidence, once for disagreeing with the group -- rather than
relying on its raw confidence alone to keep it from pulling the average
down, per the spec's own worked example.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hybrid.config import FusionConfig, HybridConfig
from hybrid.estimators import AllEstimatesResult
from hybrid.exceptions import FusionError
from hybrid.logging_config import get_logger
from hybrid.repnet_branch import RepNetResult

logger = get_logger(__name__)


@dataclass(frozen=True)
class CandidateEstimate:
    name: str
    cpm: float | None
    confidence: float
    agreement_weight: float
    final_weight: float


@dataclass
class FusionResult:
    video_id: str
    candidates: list[CandidateEstimate]
    center_cpm: float  # confidence-weighted median used for the disagreement calc
    final_cpm: float
    overall_confidence: float
    spread_cpm: float  # max - min across usable candidates
    std_cpm: float


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cum_weights = np.cumsum(weights)
    cutoff = cum_weights[-1] / 2.0
    idx = int(np.searchsorted(cum_weights, cutoff))
    idx = min(idx, len(values) - 1)
    return float(values[idx])


def fuse_estimates(
    estimates: AllEstimatesResult, repnet_result: RepNetResult, config: HybridConfig, video_id: str
) -> FusionResult:
    f_config: FusionConfig = config.fusion

    named = [
        ("cwt", estimates.cwt.cpm, estimates.cwt.confidence),
        ("autocorrelation", estimates.autocorrelation.cpm, estimates.autocorrelation.confidence),
        ("fft", estimates.fft.cpm, estimates.fft.confidence),
        ("peaks", estimates.peaks.cpm, estimates.peaks.confidence),
        ("repnet", repnet_result.cpm, repnet_result.confidence),
    ]

    usable = [(name, cpm, conf) for name, cpm, conf in named if cpm is not None]
    if not usable:
        raise FusionError(f"{video_id}: no usable candidate estimate to fuse (all cpm values are None)")

    cpm_arr = np.array([c for _, c, _ in usable])
    conf_arr = np.array([c for _, _, c in usable])
    median_weights = conf_arr if conf_arr.sum() > 0 else np.ones_like(conf_arr)
    center = _weighted_median(cpm_arr, median_weights)

    candidates: list[CandidateEstimate] = []
    for name, cpm, conf in named:
        if cpm is None:
            candidates.append(
                CandidateEstimate(name=name, cpm=None, confidence=conf, agreement_weight=0.0, final_weight=0.0)
            )
            continue
        deviation = abs(cpm - center)
        agreement_weight = 1.0 / (1.0 + deviation / f_config.disagreement_scale_cpm)
        final_weight = conf * agreement_weight
        candidates.append(
            CandidateEstimate(
                name=name, cpm=cpm, confidence=conf, agreement_weight=agreement_weight, final_weight=final_weight
            )
        )

    total_weight = sum(c.final_weight for c in candidates)
    if total_weight <= 0:
        # Every usable candidate had zero confidence -- fall back to an
        # unweighted mean of usable candidates rather than raising, since
        # there is data, just no trusted way to weight it.
        final_cpm = float(np.mean(cpm_arr))
        overall_confidence = 0.0
        logger.warning(
            "%s: all fusion weights were zero -- falling back to unweighted mean of usable candidates", video_id
        )
    else:
        final_cpm = sum(c.final_weight * c.cpm for c in candidates if c.cpm is not None) / total_weight
        overall_confidence = total_weight / len(named)

    spread_cpm = float(cpm_arr.max() - cpm_arr.min())
    std_cpm = float(np.std(cpm_arr))

    logger.info(
        "%s: fusion -> final_cpm=%.2f overall_confidence=%.2f spread=%.2f (center=%.2f)",
        video_id,
        final_cpm,
        overall_confidence,
        spread_cpm,
        center,
    )

    return FusionResult(
        video_id=video_id,
        candidates=candidates,
        center_cpm=center,
        final_cpm=final_cpm,
        overall_confidence=float(np.clip(overall_confidence, 0.0, 1.0)),
        spread_cpm=spread_cpm,
        std_cpm=std_cpm,
    )
