"""Development-Set Evaluation (Phase 13).

Runs the complete Phase 1-12 pipeline on every development video, compares
each prediction to ground truth (post-prediction only -- GT is never an
input to inference, CLAUDE.md rule 5), and reports development MAE/RMSE
plus a bootstrap 95% CI on MAE (a 6-video dev set is too small for a point
estimate alone to be informative). These are dev-set numbers for tuning and
debugging, not the final generalization result -- the held-out test set
stays untouched until the pipeline is frozen (CLAUDE.md rules 1-2).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from hybrid.caching import CacheManager
from hybrid.config import HybridConfig
from hybrid.corrected_trajectory import correct_tracker_trajectory
from hybrid.cotracker_tracker import run_cotracker_on_video_cached
from hybrid.dataset import DevVideo, discover_development_videos
from hybrid.ego_motion import run_ego_motion_on_video_cached
from hybrid.estimators import run_estimators_on_video
from hybrid.experiment_ledger import log_run
from hybrid.filters import apply_butterworth_filter
from hybrid.fusion import fuse_estimates
from hybrid.logging_config import get_logger
from hybrid.mediapipe_roi import run_mediapipe_on_video_cached
from hybrid.motion_wave import generate_motion_wave
from hybrid.optical_flow import run_optical_flow_on_video_cached
from hybrid.repnet_branch import run_repnet_on_video_cached

logger = get_logger(__name__)


@dataclass
class VideoEvaluationResult:
    video_id: str
    gt_cpm: float
    cwt_cpm: float
    autocorrelation_cpm: float
    fft_cpm: float
    peaks_cpm: float
    repnet_cpm: float | None
    final_cpm: float
    overall_confidence: float
    signed_error: float
    absolute_error: float
    runtime_sec: float


@dataclass
class DevelopmentEvaluationResult:
    per_video: list[VideoEvaluationResult]
    mae: float
    rmse: float
    mean_signed_error: float
    median_absolute_error: float
    max_absolute_error: float
    mae_bootstrap_ci_95: tuple[float, float]


def run_full_pipeline_on_video(
    dev_video: DevVideo, config: HybridConfig, cache_manager: CacheManager
) -> VideoEvaluationResult:
    """Phase 2 through Phase 12, end to end, for one development video. Every
    branch call goes through its `_cached` wrapper, so a re-run after the
    first is cheap.
    """
    video_id = dev_video.video_id
    t0 = time.time()

    mp_result = run_mediapipe_on_video_cached(dev_video.split_path, config, video_id, cache_manager)
    ct_result = run_cotracker_on_video_cached(dev_video.split_path, mp_result, config, video_id, cache_manager)
    ego_result = run_ego_motion_on_video_cached(dev_video.split_path, mp_result, config, video_id, cache_manager)
    corrected = correct_tracker_trajectory(ct_result, ego_result, video_id)
    flow_result = run_optical_flow_on_video_cached(dev_video.split_path, mp_result, config, video_id, cache_manager)
    wave = generate_motion_wave(corrected, flow_result, video_id)
    filt = apply_butterworth_filter(wave.timestamps_sec, wave.motion_wave, config, video_id)
    estimates = run_estimators_on_video(filt, config, video_id)
    repnet_result = run_repnet_on_video_cached(dev_video.split_path, config, video_id, cache_manager)
    fusion_result = fuse_estimates(estimates, repnet_result, config, video_id)

    runtime_sec = time.time() - t0

    gt_cpm = dev_video.gt.gt_cpm
    signed_error = fusion_result.final_cpm - gt_cpm
    absolute_error = abs(signed_error)

    logger.info(
        "%s: final_cpm=%.2f gt_cpm=%.2f signed_error=%+.2f runtime=%.1fs",
        video_id,
        fusion_result.final_cpm,
        gt_cpm,
        signed_error,
        runtime_sec,
    )

    return VideoEvaluationResult(
        video_id=video_id,
        gt_cpm=gt_cpm,
        cwt_cpm=estimates.cwt.cpm,
        autocorrelation_cpm=estimates.autocorrelation.cpm,
        fft_cpm=estimates.fft.cpm,
        peaks_cpm=estimates.peaks.cpm,
        repnet_cpm=repnet_result.cpm,
        final_cpm=fusion_result.final_cpm,
        overall_confidence=fusion_result.overall_confidence,
        signed_error=signed_error,
        absolute_error=absolute_error,
        runtime_sec=runtime_sec,
    )


def _bootstrap_mae_ci(absolute_errors: np.ndarray, num_resamples: int, seed: int) -> tuple[float, float]:
    """Percentile bootstrap 95% CI on the mean absolute error: resample the
    per-video absolute errors with replacement `num_resamples` times, take
    each resample's mean, report the 2.5th/97.5th percentiles.
    """
    rng = np.random.default_rng(seed)
    n = len(absolute_errors)
    resampled_maes = np.empty(num_resamples)
    for i in range(num_resamples):
        sample = rng.choice(absolute_errors, size=n, replace=True)
        resampled_maes[i] = sample.mean()
    lower, upper = np.percentile(resampled_maes, [2.5, 97.5])
    return float(lower), float(upper)


def run_development_evaluation(
    config: HybridConfig, cache_manager: CacheManager, num_bootstrap_resamples: int = 2000
) -> DevelopmentEvaluationResult:
    dev_videos = discover_development_videos(config)
    per_video = [run_full_pipeline_on_video(v, config, cache_manager) for v in dev_videos]

    signed_errors = np.array([r.signed_error for r in per_video])
    absolute_errors = np.array([r.absolute_error for r in per_video])

    mae = float(np.mean(absolute_errors))
    rmse = float(np.sqrt(np.mean(signed_errors**2)))
    mean_signed_error = float(np.mean(signed_errors))
    median_absolute_error = float(np.median(absolute_errors))
    max_absolute_error = float(np.max(absolute_errors))
    ci = _bootstrap_mae_ci(absolute_errors, num_bootstrap_resamples, config.project.seed)

    logger.info(
        "Development evaluation: MAE=%.2f (95%% CI [%.2f, %.2f]) RMSE=%.2f mean_signed_error=%+.2f "
        "median_AE=%.2f max_AE=%.2f over %d videos",
        mae,
        ci[0],
        ci[1],
        rmse,
        mean_signed_error,
        median_absolute_error,
        max_absolute_error,
        len(per_video),
    )

    log_run(
        phase="development_evaluation",
        config=config,
        metrics={
            "dev_mae": mae,
            "dev_rmse": rmse,
            "mean_signed_error": mean_signed_error,
            "median_absolute_error": median_absolute_error,
            "max_absolute_error": max_absolute_error,
            "mae_bootstrap_ci_95_lower": ci[0],
            "mae_bootstrap_ci_95_upper": ci[1],
            "num_videos": len(per_video),
        },
        extra={"video_ids": [r.video_id for r in per_video]},
    )

    return DevelopmentEvaluationResult(
        per_video=per_video,
        mae=mae,
        rmse=rmse,
        mean_signed_error=mean_signed_error,
        median_absolute_error=median_absolute_error,
        max_absolute_error=max_absolute_error,
        mae_bootstrap_ci_95=ci,
    )
