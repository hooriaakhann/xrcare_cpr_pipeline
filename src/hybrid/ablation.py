"""Ablation Comparison (Phase 14).

Determines whether the hybrid fusion (Phase 12) actually helps, by scoring
each of eight sub-pipelines -- each derived from a subset of the branches
built in Phases 2-10 -- against development MAE/RMSE, and comparing the
full hybrid to the single best-performing one with a paired Wilcoxon test.

    A - MediaPipe wrist signal only
    B - CoTracker trajectory only (raw, no ego-motion correction)
    C - CoTracker + affine compensation (Phase 5's corrected trajectory)
    D - Farneback optical flow, foreground reading only (no background
        subtraction -- i.e. NOT camera-motion-compensated)
    E - Affine-compensated Farneback (Phase 6's foreground-minus-background
        residual, `flow_motion_y` -- camera motion already netted out via
        the background reading, unlike D)
    F - Tracker + flow fused motion (Phase 7's `motion_wave`, i.e. the full
        pipeline minus RepNet)
    G - RepNet alone (Phase 10's own CPM, no filtering/estimation)
    H - Full hybrid (Phase 12, unchanged)

A-F all reuse the *already-cached* MediaPipe/CoTracker/ego-motion/optical-
flow branch outputs -- nothing expensive is re-run, only the cheap Phase
7-9 steps (fusion, filtering, estimation) on each alternative signal. For
A-F, the four classical estimators are combined with Phase 12's own
`fuse_estimates()` (RepNet excluded via a null candidate) rather than an
ad hoc average, so each ablation's single CPM number comes from the same
fusion methodology the full hybrid uses -- an apples-to-apples comparison,
not a different aggregation rule for the ablations vs. the real pipeline.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy.stats import wilcoxon

from hybrid.caching import CacheManager
from hybrid.config import HybridConfig
from hybrid.corrected_trajectory import correct_tracker_trajectory
from hybrid.cotracker_tracker import run_cotracker_on_video_cached
from hybrid.dataset import DevVideo, discover_development_videos
from hybrid.ego_motion import run_ego_motion_on_video_cached
from hybrid.estimators import run_estimators_on_video
from hybrid.evaluation import VideoEvaluationResult, _bootstrap_mae_ci
from hybrid.exceptions import EstimatorError, FilteringError, FusionError
from hybrid.filters import apply_butterworth_filter
from hybrid.fusion import fuse_estimates
from hybrid.logging_config import get_logger
from hybrid.mediapipe_roi import MediaPipeVideoResult, run_mediapipe_on_video_cached
from hybrid.motion_wave import generate_motion_wave
from hybrid.optical_flow import run_optical_flow_on_video_cached
from hybrid.repnet_branch import RepNetResult, run_repnet_on_video_cached

logger = get_logger(__name__)


@dataclass(frozen=True)
class AblationVideoResult:
    ablation: str
    video_id: str
    cpm: float | None
    absolute_error: float | None
    runtime_sec: float


@dataclass(frozen=True)
class AblationSummary:
    ablation: str
    mae: float
    rmse: float
    mean_runtime_sec: float
    mae_bootstrap_ci_95: tuple[float, float]
    num_videos: int
    notes: str


def _mediapipe_wrist_signal(mp_result: MediaPipeVideoResult) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame wrist Y position, NaN where undetected -- ablation A's raw
    signal. `apply_butterworth_filter`'s resampling already skips NaNs.
    """
    timestamps = np.array([d.timestamp_sec for d in mp_result.detections])
    wrist_y = np.array(
        [d.wrist_xy[1] if d.detected and d.wrist_xy is not None else np.nan for d in mp_result.detections]
    )
    return timestamps, wrist_y


def _null_repnet_result(video_id: str) -> RepNetResult:
    """Excludes RepNet from fuse_estimates() for single-signal ablations
    (A-F) without duplicating its confidence-weighted/disagreement logic.
    """
    return RepNetResult(
        video_id=video_id,
        cpm=None,
        confidence=0.0,
        pred_period_frames=0.0,
        chosen_stride=0,
        fps=0.0,
        num_frames=0,
        reason="excluded from single-signal ablation",
    )


def _cpm_from_signal(timestamps: np.ndarray, signal: np.ndarray, config: HybridConfig, video_id: str) -> float:
    filt = apply_butterworth_filter(timestamps, signal, config, video_id)
    estimates = run_estimators_on_video(filt, config, video_id)
    fusion_result = fuse_estimates(estimates, _null_repnet_result(video_id), config, video_id)
    return fusion_result.final_cpm


def run_ablations_on_video(
    dev_video: DevVideo,
    config: HybridConfig,
    cache_manager: CacheManager,
    full_hybrid_result: VideoEvaluationResult,
) -> list[AblationVideoResult]:
    video_id = dev_video.video_id
    gt_cpm = dev_video.gt.gt_cpm
    results: list[AblationVideoResult] = []

    mp_result = run_mediapipe_on_video_cached(dev_video.split_path, config, video_id, cache_manager)
    ct_result = run_cotracker_on_video_cached(dev_video.split_path, mp_result, config, video_id, cache_manager)
    ego_result = run_ego_motion_on_video_cached(dev_video.split_path, mp_result, config, video_id, cache_manager)
    corrected = correct_tracker_trajectory(ct_result, ego_result, video_id)
    flow_result = run_optical_flow_on_video_cached(dev_video.split_path, mp_result, config, video_id, cache_manager)
    wave = generate_motion_wave(corrected, flow_result, video_id)
    repnet_result = run_repnet_on_video_cached(dev_video.split_path, config, video_id, cache_manager)

    def _add(name: str, timestamps: np.ndarray, signal: np.ndarray) -> None:
        t0 = time.time()
        try:
            cpm = _cpm_from_signal(timestamps, signal, config, video_id)
            absolute_error = abs(cpm - gt_cpm)
        except (FilteringError, EstimatorError, FusionError) as e:
            logger.warning("%s: ablation %s could not produce a CPM: %s", video_id, name, e)
            cpm, absolute_error = None, None
        results.append(
            AblationVideoResult(
                ablation=name, video_id=video_id, cpm=cpm, absolute_error=absolute_error, runtime_sec=time.time() - t0
            )
        )

    mp_t, mp_signal = _mediapipe_wrist_signal(mp_result)
    _add("A_mediapipe_wrist", mp_t, mp_signal)
    _add("B_cotracker_raw", ct_result.timestamps_sec, ct_result.tracker_motion_y)
    _add("C_cotracker_affine", corrected.timestamps_sec, corrected.corrected_tracker_motion_y)

    flow_t = np.array([f.timestamp_sec for f in flow_result.frames])
    flow_fg = np.array([f.foreground_flow_y for f in flow_result.frames])
    _add("D_flow_raw", flow_t, flow_fg)
    _add("E_flow_affine_compensated", flow_t, flow_result.flow_motion_y)
    _add("F_tracker_flow_fused", wave.timestamps_sec, wave.motion_wave)

    t0 = time.time()
    g_cpm = repnet_result.cpm
    g_error = abs(g_cpm - gt_cpm) if g_cpm is not None else None
    results.append(
        AblationVideoResult(
            ablation="G_repnet_alone",
            video_id=video_id,
            cpm=g_cpm,
            absolute_error=g_error,
            runtime_sec=time.time() - t0,
        )
    )

    results.append(
        AblationVideoResult(
            ablation="H_full_hybrid",
            video_id=video_id,
            cpm=full_hybrid_result.final_cpm,
            absolute_error=full_hybrid_result.absolute_error,
            runtime_sec=full_hybrid_result.runtime_sec,
        )
    )

    return results


def summarize_ablations(
    all_video_results: list[list[AblationVideoResult]], seed: int, num_bootstrap_resamples: int = 2000
) -> list[AblationSummary]:
    by_ablation: dict[str, list[AblationVideoResult]] = {}
    for video_results in all_video_results:
        for r in video_results:
            by_ablation.setdefault(r.ablation, []).append(r)

    summaries = []
    for name, results in by_ablation.items():
        usable = [r for r in results if r.cpm is not None]
        n_missing = len(results) - len(usable)
        if not usable:
            logger.warning("Ablation %s: no video produced a usable CPM -- skipped", name)
            continue

        abs_errors = np.array([r.absolute_error for r in usable])
        mae = float(np.mean(abs_errors))
        rmse = float(np.sqrt(np.mean(abs_errors**2)))  # abs_error^2 == signed_error^2
        mean_runtime = float(np.mean([r.runtime_sec for r in usable]))
        ci = _bootstrap_mae_ci(abs_errors, num_bootstrap_resamples, seed)
        notes = f"{n_missing} video(s) excluded (no CPM produced)" if n_missing else ""

        summaries.append(
            AblationSummary(
                ablation=name,
                mae=mae,
                rmse=rmse,
                mean_runtime_sec=mean_runtime,
                mae_bootstrap_ci_95=ci,
                num_videos=len(usable),
                notes=notes,
            )
        )

    return sorted(summaries, key=lambda s: s.mae)


def wilcoxon_hybrid_vs_best_ablation(
    all_video_results: list[list[AblationVideoResult]], summaries: list[AblationSummary]
) -> dict:
    """Paired Wilcoxon signed-rank test on per-video absolute error, full
    hybrid (H) vs. the best-performing single-branch ablation (A-G). Spec's
    own caution applies directly: a 6-video dev set gives this test very
    little power -- report the result honestly (likely not significant),
    not something to force into "significance."
    """
    single_branch = [s for s in summaries if s.ablation != "H_full_hybrid"]
    if not single_branch:
        return {
            "best_ablation": None,
            "statistic": None,
            "p_value": None,
            "note": "no single-branch ablation available",
        }
    best = min(single_branch, key=lambda s: s.mae)

    def _errors_by_video(ablation_name: str) -> dict[str, float]:
        return {
            r.video_id: r.absolute_error
            for video_results in all_video_results
            for r in video_results
            if r.ablation == ablation_name and r.absolute_error is not None
        }

    hybrid_errors = _errors_by_video("H_full_hybrid")
    best_errors = _errors_by_video(best.ablation)
    common_ids = sorted(set(hybrid_errors) & set(best_errors))

    if len(common_ids) < 2:
        return {
            "best_ablation": best.ablation,
            "statistic": None,
            "p_value": None,
            "note": f"fewer than 2 paired videos ({len(common_ids)}) -- cannot run Wilcoxon",
        }

    x = [hybrid_errors[v] for v in common_ids]
    y = [best_errors[v] for v in common_ids]
    if all(a == b for a, b in zip(x, y)):
        return {
            "best_ablation": best.ablation,
            "statistic": None,
            "p_value": None,
            "note": "all paired differences are zero -- Wilcoxon undefined",
        }

    statistic, p_value = wilcoxon(x, y)
    return {
        "best_ablation": best.ablation,
        "statistic": float(statistic),
        "p_value": float(p_value),
        "note": f"n={len(common_ids)} paired videos",
    }


def run_ablation_comparison(
    config: HybridConfig,
    cache_manager: CacheManager,
    full_hybrid_results: list[VideoEvaluationResult],
) -> tuple[list[AblationSummary], dict]:
    dev_videos = {v.video_id: v for v in discover_development_videos(config)}
    all_video_results = [
        run_ablations_on_video(dev_videos[r.video_id], config, cache_manager, r) for r in full_hybrid_results
    ]
    summaries = summarize_ablations(all_video_results, config.project.seed)
    wilcoxon_result = wilcoxon_hybrid_vs_best_ablation(all_video_results, summaries)
    return summaries, wilcoxon_result
