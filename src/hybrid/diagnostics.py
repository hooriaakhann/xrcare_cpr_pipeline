"""Save Diagnostics (Phase 16).

For every development video, writes a run folder
`runs/development/<video_id>/` with the intermediate/final artifacts spec
asks for: `summary.json`, `signals.csv`, `cotracker_tracks.csv`,
`ego_motion.csv`, `optical_flow_signal.csv`, `filtered_signal.csv`,
`estimator_results.json`, `repnet_result.json`, `peaks.csv`. Diagnostic
*plots* ("if practical" per spec) are not generated in this pass -- a
deliberate scope decision given this session's remaining time budget, not
an oversight; the CSV/JSON artifacts already contain everything a plot
would visualize, so building one later doesn't require re-running anything.

Reuses the same cached-branch-call pattern as Phase 14's ablations: nothing
expensive is re-run if MediaPipe/CoTracker/ego-motion/optical-flow/RepNet
are already cached for a video.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from hybrid.caching import CacheManager
from hybrid.config import HybridConfig
from hybrid.corrected_trajectory import correct_tracker_trajectory
from hybrid.cotracker_tracker import run_cotracker_on_video_cached
from hybrid.dataset import DevVideo
from hybrid.ego_motion import run_ego_motion_on_video_cached
from hybrid.estimators import run_estimators_on_video
from hybrid.experiment_ledger import get_git_commit_hash
from hybrid.filters import apply_butterworth_filter
from hybrid.fusion import fuse_estimates
from hybrid.logging_config import get_logger
from hybrid.mediapipe_roi import run_mediapipe_on_video_cached
from hybrid.motion_wave import generate_motion_wave
from hybrid.optical_flow import run_optical_flow_on_video_cached
from hybrid.repnet_branch import run_repnet_on_video_cached

logger = get_logger(__name__)


def _write_csv(path: Path, header: list[str], rows: list[list[Any]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    return value


def save_diagnostics_for_video(dev_video: DevVideo, config: HybridConfig, cache_manager: CacheManager) -> Path:
    video_id = dev_video.video_id
    run_dir = config.paths.runs_dir / "development" / video_id
    run_dir.mkdir(parents=True, exist_ok=True)

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

    ct_rows = [
        [
            int(ct_result.frame_indices[t]),
            float(ct_result.timestamps_sec[t]),
            n,
            float(ct_result.tracks_x[t, n]),
            float(ct_result.tracks_y[t, n]),
            bool(ct_result.visibility[t, n]),
        ]
        for t in range(len(ct_result.frame_indices))
        for n in range(ct_result.num_points)
    ]
    _write_csv(
        run_dir / "cotracker_tracks.csv",
        ["frame_index", "timestamp_sec", "point_index", "x", "y", "visible"],
        ct_rows,
    )

    ego_rows = [
        [
            f.frame_index,
            f.timestamp_sec,
            f.translation_x,
            f.translation_y,
            f.rotation_rad,
            f.scale,
            f.inlier_ratio,
            f.transform_valid,
            f.confidence,
        ]
        for f in ego_result.frames
    ]
    _write_csv(
        run_dir / "ego_motion.csv",
        [
            "frame_index",
            "timestamp_sec",
            "translation_x",
            "translation_y",
            "rotation_rad",
            "scale",
            "inlier_ratio",
            "transform_valid",
            "confidence",
        ],
        ego_rows,
    )

    flow_rows = [
        [
            f.frame_index,
            f.timestamp_sec,
            f.foreground_flow_y,
            f.background_flow_y,
            f.residual_flow_y,
            f.flow_magnitude,
            f.valid,
        ]
        for f in flow_result.frames
    ]
    _write_csv(
        run_dir / "optical_flow_signal.csv",
        [
            "frame_index",
            "timestamp_sec",
            "foreground_flow_y",
            "background_flow_y",
            "residual_flow_y",
            "flow_magnitude",
            "valid",
        ],
        flow_rows,
    )

    signal_rows = [
        [
            int(wave.frame_indices[i]),
            float(wave.timestamps_sec[i]),
            float(wave.raw_tracker_signal[i]),
            float(wave.raw_flow_signal[i]),
            float(wave.normalized_tracker_signal[i]),
            float(wave.normalized_flow_signal[i]),
            float(wave.tracker_weight[i]),
            float(wave.flow_weight[i]),
            float(wave.motion_wave[i]),
        ]
        for i in range(len(wave.frame_indices))
    ]
    _write_csv(
        run_dir / "signals.csv",
        [
            "frame_index",
            "timestamp_sec",
            "raw_tracker",
            "raw_flow",
            "normalized_tracker",
            "normalized_flow",
            "tracker_weight",
            "flow_weight",
            "motion_wave",
        ],
        signal_rows,
    )

    filtered_rows = list(
        zip(filt.uniform_timestamps_sec.tolist(), filt.unfiltered_signal.tolist(), filt.filtered_signal.tolist())
    )
    _write_csv(
        run_dir / "filtered_signal.csv",
        ["uniform_timestamp_sec", "unfiltered_signal", "filtered_signal"],
        filtered_rows,
    )

    peak_rows = [[float(t)] for t in estimates.peaks.peak_timestamps_sec]
    _write_csv(run_dir / "peaks.csv", ["peak_timestamp_sec"], peak_rows)

    estimator_payload = {
        "cwt": _jsonable(asdict(estimates.cwt)),
        "autocorrelation": _jsonable(asdict(estimates.autocorrelation)),
        "fft": _jsonable(asdict(estimates.fft)),
        "peaks": _jsonable(asdict(estimates.peaks)),
        "fusion": {
            "candidates": [_jsonable(asdict(c)) for c in fusion_result.candidates],
            "center_cpm": fusion_result.center_cpm,
            "final_cpm": fusion_result.final_cpm,
            "overall_confidence": fusion_result.overall_confidence,
            "spread_cpm": fusion_result.spread_cpm,
            "std_cpm": fusion_result.std_cpm,
        },
    }
    with open(run_dir / "estimator_results.json", "w", encoding="utf-8") as f:
        json.dump(estimator_payload, f, indent=2)

    with open(run_dir / "repnet_result.json", "w", encoding="utf-8") as f:
        json.dump(_jsonable(asdict(repnet_result)), f, indent=2)

    summary = {
        "video_id": video_id,
        "gt_cpm": dev_video.gt.gt_cpm,
        "final_cpm": fusion_result.final_cpm,
        "signed_error": fusion_result.final_cpm - dev_video.gt.gt_cpm,
        "absolute_error": abs(fusion_result.final_cpm - dev_video.gt.gt_cpm),
        "overall_confidence": fusion_result.overall_confidence,
        "mediapipe_detection_rate": mp_result.detection_rate,
        "cotracker_mean_valid_ratio": float(np.mean(ct_result.valid_ratio_per_frame)),
        "git_commit": get_git_commit_hash(),
        "config_hash": config.config_hash(),
        "seed": config.project.seed,
    }
    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("%s: diagnostics saved to %s", video_id, run_dir)
    return run_dir


def save_diagnostics_for_all_videos(
    dev_videos: list[DevVideo], config: HybridConfig, cache_manager: CacheManager
) -> list[Path]:
    return [save_diagnostics_for_video(v, config, cache_manager) for v in dev_videos]
