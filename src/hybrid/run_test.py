"""Held-out test-set pipeline entrypoint.

Usage:
    python -m hybrid.run_test                  # full test-set evaluation
    python -m hybrid.run_test VIDEO [VIDEO...]  # just these video paths

This is the ONLY script in the repo allowed to load `*_test.mp4` files --
it refuses anything else before it reaches the pipeline, mirroring
`run_development.py`'s Phase 18 guard (CLAUDE.md rules 1-2).

Always loads `config/frozen.yaml` explicitly, never `config/default.yaml`
-- a test run's result is tied to one specific, tagged, immutable config
snapshot (see PROGRESS.md's Phase 15 entry and the `frozen-for-test-v1`
git tag) regardless of what `default.yaml` happens to contain by the time
this runs.

Run exactly once. From the point `config/frozen.yaml` was created,
estimator logic, the fusion formula, confidence weighting, and the frozen
config itself must not change in response to anything this script
reports -- tuning against test results would quietly turn the test set
into another dev set. A genuine bug found later gets fixed, re-frozen
under a new tag/config snapshot, and re-run from scratch; it does not get
patched in place to chase a number.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from hybrid.caching import CacheManager
from hybrid.config import PROJECT_ROOT, HybridConfig, load_config
from hybrid.dataset import DevVideo, discover_test_videos, load_ground_truth, map_split_filename_to_gt_key
from hybrid.evaluation import VideoEvaluationResult, _bootstrap_mae_ci, run_full_pipeline_on_video
from hybrid.exceptions import HeldOutVideoError, HybridError
from hybrid.experiment_ledger import log_run
from hybrid.logging_config import get_logger, setup_logging

logger = get_logger(__name__)

FROZEN_CONFIG_PATH = PROJECT_ROOT / "config" / "frozen.yaml"


def guard_test_only(path: Path) -> None:
    """Raises HeldOutVideoError for anything that isn't a `*_test.mp4`
    file. Call this before any video path reaches the pipeline -- never
    trust a caller-supplied path implicitly. Mirror image of
    `run_development.py`'s `guard_development_only`: that script refuses
    everything except `*_development.mp4`, this one refuses everything
    except `*_test.mp4`.
    """
    path = Path(path)
    if not path.name.endswith("_test.mp4"):
        raise HeldOutVideoError(
            f"Refusing to process {path.name!r}: only '*_test.mp4' files may be processed here. "
            f"This is the dedicated held-out test-set runner -- development videos belong to "
            f"hybrid.run_development, not this script."
        )


def _test_video_for_path(path: Path, config: HybridConfig) -> DevVideo:
    guard_test_only(path)
    gt_rows = load_ground_truth(config.paths.ground_truth_csv)
    gt_key = map_split_filename_to_gt_key(path.name)
    if gt_key not in gt_rows:
        raise HeldOutVideoError(f"{path.name!r} maps to ground-truth key {gt_key!r}, which has no row")
    video_id = Path(gt_key).stem
    return DevVideo(video_id=video_id, split_path=path, gt=gt_rows[gt_key])


def _aggregate_metrics(
    per_video: list[VideoEvaluationResult], seed: int, num_bootstrap_resamples: int = 2000
) -> dict[str, float]:
    """Same formulas, and the same percentile-bootstrap MAE CI, as
    `evaluation.run_development_evaluation` -- standard reporting, not a
    pipeline/config behavior change, so it applies equally to a one-shot
    held-out run."""
    signed_errors = np.array([r.signed_error for r in per_video])
    absolute_errors = np.array([r.absolute_error for r in per_video])
    ci_low, ci_high = _bootstrap_mae_ci(absolute_errors, num_bootstrap_resamples, seed)
    return {
        "mae": float(np.mean(absolute_errors)),
        "rmse": float(np.sqrt(np.mean(signed_errors**2))),
        "mean_signed_error": float(np.mean(signed_errors)),
        "median_absolute_error": float(np.median(absolute_errors)),
        "max_absolute_error": float(np.max(absolute_errors)),
        "mae_bootstrap_ci_95_lower": ci_low,
        "mae_bootstrap_ci_95_upper": ci_high,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("videos", nargs="*", help="specific *_test.mp4 paths; omit to run the full held-out test set")
    args = parser.parse_args(argv)

    config = load_config(path=FROZEN_CONFIG_PATH)
    setup_logging(config.logging.level, config.logging.log_dir)
    cache_manager = CacheManager(config.paths.cache_dir)

    logger.info("Loaded frozen config from %s (config_hash=%s)", FROZEN_CONFIG_PATH, config.config_hash())

    if args.videos:
        test_videos = [_test_video_for_path(Path(p), config) for p in args.videos]
    else:
        test_videos = discover_test_videos(config)

    per_video: list[VideoEvaluationResult] = []
    exit_code = 0
    for test_video in test_videos:
        try:
            result = run_full_pipeline_on_video(test_video, config, cache_manager)
        except HybridError as e:
            logger.error("%s: pipeline failed: %s", test_video.video_id, e)
            exit_code = 1
            continue
        per_video.append(result)
        logger.info(
            "%s: gt=%.2f cwt=%.2f autocorr=%.2f fft=%.2f peaks=%.2f repnet=%s final=%.2f "
            "signed_error=%+.2f absolute_error=%.2f confidence=%.3f runtime=%.1fs",
            result.video_id,
            result.gt_cpm,
            result.cwt_cpm,
            result.autocorrelation_cpm,
            result.fft_cpm,
            result.peaks_cpm,
            f"{result.repnet_cpm:.2f}" if result.repnet_cpm is not None else "None",
            result.final_cpm,
            result.signed_error,
            result.absolute_error,
            result.overall_confidence,
            result.runtime_sec,
        )

    if per_video:
        aggregate = _aggregate_metrics(per_video, config.project.seed)
        logger.info(
            "Test-set evaluation: MAE=%.3f (95%% CI [%.3f, %.3f]) RMSE=%.3f mean_signed_error=%+.3f "
            "median_AE=%.3f max_AE=%.3f over %d video(s)",
            aggregate["mae"],
            aggregate["mae_bootstrap_ci_95_lower"],
            aggregate["mae_bootstrap_ci_95_upper"],
            aggregate["rmse"],
            aggregate["mean_signed_error"],
            aggregate["median_absolute_error"],
            aggregate["max_absolute_error"],
            len(per_video),
        )
        log_run(
            phase="test_evaluation",
            config=config,
            metrics=aggregate,
            extra={
                "config_path": str(FROZEN_CONFIG_PATH),
                "per_video": [
                    {
                        "video_id": r.video_id,
                        "gt_cpm": r.gt_cpm,
                        "final_cpm": r.final_cpm,
                        "signed_error": r.signed_error,
                        "absolute_error": r.absolute_error,
                        "overall_confidence": r.overall_confidence,
                    }
                    for r in per_video
                ],
            },
        )

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
