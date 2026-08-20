"""Development-set pipeline entrypoint (Phase 18 test-safety guard).

Usage:
    python -m hybrid.run_development                  # full dev-set evaluation (Phase 13)
    python -m hybrid.run_development VIDEO [VIDEO...]  # just these video paths

Refuses anything that isn't a `*_development.mp4` file -- CLAUDE.md rules
1-2, enforced here rather than trusted to the caller. There is no
test-video equivalent of this script: Phase 18 explicitly forbids writing
one until the pipeline is finalized and frozen. The held-out test person
must remain unseen.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from hybrid.caching import CacheManager
from hybrid.config import HybridConfig, load_config
from hybrid.dataset import DevVideo, discover_development_videos, load_ground_truth, map_split_filename_to_gt_key
from hybrid.evaluation import run_full_pipeline_on_video
from hybrid.exceptions import HeldOutVideoError, HybridError
from hybrid.logging_config import get_logger, setup_logging

logger = get_logger(__name__)


def guard_development_only(path: Path) -> None:
    """Raises HeldOutVideoError for anything that isn't a
    `*_development.mp4` file. Call this before any video path reaches the
    pipeline -- never trust a caller-supplied path implicitly.
    """
    path = Path(path)
    if not path.name.endswith("_development.mp4"):
        raise HeldOutVideoError(
            f"Refusing to process {path.name!r}: only '*_development.mp4' files may be processed here "
            f"(CLAUDE.md rules 1-2). Test videos stay held out until the pipeline is finalized and "
            f"frozen; there is no test-run script yet, and Phase 18 forbids writing one now."
        )


def _dev_video_for_path(path: Path, config: HybridConfig) -> DevVideo:
    guard_development_only(path)
    gt_rows = load_ground_truth(config.paths.ground_truth_csv)
    gt_key = map_split_filename_to_gt_key(path.name)
    if gt_key not in gt_rows:
        raise HeldOutVideoError(f"{path.name!r} maps to ground-truth key {gt_key!r}, which has no row")
    video_id = Path(gt_key).stem
    return DevVideo(video_id=video_id, split_path=path, gt=gt_rows[gt_key])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "videos", nargs="*", help="specific *_development.mp4 paths; omit to run the full development set"
    )
    args = parser.parse_args(argv)

    config = load_config()
    setup_logging(config.logging.level, config.logging.log_dir)
    cache_manager = CacheManager(config.paths.cache_dir)

    if args.videos:
        dev_videos = [_dev_video_for_path(Path(p), config) for p in args.videos]
    else:
        dev_videos = discover_development_videos(config)

    exit_code = 0
    for dev_video in dev_videos:
        try:
            result = run_full_pipeline_on_video(dev_video, config, cache_manager)
        except HybridError as e:
            logger.error("%s: pipeline failed: %s", dev_video.video_id, e)
            exit_code = 1
            continue
        logger.info(
            "%s: GT=%.2f final=%.2f signed_error=%+.2f confidence=%.3f runtime=%.1fs",
            dev_video.video_id,
            result.gt_cpm,
            result.final_cpm,
            result.signed_error,
            result.overall_confidence,
            result.runtime_sec,
        )

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
