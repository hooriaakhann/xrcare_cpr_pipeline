"""Development-set discovery and ground-truth mapping.

CLAUDE.md rules 1-3: only `data/split/*_development.mp4` may be used during
implementation/tuning; `*_test.mp4` stays untouched until the pipeline is
frozen. GT lives in `data/metadata/ground_truth_summary.csv`, keyed by
`video{N}.mp4`; split files map to it by stripping the `_development`/`_test`
suffix (e.g. `video4_development.mp4` -> `video4.mp4`). Any filename that
doesn't fit this pattern, or maps to a missing/duplicate GT row, raises
`GroundTruthMappingError` rather than being silently skipped or guessed at.
"""

from __future__ import annotations

import csv
import glob
import re
from dataclasses import dataclass
from pathlib import Path

from hybrid.config import HybridConfig
from hybrid.exceptions import GroundTruthMappingError, VideoReadError
from hybrid.logging_config import get_logger

logger = get_logger(__name__)

_SPLIT_SUFFIX_RE = re.compile(r"^(?P<video_id>.+)_(development|test)$")


@dataclass(frozen=True)
class GroundTruth:
    filename: str
    cpr_start_sec: float
    cpr_end_sec: float
    cpr_duration_sec: float
    known_compression_count: int
    gt_cpm: float


@dataclass(frozen=True)
class DevVideo:
    video_id: str  # e.g. "video4"
    split_path: Path  # data/split/video4_development.mp4
    gt: GroundTruth


def load_ground_truth(csv_path: Path) -> dict[str, GroundTruth]:
    """Ground-truth rows keyed by GT filename (e.g. "video4.mp4")."""
    path = Path(csv_path)
    if not path.exists():
        raise GroundTruthMappingError(f"Ground-truth CSV not found: {path}")

    rows: dict[str, GroundTruth] = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row["filename"].strip()
            if filename in rows:
                raise GroundTruthMappingError(f"Duplicate ground-truth row for {filename!r} in {path}")
            try:
                rows[filename] = GroundTruth(
                    filename=filename,
                    cpr_start_sec=float(row["cpr_start_sec"]),
                    cpr_end_sec=float(row["cpr_end_sec"]),
                    cpr_duration_sec=float(row["cpr_duration_sec"]),
                    known_compression_count=int(row["known_compression_count"]),
                    gt_cpm=float(row["gt_cpm"]),
                )
            except (KeyError, ValueError) as e:
                raise GroundTruthMappingError(f"Malformed ground-truth row for {filename!r} in {path}: {e}") from e

    if not rows:
        raise GroundTruthMappingError(f"Ground-truth CSV has no data rows: {path}")
    return rows


def map_split_filename_to_gt_key(split_filename: str) -> str:
    """ "video4_development.mp4" -> "video4.mp4".

    Raises GroundTruthMappingError if the filename doesn't end in
    `_development` or `_test` before the extension — an unrecognized naming
    scheme must stop the run, not be guessed at.
    """
    p = Path(split_filename)
    match = _SPLIT_SUFFIX_RE.match(p.stem)
    if match is None:
        raise GroundTruthMappingError(
            f"Cannot map {split_filename!r} to a ground-truth row: stem does not end in '_development' or '_test'"
        )
    return f"{match.group('video_id')}{p.suffix}"


def _discover_split_videos(config: HybridConfig, glob_pattern: str, split_label: str) -> list[DevVideo]:
    """Shared body for `discover_development_videos`/`discover_test_videos`.

    `DevVideo` is reused as the return type for test videos too -- its
    fields (video_id, split_path, gt) aren't development-specific, and a
    structurally identical `TestVideo` dataclass would just duplicate it.
    """
    gt_rows = load_ground_truth(config.paths.ground_truth_csv)

    name_pattern = Path(glob_pattern).name
    pattern = str(config.paths.split_dir / name_pattern)
    paths = sorted(Path(p) for p in glob.glob(pattern))

    if not paths:
        raise VideoReadError(f"No {split_label} videos found matching {pattern!r}")

    videos: list[DevVideo] = []
    seen_ids: set[str] = set()
    for path in paths:
        gt_key = map_split_filename_to_gt_key(path.name)
        if gt_key not in gt_rows:
            raise GroundTruthMappingError(
                f"{path.name!r} maps to ground-truth key {gt_key!r}, which has no row in "
                f"{config.paths.ground_truth_csv}"
            )
        video_id = Path(gt_key).stem
        if video_id in seen_ids:
            raise GroundTruthMappingError(f"Duplicate video id {video_id!r} discovered from {path}")
        seen_ids.add(video_id)
        videos.append(DevVideo(video_id=video_id, split_path=path, gt=gt_rows[gt_key]))
        logger.info("Mapped %s -> GT row %s (gt_cpm=%.4f)", path.name, gt_key, gt_rows[gt_key].gt_cpm)

    return videos


def discover_development_videos(config: HybridConfig) -> list[DevVideo]:
    """Every development-split video matched to its ground-truth row.

    The glob's directory comes from `config.paths.split_dir` (already
    resolved absolute); the filename pattern comes from
    `config.video.development_glob`, so both stay config-driven per the
    engineering standards rather than hardcoding either.
    """
    return _discover_split_videos(config, config.video.development_glob, "development")


def discover_test_videos(config: HybridConfig) -> list[DevVideo]:
    """Every held-out test-split video matched to its ground-truth row.

    Mirrors `discover_development_videos`, using `config.video.test_glob`.
    Only `hybrid.run_test` may call this -- CLAUDE.md rules 1-2, enforced
    there by `guard_test_only`, not by this function.
    """
    return _discover_split_videos(config, config.video.test_glob, "test")
