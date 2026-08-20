"""RepNet Branch (Phase 10).

RepNet (Dwibedi et al., "Counting Out Time," CVPR 2020) is an independent
learned-periodicity branch, pretrained only (CLAUDE.md rule 4) -- its
official reference implementation ships only as a Colab notebook
(google-research/google-research, repnet/repnet_colab.ipynb), which was
vendored (near-verbatim, only Colab-specific pieces like matplotlib
visualization/webcam capture stripped) into `src/repnet_env/`. It requires
TensorFlow, which conflicts with nothing the main venv already has -- but
given the size of the checkpoint (~300MB) and TF's footprint, it's kept in
its own isolated `.venv-tf` environment (ADR 0003) and invoked through a
subprocess, per the Phase 0.5 default rather than merging it into the main
venv the way CoTracker's torch dependency was (see Phase 3 PROGRESS.md --
that merge was fine only because torch and mediapipe didn't conflict; TF
and the rest of this project's stack were never checked for that, and
subprocess isolation sidesteps the question entirely).

Evaluated on the full CPR-active development video (spec's "Variant A") --
Variant B (a separately stabilized/cropped input) is out of scope for this
pass; noted as a deliberate deferral, not an oversight, in PROGRESS.md.

RepNet finding no clear periodicity is a valid low-confidence *result*
(confidence -> 0), not a pipeline failure -- CLAUDE.md rule 11 (never hide
failures) and the fusion design (Phase 12) both expect branches to report
uncertainty rather than being silently excluded or crashing the run.
`RepNetUnavailableError` is reserved for actual infrastructure failures:
missing venv, missing checkpoint, subprocess crash/timeout.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from hybrid.caching import CacheManager
from hybrid.config import HybridConfig, RepNetConfig
from hybrid.exceptions import RepNetUnavailableError
from hybrid.logging_config import get_logger
from hybrid.models import ensure_repnet_checkpoint

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPNET_ENV_DIR = PROJECT_ROOT / "src" / "repnet_env"
REPNET_VENV_DIR = PROJECT_ROOT / ".venv-tf"


@dataclass(frozen=True)
class RepNetResult:
    video_id: str
    cpm: float | None  # None if RepNet found no periodicity it trusted
    confidence: float  # 0.0-1.0
    pred_period_frames: float
    chosen_stride: int
    fps: float
    num_frames: int
    reason: str | None  # why cpm is None, if it is


def _venv_python(venv_dir: Path) -> Path:
    if sys.platform == "win32":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _check_environment(venv_dir: Path) -> None:
    python_path = _venv_python(venv_dir)
    if not python_path.exists():
        raise RepNetUnavailableError(
            f"RepNet's isolated TensorFlow venv not found at {venv_dir} (expected interpreter at "
            f"{python_path}). Run: python -m venv {venv_dir.name} && "
            f"{venv_dir.name}/{'Scripts' if sys.platform == 'win32' else 'bin'}/python -m pip install "
            f"tensorflow opencv-python-headless scipy"
        )


def run_repnet_on_video(video_path: Path, config: HybridConfig, video_id: str) -> RepNetResult:
    r_config: RepNetConfig = config.repnet
    _check_environment(REPNET_VENV_DIR)
    checkpoint_dir = ensure_repnet_checkpoint(config.paths.models_dir)

    python_path = _venv_python(REPNET_VENV_DIR)
    script_path = REPNET_ENV_DIR / "run_repnet.py"
    output_json_path = config.paths.cache_dir / f"_repnet_tmp_{video_id}.json"
    output_json_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(python_path),
        str(script_path),
        "--video",
        str(video_path),
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--output-json",
        str(output_json_path),
        "--threshold",
        str(r_config.threshold),
        "--within-period-threshold",
        str(r_config.within_period_threshold),
        "--strides",
        ",".join(str(s) for s in r_config.strides),
        "--batch-size",
        str(r_config.batch_size),
    ]
    if r_config.constant_speed:
        cmd.append("--constant-speed")
    if r_config.median_filter:
        cmd.append("--median-filter")
    if r_config.fully_periodic:
        cmd.append("--fully-periodic")

    logger.info("%s: launching RepNet subprocess (isolated TF venv)", video_id)
    try:
        proc = subprocess.run(
            cmd,
            cwd=REPNET_ENV_DIR,
            capture_output=True,
            text=True,
            timeout=r_config.subprocess_timeout_sec,
        )
    except subprocess.TimeoutExpired as e:
        raise RepNetUnavailableError(
            f"{video_id}: RepNet subprocess exceeded subprocess_timeout_sec={r_config.subprocess_timeout_sec}s"
        ) from e

    if proc.returncode != 0:
        raise RepNetUnavailableError(
            f"{video_id}: RepNet subprocess failed (exit {proc.returncode}): {proc.stderr.strip()[-1000:]}"
        )

    try:
        with open(output_json_path, encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        raise RepNetUnavailableError(f"{video_id}: RepNet subprocess produced no readable output JSON: {e}") from e
    finally:
        output_json_path.unlink(missing_ok=True)

    if payload["reason"]:
        logger.info("%s: RepNet found no trusted periodicity -- %s", video_id, payload["reason"])
    else:
        logger.info(
            "%s: RepNet -> %.1f CPM (confidence=%.2f, stride=%d)",
            video_id,
            payload["cpm"],
            payload["confidence"],
            payload["chosen_stride"],
        )

    return RepNetResult(
        video_id=video_id,
        cpm=payload["cpm"],
        confidence=payload["confidence"],
        pred_period_frames=payload["pred_period_frames"],
        chosen_stride=payload["chosen_stride"],
        fps=payload["fps"],
        num_frames=payload["num_frames"],
        reason=payload["reason"],
    )


def _repnet_config_hash(config: HybridConfig) -> str:
    import hashlib

    payload = config.repnet.model_dump_json()
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def run_repnet_on_video_cached(
    video_path: Path, config: HybridConfig, video_id: str, cache_manager: CacheManager
) -> RepNetResult:
    """RepNet is the most expensive branch (spec, Phase 2) -- always cached."""
    config_hash = _repnet_config_hash(config)
    key = cache_manager.make_key("repnet", video_id, config_hash)

    if config.caching.enabled and cache_manager.exists(key):
        logger.info("%s: loaded RepNet result from cache (key=%s)", video_id, key)
        return cache_manager.load(key)

    result = run_repnet_on_video(video_path, config, video_id)
    if config.caching.enabled:
        cache_manager.save(key, result)
    return result
