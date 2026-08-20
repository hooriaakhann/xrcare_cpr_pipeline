"""Append-only experiment ledger. Every dev run / tuning iteration gets one
JSON line in `<runs_dir>/experiment_ledger.jsonl`, tagged with everything
needed to reproduce it: git commit hash, config hash, and seed.
"""

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hybrid.config import PROJECT_ROOT, HybridConfig
from hybrid.logging_config import get_logger

logger = get_logger(__name__)


def get_git_commit_hash(project_root: Path = PROJECT_ROOT) -> str | None:
    """Returns the current HEAD commit hash, or None if unavailable (no git repo, no commits yet)."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def log_run(
    phase: str,
    config: HybridConfig,
    metrics: dict[str, Any] | None = None,
    video_id: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Append one record to the experiment ledger and return it.

    Args:
        phase: name of the pipeline phase/stage producing this run (e.g. "mediapipe_branch", "fusion_tuning").
        config: the resolved config used for this run (supplies config_hash and seed).
        metrics: any numeric/summary results worth tracking (e.g. predicted CPM, error vs GT).
        video_id: video this run pertains to, if any (e.g. "video4").
        extra: anything else worth recording (hyperparameter overrides, notes).
    """
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "phase": phase,
        "video_id": video_id,
        "git_commit": get_git_commit_hash(),
        "config_hash": config.config_hash(),
        "seed": config.project.seed,
        "metrics": metrics or {},
        "extra": extra or {},
    }

    ledger_path = config.paths.runs_dir / "experiment_ledger.jsonl"
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with open(ledger_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    logger.info(
        "Logged experiment run: phase=%s video_id=%s config_hash=%s",
        phase,
        video_id,
        record["config_hash"],
    )
    return record
