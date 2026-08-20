"""Pinned pretrained-model checkpoints.

Downloaded on demand into `models/` (gitignored — large binaries) and never
fine-tuned (CLAUDE.md rule 4). Each entry pins an exact source URL and sha256
so the checkpoint used for a given run is reproducible even though the binary
itself isn't versioned in git — the pinned hash is what belongs in
`summary.json`/the experiment ledger, not the mutable "latest" URL.
"""

from __future__ import annotations

import hashlib
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from hybrid.exceptions import HybridError
from hybrid.logging_config import get_logger

logger = get_logger(__name__)


class ModelDownloadError(HybridError):
    """A pinned model checkpoint could not be downloaded or failed checksum verification."""


@dataclass(frozen=True)
class PinnedModel:
    filename: str
    url: str
    sha256: str


HAND_LANDMARKER = PinnedModel(
    filename="hand_landmarker.task",
    url=(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/latest/hand_landmarker.task"
    ),
    sha256="fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_model(model: PinnedModel, models_dir: Path) -> Path:
    """Local path to `model`, downloading it if missing and verifying its
    sha256 either way. Raises ModelDownloadError on download failure or
    checksum mismatch rather than silently proceeding with a possibly-wrong
    or corrupt checkpoint.
    """
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / model.filename

    if not path.exists():
        logger.info("Downloading pretrained model %s from %s", model.filename, model.url)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        try:
            urllib.request.urlretrieve(model.url, tmp_path)
            tmp_path.replace(path)
        except (OSError, urllib.error.URLError) as e:
            tmp_path.unlink(missing_ok=True)
            raise ModelDownloadError(f"Failed to download {model.filename} from {model.url}: {e}") from e

    actual = _sha256(path)
    if actual != model.sha256:
        raise ModelDownloadError(
            f"{path} sha256 mismatch: expected {model.sha256}, got {actual}. "
            f"Delete the file and re-run to re-download, or the pinned checksum in models.py is stale."
        )
    return path
