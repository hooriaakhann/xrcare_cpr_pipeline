"""Per-video cache for expensive branch outputs (MediaPipe, CoTracker,
RepNet, ...) so tuning iterations don't re-run inference. Cache keys are
scoped by (stage, video_id, config_hash) so a config change invalidates
only the entries it affects.
"""

import hashlib
import pickle
from pathlib import Path
from typing import Any

from hybrid.exceptions import CacheError
from hybrid.logging_config import get_logger

logger = get_logger(__name__)


class CacheManager:
    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def make_key(stage: str, video_id: str, config_hash: str = "") -> str:
        raw = f"{stage}:{video_id}:{config_hash}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]

    def _path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.pkl"

    def exists(self, key: str) -> bool:
        return self._path(key).exists()

    def load(self, key: str) -> Any:
        path = self._path(key)
        if not path.exists():
            raise CacheError(f"No cache entry for key {key!r} at {path}")
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except (pickle.PickleError, EOFError, OSError) as e:
            raise CacheError(f"Corrupt cache entry at {path}: {e}") from e

    def save(self, key: str, value: Any) -> Path:
        path = self._path(key)
        tmp_path = path.with_suffix(".pkl.tmp")
        try:
            with open(tmp_path, "wb") as f:
                pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
            tmp_path.replace(path)
        except (pickle.PickleError, OSError) as e:
            tmp_path.unlink(missing_ok=True)
            raise CacheError(f"Failed to write cache entry at {path}: {e}") from e
        return path

    def clear(self, key: str) -> None:
        self._path(key).unlink(missing_ok=True)
