"""Schema-validated config. Every parameter used anywhere in the pipeline
flows through this object — no module reads a hardcoded constant or a raw
YAML dict directly. Later phases (branches, fusion, tuning) extend
`HybridConfig` with their own sub-models rather than adding ad-hoc args.
"""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from hybrid.exceptions import ConfigError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"


class StrictModel(BaseModel):
    """Base for all config models: unknown keys fail validation instead of
    being silently ignored (a typo'd YAML key should error, not vanish).
    """

    model_config = ConfigDict(extra="forbid")


class PathsConfig(StrictModel):
    """All paths are resolved to absolute, relative to the project root."""

    data_root: Path = Path("data")
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    split_dir: Path = Path("data/split")
    metadata_dir: Path = Path("data/metadata")
    ground_truth_csv: Path = Path("data/metadata/ground_truth_summary.csv")
    cache_dir: Path = Path("data/cache")
    runs_dir: Path = Path("runs")
    models_dir: Path = Path("models")

    def resolve(self, root: Path) -> "PathsConfig":
        return PathsConfig(**{field: root / value for field, value in self})


class VideoConfig(StrictModel):
    """Glob patterns for the dev/test split. See CLAUDE.md rule 1-2:
    development videos only during implementation/tuning, test videos stay
    held out until the pipeline is frozen.
    """

    development_glob: str = "data/split/*_development.mp4"
    test_glob: str = "data/split/*_test.mp4"


class LoggingConfig(StrictModel):
    level: str = "INFO"
    log_dir: Path = Path("runs/logs")


class CachingConfig(StrictModel):
    enabled: bool = True


class ProjectConfig(StrictModel):
    seed: int = 42


class MediaPipeConfig(StrictModel):
    """Hand/wrist localization branch (Phase 2). Detection is the semantic
    signal — it does not compute CPM itself, only the dynamic foreground ROI
    and CoTracker init points.
    """

    min_hand_detection_confidence: float = 0.5
    min_hand_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    max_num_hands: int = 1
    # ROI = hand bbox expanded by this fraction of its own size on each side,
    # then extended toward the forearm (opposite the middle-finger-MCP ->
    # wrist direction) by forearm_extension_ratio x the bbox's long side.
    roi_padding_ratio: float = 0.5
    forearm_extension_ratio: float = 1.2
    # A gap in successful hand detection longer than this is logged as a
    # warning (not silently absorbed) — see Phase 2 diagnostics.
    max_detection_gap_sec: float = 1.0

    @field_validator("min_hand_detection_confidence", "min_hand_presence_confidence", "min_tracking_confidence")
    @classmethod
    def _in_unit_interval(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"must be in [0.0, 1.0], got {v}")
        return v

    @field_validator("max_num_hands")
    @classmethod
    def _positive_hand_count(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"max_num_hands must be >= 1, got {v}")
        return v

    @field_validator("roi_padding_ratio", "forearm_extension_ratio", "max_detection_gap_sec")
    @classmethod
    def _non_negative(cls, v: float) -> float:
        if v < 0.0:
            raise ValueError(f"must be >= 0.0, got {v}")
        return v


class HybridConfig(StrictModel):
    project: ProjectConfig = Field(default_factory=ProjectConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    video: VideoConfig = Field(default_factory=VideoConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    caching: CachingConfig = Field(default_factory=CachingConfig)
    mediapipe: MediaPipeConfig = Field(default_factory=MediaPipeConfig)

    def config_hash(self) -> str:
        """Short hash of the fully-resolved config, for cache keys and the experiment ledger."""
        import hashlib

        digest = hashlib.sha256(self.model_dump_json().encode("utf-8"))
        return digest.hexdigest()[:12]


def load_config(path: Path | None = None, project_root: Path | None = None) -> HybridConfig:
    """Load and validate config from YAML. Raises ConfigError on any failure
    (missing file, malformed YAML, schema violation) — never returns a
    partially-valid config.
    """
    path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    root = Path(project_root) if project_root is not None else PROJECT_ROOT

    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")

    try:
        raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as e:
        raise ConfigError(f"Malformed YAML in {path}: {e}") from e

    try:
        config = HybridConfig(**raw)
    except ValidationError as e:
        raise ConfigError(f"Config validation failed for {path}:\n{e}") from e

    config.paths = config.paths.resolve(root)
    config.logging.log_dir = root / config.logging.log_dir
    return config
