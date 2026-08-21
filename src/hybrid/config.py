"""Schema-validated config. Every parameter used anywhere in the pipeline
flows through this object — no module reads a hardcoded constant or a raw
YAML dict directly. Later phases (branches, fusion, tuning) extend
`HybridConfig` with their own sub-models rather than adding ad-hoc args.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

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

    def resolve(self, root: Path) -> PathsConfig:
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


class CoTrackerConfig(StrictModel):
    """Multi-point tracking branch (Phase 3). The pretrained CoTracker3
    "offline" model holds its whole input video tensor in memory at once, so
    videos are processed in memory-bounded temporal windows rather than a
    single call over the full video (this machine has 16GB RAM, no GPU).
    """

    num_points: int = 40
    # A frame's tracked points are "valid" when the visible ratio is at or
    # above this; used for track-loss-period detection.
    visibility_threshold: float = 0.6
    # End-of-window visible ratio below this triggers a MediaPipe reinit
    # (fresh query points) for the next window.
    reinit_visibility_threshold: float = 0.4
    max_reinits: int = 10
    # Longest allowed contiguous run of visibility below visibility_threshold
    # before raising TrackLostError (checked after all reinits are spent).
    max_track_loss_sec: float = 2.0
    window_frames: int = 100
    # Frames are resized so max(H, W) <= this before feeding CoTracker
    # (memory/speed; the model downsamples internally regardless).
    working_max_dim: int = 1024

    @field_validator("num_points")
    @classmethod
    def _positive_num_points(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"num_points must be >= 1, got {v}")
        return v

    @field_validator("visibility_threshold", "reinit_visibility_threshold")
    @classmethod
    def _in_unit_interval(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"must be in [0.0, 1.0], got {v}")
        return v

    @field_validator("window_frames", "working_max_dim")
    @classmethod
    def _positive_int(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"must be >= 1, got {v}")
        return v

    @field_validator("max_reinits")
    @classmethod
    def _non_negative_int(cls, v: int) -> int:
        # 0 is a legitimate value -- it means "never reinitialize", useful
        # for ablations that isolate the effect of the reinit mechanism.
        if v < 0:
            raise ValueError(f"must be >= 0, got {v}")
        return v

    @field_validator("max_track_loss_sec")
    @classmethod
    def _non_negative(cls, v: float) -> float:
        if v < 0.0:
            raise ValueError(f"must be >= 0.0, got {v}")
        return v


class EgoMotionConfig(StrictModel):
    """RANSAC affine (similarity) ego-motion compensation (Phase 4). RNG/RANSAC
    reproducibility uses `project.seed` (CLAUDE.md: "RNG/RANSAC seed recorded
    with every run") rather than a separate seed field here.
    """

    max_features: int = 200  # goodFeaturesToTrack maxCorners
    quality_level: float = 0.01
    min_distance: float = 8.0
    roi_exclusion_dilate_px: int = 20  # extra margin excluded around the CPR ROI
    ransac_reproj_threshold: float = 3.0
    min_inlier_ratio: float = 0.5
    max_rotation_deg: float = 15.0  # sanity bound: head motion is not full spins
    min_scale: float = 0.8
    max_scale: float = 1.25
    # Longest allowed contiguous run of an unreliable transform before
    # raising EgoMotionUnreliableError.
    max_unreliable_span_sec: float = 2.0

    @field_validator("max_features", "roi_exclusion_dilate_px")
    @classmethod
    def _non_negative_int(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"must be >= 0, got {v}")
        return v

    @field_validator("quality_level")
    @classmethod
    def _quality_level_range(cls, v: float) -> float:
        if not 0.0 < v <= 1.0:
            raise ValueError(f"quality_level must be in (0.0, 1.0], got {v}")
        return v

    @field_validator("min_distance", "ransac_reproj_threshold", "max_rotation_deg", "max_unreliable_span_sec")
    @classmethod
    def _non_negative_float(cls, v: float) -> float:
        if v < 0.0:
            raise ValueError(f"must be >= 0.0, got {v}")
        return v

    @field_validator("min_inlier_ratio")
    @classmethod
    def _in_unit_interval(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"must be in [0.0, 1.0], got {v}")
        return v

    @field_validator("min_scale", "max_scale")
    @classmethod
    def _positive_scale(cls, v: float) -> float:
        if v <= 0.0:
            raise ValueError(f"must be > 0.0, got {v}")
        return v

    @model_validator(mode="after")
    def _scale_bounds_ordered(self) -> EgoMotionConfig:
        if self.min_scale > self.max_scale:
            raise ValueError(f"min_scale ({self.min_scale}) must be <= max_scale ({self.max_scale})")
        return self


class OpticalFlowConfig(StrictModel):
    """Farneback dense optical flow (Phase 6). Split into a foreground
    reading (median flow within the current CPR ROI) and a background
    reading (median flow outside it) -- never a single whole-frame mean.
    """

    pyr_scale: float = 0.5
    levels: int = 3
    winsize: int = 15
    iterations: int = 3
    poly_n: int = 5
    poly_sigma: float = 1.2
    # Foreground flow is the median Y-flow of only the top
    # (100 - motion_percentile)% most-moving pixels in the ROI by magnitude
    # -- a plain median over the whole ROI is dominated by its mostly-static
    # majority (skin/background) and washes out the real compression signal
    # (verified against real footage during Phase 6 development).
    motion_percentile: float = 75.0
    # Per-frame flow magnitude (px) beyond this is flagged unstable.
    max_flow_magnitude: float = 50.0
    max_unstable_span_sec: float = 2.0

    @field_validator("pyr_scale")
    @classmethod
    def _pyr_scale_range(cls, v: float) -> float:
        if not 0.0 < v < 1.0:
            raise ValueError(f"pyr_scale must be in (0.0, 1.0), got {v}")
        return v

    @field_validator("motion_percentile")
    @classmethod
    def _motion_percentile_range(cls, v: float) -> float:
        if not 0.0 <= v < 100.0:
            raise ValueError(f"motion_percentile must be in [0.0, 100.0), got {v}")
        return v

    @field_validator("levels", "winsize", "iterations", "poly_n")
    @classmethod
    def _positive_int(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"must be >= 1, got {v}")
        return v

    @field_validator("poly_sigma", "max_flow_magnitude", "max_unstable_span_sec")
    @classmethod
    def _positive_float(cls, v: float) -> float:
        if v <= 0.0:
            raise ValueError(f"must be > 0.0, got {v}")
        return v


class FilteringConfig(StrictModel):
    """Butterworth band-pass filter (Phase 8). Defaults (1.0-3.0 Hz = 60-180
    CPM) deliberately bracket this dev set's observed GT range (69-106 CPM,
    see PROGRESS.md Phase 0) with generous headroom on both sides -- a
    physiologically-motivated clinical range, not fitted tightly to the dev
    videos, since the band must also generalize to the held-out test set.
    """

    butterworth_low_hz: float = 1.0
    butterworth_high_hz: float = 3.0
    butterworth_order: int = 4

    @field_validator("butterworth_low_hz", "butterworth_high_hz")
    @classmethod
    def _positive(cls, v: float) -> float:
        if v <= 0.0:
            raise ValueError(f"must be > 0.0, got {v}")
        return v

    @field_validator("butterworth_order")
    @classmethod
    def _positive_order(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"butterworth_order must be >= 1, got {v}")
        return v

    @model_validator(mode="after")
    def _low_less_than_high(self) -> FilteringConfig:
        if self.butterworth_low_hz >= self.butterworth_high_hz:
            raise ValueError(
                f"butterworth_low_hz ({self.butterworth_low_hz}) must be < "
                f"butterworth_high_hz ({self.butterworth_high_hz})"
            )
        return self


class EstimatorsConfig(StrictModel):
    """Four classical rate estimators (Phase 9): CWT, autocorrelation, FFT,
    peak detection. Frequency/lag search ranges default to the same band as
    `filtering` but are independently tunable (Phase 15) since each
    technique may benefit from a different effective band.
    """

    cwt_freq_range_hz: tuple[float, float] = (1.0, 3.0)
    cwt_wavelet: str = "morl"
    cwt_num_scales: int = 80
    fft_freq_range_hz: tuple[float, float] = (1.0, 3.0)
    autocorr_lag_range_sec: tuple[float, float] = (0.3, 1.0)
    peak_min_distance_sec: float = 0.35
    peak_prominence: float = 0.3

    @field_validator("cwt_freq_range_hz", "fft_freq_range_hz", "autocorr_lag_range_sec")
    @classmethod
    def _range_positive_and_ordered(cls, v: tuple[float, float]) -> tuple[float, float]:
        low, high = v
        if not (0.0 < low < high):
            raise ValueError(f"range must satisfy 0 < low < high, got {v}")
        return v

    @field_validator("cwt_num_scales")
    @classmethod
    def _min_scales(cls, v: int) -> int:
        if v < 2:
            raise ValueError(f"cwt_num_scales must be >= 2, got {v}")
        return v

    @field_validator("peak_min_distance_sec")
    @classmethod
    def _positive(cls, v: float) -> float:
        if v <= 0.0:
            raise ValueError(f"peak_min_distance_sec must be > 0.0, got {v}")
        return v

    @field_validator("peak_prominence")
    @classmethod
    def _non_negative(cls, v: float) -> float:
        if v < 0.0:
            raise ValueError(f"peak_prominence must be >= 0.0, got {v}")
        return v


class RepNetConfig(StrictModel):
    """RepNet learned-periodicity branch (Phase 10). Runs in an isolated
    TensorFlow venv (`.venv-tf`, see ADR 0003) invoked via subprocess -- this
    config only carries inference parameters; the venv/script locations are
    fixed source-tree paths, not runtime-configurable (see repnet_branch.py).
    """

    threshold: float = 0.2
    within_period_threshold: float = 0.5
    strides: tuple[int, ...] = (1, 2, 3, 4)
    batch_size: int = 4
    constant_speed: bool = False
    median_filter: bool = True
    fully_periodic: bool = False
    subprocess_timeout_sec: float = 1800.0

    @field_validator("threshold", "within_period_threshold")
    @classmethod
    def _in_unit_interval(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"must be in [0.0, 1.0], got {v}")
        return v

    @field_validator("strides")
    @classmethod
    def _strides_positive(cls, v: tuple[int, ...]) -> tuple[int, ...]:
        if not v or any(s < 1 for s in v):
            raise ValueError(f"strides must be a non-empty tuple of ints >= 1, got {v}")
        return v

    @field_validator("batch_size")
    @classmethod
    def _positive_batch(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"batch_size must be >= 1, got {v}")
        return v

    @field_validator("subprocess_timeout_sec")
    @classmethod
    def _positive_timeout(cls, v: float) -> float:
        if v <= 0.0:
            raise ValueError(f"subprocess_timeout_sec must be > 0.0, got {v}")
        return v


class FusionConfig(StrictModel):
    """Final hybrid fusion (Phase 12). Beyond plain confidence weighting,
    each candidate estimate is also down-weighted the further it sits from
    the group's confidence-weighted-median center -- "basic disagreement
    handling" per spec, so one low-confidence outlier can't pull the final
    value strongly toward it even if its own confidence isn't dramatically
    low.
    """

    disagreement_scale_cpm: float = 10.0

    @field_validator("disagreement_scale_cpm")
    @classmethod
    def _positive(cls, v: float) -> float:
        if v <= 0.0:
            raise ValueError(f"disagreement_scale_cpm must be > 0.0, got {v}")
        return v


class OverlayVideoConfig(StrictModel):
    """Per-video diagnostic overlay MP4 (not part of inference/estimation --
    a presentation-only artifact combining MediaPipe's ROI, CoTracker's raw
    tracked points, and the ego-motion-corrected motion signal on one video
    for visual sanity-checking). Colors are small fixed constants in
    `overlay_video.py`, not config fields -- they don't affect any
    measurement, only how a debug video looks, so making every BGR tuple a
    tunable parameter would be config bloat with no real benefit.
    """

    point_radius_px: int = 4
    roi_box_thickness_px: int = 2
    motion_strip_height_px: int = 150
    fourcc: str = "mp4v"

    @field_validator("point_radius_px", "roi_box_thickness_px", "motion_strip_height_px")
    @classmethod
    def _positive_int(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"must be > 0, got {v}")
        return v


class HybridConfig(StrictModel):
    project: ProjectConfig = Field(default_factory=ProjectConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    video: VideoConfig = Field(default_factory=VideoConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    caching: CachingConfig = Field(default_factory=CachingConfig)
    mediapipe: MediaPipeConfig = Field(default_factory=MediaPipeConfig)
    cotracker: CoTrackerConfig = Field(default_factory=CoTrackerConfig)
    ego_motion: EgoMotionConfig = Field(default_factory=EgoMotionConfig)
    optical_flow: OpticalFlowConfig = Field(default_factory=OpticalFlowConfig)
    filtering: FilteringConfig = Field(default_factory=FilteringConfig)
    estimators: EstimatorsConfig = Field(default_factory=EstimatorsConfig)
    repnet: RepNetConfig = Field(default_factory=RepNetConfig)
    fusion: FusionConfig = Field(default_factory=FusionConfig)
    overlay_video: OverlayVideoConfig = Field(default_factory=OverlayVideoConfig)

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
