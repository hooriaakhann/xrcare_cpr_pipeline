import pytest
from pydantic import ValidationError

from hybrid.config import HybridConfig, load_config
from hybrid.exceptions import ConfigError


def test_default_config_loads_and_validates():
    config = load_config()
    assert config.project.seed == 42
    assert config.video.development_glob == "data/split/*_development.mp4"
    assert config.video.test_glob == "data/split/*_test.mp4"


def test_paths_are_resolved_to_absolute():
    config = load_config()
    assert config.paths.raw_dir.is_absolute()
    assert config.paths.raw_dir.name == "raw"
    assert config.paths.ground_truth_csv.name == "ground_truth_summary.csv"


def test_missing_config_file_raises_config_error(tmp_path):
    with pytest.raises(ConfigError):
        load_config(path=tmp_path / "does_not_exist.yaml")


def test_malformed_yaml_raises_config_error(tmp_path):
    bad_file = tmp_path / "bad.yaml"
    bad_file.write_text("project: [unterminated", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(path=bad_file)


def test_unknown_field_fails_schema_validation():
    with pytest.raises(ValidationError):
        HybridConfig(project={"seed": 1, "not_a_real_field": True})


def test_config_hash_is_stable_and_sensitive_to_changes():
    a = load_config()
    b = load_config()
    assert a.config_hash() == b.config_hash()

    c = load_config()
    c.project.seed = 999
    assert c.config_hash() != a.config_hash()


@pytest.mark.parametrize(
    "overrides",
    [
        {"min_hand_detection_confidence": 1.5},
        {"min_hand_detection_confidence": -0.1},
        {"max_num_hands": 0},
        {"roi_padding_ratio": -1.0},
        {"max_detection_gap_sec": -1.0},
    ],
)
def test_mediapipe_config_rejects_invalid_values(overrides):
    with pytest.raises(ValidationError):
        HybridConfig(mediapipe=overrides)


@pytest.mark.parametrize(
    "overrides",
    [
        {"num_points": 0},
        {"visibility_threshold": 1.5},
        {"reinit_visibility_threshold": -0.1},
        {"max_reinits": -1},
        {"window_frames": 0},
        {"working_max_dim": 0},
        {"max_track_loss_sec": -1.0},
    ],
)
def test_cotracker_config_rejects_invalid_values(overrides):
    with pytest.raises(ValidationError):
        HybridConfig(cotracker=overrides)


def test_cotracker_config_allows_zero_max_reinits():
    # 0 is legitimate -- disables reinitialization entirely (ablation use case).
    config = HybridConfig(cotracker={"max_reinits": 0})
    assert config.cotracker.max_reinits == 0
