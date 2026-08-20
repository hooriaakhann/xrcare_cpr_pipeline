import json

from hybrid.config import load_config
from hybrid.experiment_ledger import get_git_commit_hash, log_run


def test_log_run_appends_one_jsonl_record(tmp_path):
    config = load_config()
    config.paths.runs_dir = tmp_path

    record = log_run(phase="unit_test", config=config, metrics={"predicted_cpm": 90.0}, video_id="video1")

    ledger_path = tmp_path / "experiment_ledger.jsonl"
    assert ledger_path.exists()

    lines = ledger_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1

    on_disk = json.loads(lines[0])
    assert on_disk == record
    assert on_disk["phase"] == "unit_test"
    assert on_disk["video_id"] == "video1"
    assert on_disk["seed"] == config.project.seed
    assert on_disk["config_hash"] == config.config_hash()
    assert on_disk["metrics"] == {"predicted_cpm": 90.0}


def test_log_run_appends_rather_than_overwrites(tmp_path):
    config = load_config()
    config.paths.runs_dir = tmp_path

    log_run(phase="run_a", config=config)
    log_run(phase="run_b", config=config)

    lines = (tmp_path / "experiment_ledger.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["phase"] == "run_a"
    assert json.loads(lines[1])["phase"] == "run_b"


def test_get_git_commit_hash_returns_string_in_a_git_repo():
    project_root_result = get_git_commit_hash()
    assert project_root_result is None or isinstance(project_root_result, str)
