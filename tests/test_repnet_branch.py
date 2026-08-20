import json
import subprocess
import sys

import pytest

from hybrid.caching import CacheManager
from hybrid.config import HybridConfig
from hybrid.exceptions import RepNetUnavailableError
from hybrid.repnet_branch import (
    RepNetResult,
    _check_environment,
    _venv_python,
    run_repnet_on_video,
    run_repnet_on_video_cached,
)

# ---------------------------------------------------------------------------
# Pure-logic tests
# ---------------------------------------------------------------------------


def test_venv_python_platform_specific(tmp_path):
    path = _venv_python(tmp_path)
    if sys.platform == "win32":
        assert path == tmp_path / "Scripts" / "python.exe"
    else:
        assert path == tmp_path / "bin" / "python"


def test_check_environment_raises_when_venv_missing(tmp_path):
    missing_venv = tmp_path / "does_not_exist"
    with pytest.raises(RepNetUnavailableError):
        _check_environment(missing_venv)


def test_check_environment_passes_when_interpreter_exists(tmp_path):
    python_path = _venv_python(tmp_path)
    python_path.parent.mkdir(parents=True)
    python_path.write_bytes(b"")
    _check_environment(tmp_path)  # should not raise


# ---------------------------------------------------------------------------
# run_repnet_on_video: monkeypatch subprocess.run and the checkpoint/venv
# checks so these stay fast and offline
# ---------------------------------------------------------------------------


def _patch_environment(monkeypatch, tmp_path):
    monkeypatch.setattr("hybrid.repnet_branch._check_environment", lambda venv_dir: None)
    monkeypatch.setattr("hybrid.repnet_branch.ensure_repnet_checkpoint", lambda models_dir: tmp_path / "ckpt")


def test_run_repnet_on_video_success(tmp_path, monkeypatch):
    _patch_environment(monkeypatch, tmp_path)

    def fake_run(cmd, cwd, capture_output, text, timeout):
        output_path = cmd[cmd.index("--output-json") + 1]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "cpm": 102.5,
                    "confidence": 0.87,
                    "pred_period_frames": 17.5,
                    "chosen_stride": 2,
                    "fps": 30.0,
                    "num_frames": 509,
                    "reason": None,
                },
                f,
            )
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr("hybrid.repnet_branch.subprocess.run", fake_run)

    config = HybridConfig()
    config.paths.cache_dir = tmp_path / "cache"
    result = run_repnet_on_video("dummy.mp4", config, "video_x")

    assert isinstance(result, RepNetResult)
    assert result.cpm == pytest.approx(102.5)
    assert result.confidence == pytest.approx(0.87)
    assert result.reason is None


def test_run_repnet_on_video_no_periodicity_found(tmp_path, monkeypatch):
    _patch_environment(monkeypatch, tmp_path)

    def fake_run(cmd, cwd, capture_output, text, timeout):
        output_path = cmd[cmd.index("--output-json") + 1]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "cpm": None,
                    "confidence": 0.05,
                    "pred_period_frames": 3.0,
                    "chosen_stride": 1,
                    "fps": 30.0,
                    "num_frames": 100,
                    "reason": "pred_score (0.050) below threshold (0.2)",
                },
                f,
            )
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr("hybrid.repnet_branch.subprocess.run", fake_run)

    config = HybridConfig()
    config.paths.cache_dir = tmp_path / "cache"
    result = run_repnet_on_video("dummy.mp4", config, "video_x")

    assert result.cpm is None
    assert result.reason is not None
    assert result.confidence == pytest.approx(0.05)


def test_run_repnet_on_video_subprocess_failure_raises(tmp_path, monkeypatch):
    _patch_environment(monkeypatch, tmp_path)

    def fake_run(cmd, cwd, capture_output, text, timeout):
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="run_repnet.py failed: RuntimeError: boom")

    monkeypatch.setattr("hybrid.repnet_branch.subprocess.run", fake_run)

    config = HybridConfig()
    config.paths.cache_dir = tmp_path / "cache"
    with pytest.raises(RepNetUnavailableError, match="boom"):
        run_repnet_on_video("dummy.mp4", config, "video_x")


def test_run_repnet_on_video_timeout_raises(tmp_path, monkeypatch):
    _patch_environment(monkeypatch, tmp_path)

    def fake_run(cmd, cwd, capture_output, text, timeout):
        raise subprocess.TimeoutExpired(cmd, timeout)

    monkeypatch.setattr("hybrid.repnet_branch.subprocess.run", fake_run)

    config = HybridConfig()
    config.paths.cache_dir = tmp_path / "cache"
    with pytest.raises(RepNetUnavailableError):
        run_repnet_on_video("dummy.mp4", config, "video_x")


# ---------------------------------------------------------------------------
# Caching wrapper
# ---------------------------------------------------------------------------


def _fake_repnet_result(video_id="video_x"):
    return RepNetResult(
        video_id=video_id,
        cpm=100.0,
        confidence=0.9,
        pred_period_frames=18.0,
        chosen_stride=1,
        fps=30.0,
        num_frames=509,
        reason=None,
    )


def test_cached_wrapper_avoids_recompute_on_hit(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        "hybrid.repnet_branch.run_repnet_on_video",
        lambda video_path, config, video_id: calls.append(video_id) or _fake_repnet_result(video_id),
    )
    config = HybridConfig()
    cache = CacheManager(tmp_path)

    run_repnet_on_video_cached("dummy.mp4", config, "video_x", cache)
    run_repnet_on_video_cached("dummy.mp4", config, "video_x", cache)

    assert calls == ["video_x"]


def test_cached_wrapper_recomputes_on_config_change(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        "hybrid.repnet_branch.run_repnet_on_video",
        lambda video_path, config, video_id: calls.append(video_id) or _fake_repnet_result(video_id),
    )
    cache = CacheManager(tmp_path)

    config_a = HybridConfig()
    run_repnet_on_video_cached("dummy.mp4", config_a, "video_x", cache)

    config_b = HybridConfig()
    config_b.repnet.threshold = 0.5
    run_repnet_on_video_cached("dummy.mp4", config_b, "video_x", cache)

    assert calls == ["video_x", "video_x"]
