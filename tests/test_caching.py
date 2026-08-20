import numpy as np
import pytest

from hybrid.caching import CacheManager
from hybrid.exceptions import CacheError


def test_save_and_load_roundtrip(tmp_path):
    cache = CacheManager(tmp_path)
    key = cache.make_key("mediapipe", "video1", config_hash="abc123")
    payload = {"landmarks": np.arange(12).reshape(4, 3), "fps": 30.0}

    cache.save(key, payload)
    loaded = cache.load(key)

    assert np.array_equal(loaded["landmarks"], payload["landmarks"])
    assert loaded["fps"] == payload["fps"]


def test_exists_reflects_cache_state(tmp_path):
    cache = CacheManager(tmp_path)
    key = cache.make_key("cotracker", "video2")
    assert not cache.exists(key)
    cache.save(key, [1, 2, 3])
    assert cache.exists(key)


def test_load_missing_key_raises_cache_error(tmp_path):
    cache = CacheManager(tmp_path)
    with pytest.raises(CacheError):
        cache.load("nonexistent-key")


def test_different_config_hash_gives_different_key(tmp_path):
    cache = CacheManager(tmp_path)
    key_a = cache.make_key("repnet", "video3", config_hash="hash-a")
    key_b = cache.make_key("repnet", "video3", config_hash="hash-b")
    assert key_a != key_b


def test_clear_removes_entry(tmp_path):
    cache = CacheManager(tmp_path)
    key = cache.make_key("repnet", "video4")
    cache.save(key, {"count": 30})
    assert cache.exists(key)
    cache.clear(key)
    assert not cache.exists(key)
