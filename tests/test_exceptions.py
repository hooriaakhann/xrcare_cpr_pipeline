from hybrid.exceptions import CacheError, ConfigError, HybridError


def test_all_typed_exceptions_are_hybrid_errors():
    assert issubclass(ConfigError, HybridError)
    assert issubclass(CacheError, HybridError)


def test_hybrid_error_is_a_real_exception():
    assert issubclass(HybridError, Exception)
