import logging

from hybrid.logging_config import get_logger, setup_logging


def test_setup_logging_is_idempotent(tmp_path):
    root = logging.getLogger()
    handlers_before = list(root.handlers)

    setup_logging(level="DEBUG", log_dir=tmp_path)
    handlers_after_first = list(root.handlers)
    setup_logging(level="DEBUG", log_dir=tmp_path)
    handlers_after_second = list(root.handlers)

    assert len(handlers_after_second) == len(handlers_after_first)
    assert len(handlers_after_first) >= len(handlers_before)


def test_get_logger_returns_named_logger():
    logger = get_logger("hybrid.some_module")
    assert logger.name == "hybrid.some_module"
