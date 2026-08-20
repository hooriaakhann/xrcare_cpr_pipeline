"""Shared logging setup. No module should call the bare `print`/`logging`
functions directly — call `setup_logging()` once at the entry point of a
script, then `get_logger(__name__)` everywhere else.
"""

import logging
import sys
from pathlib import Path

_LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_configured = False


def setup_logging(level: str = "INFO", log_dir: Path | None = None, log_filename: str = "pipeline.log") -> None:
    """Configure the root logger. Safe to call multiple times (no-op after the first).

    Args:
        level: standard logging level name (DEBUG/INFO/WARNING/ERROR).
        log_dir: if given, also writes logs to `log_dir/log_filename` (dir is created).
        log_filename: file name used when log_dir is given.
    """
    global _configured
    if _configured:
        return

    root = logging.getLogger()
    root.setLevel(level.upper())

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / log_filename, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger. Call `setup_logging()` first to attach handlers."""
    return logging.getLogger(name)
