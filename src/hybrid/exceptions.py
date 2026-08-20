"""Typed exceptions for the hybrid pipeline.

Every module raises one of these (never returns None on failure); the
orchestration layer is responsible for catching, logging, and deciding
whether to abort or skip. Phase-specific modules (branches, fusion, ...)
should subclass HybridError rather than raising bare Exception/ValueError.
"""


class HybridError(Exception):
    """Base class for all pipeline errors."""


class ConfigError(HybridError):
    """Config file missing, malformed, or fails schema validation."""


class CacheError(HybridError):
    """Cache read/write failure (corrupt entry, unwritable cache dir, ...)."""
