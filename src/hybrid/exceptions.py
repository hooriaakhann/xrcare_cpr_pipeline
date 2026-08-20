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


class VideoReadError(HybridError):
    """Video file missing, unopenable, or yielded zero decodable frames."""


class TimestampError(HybridError):
    """Frame PTS could not be read, was malformed, or disagreed with the decoder."""


class GroundTruthMappingError(HybridError):
    """A development/test split filename could not be mapped unambiguously to
    its ground-truth row (see CLAUDE.md rule 3)."""


class HandNotDetectedError(HybridError):
    """MediaPipe found no hand for longer than the configured gap threshold."""


class TrackLostError(HybridError):
    """CoTracker's visible-point ratio dropped below threshold for too long."""


class EgoMotionUnreliableError(HybridError):
    """RANSAC affine ego-motion estimate failed its inlier-ratio/validity checks."""


class OpticalFlowUnstableError(HybridError):
    """Farneback optical flow diagnostics indicate an unstable/invalid estimate."""


class RepNetUnavailableError(HybridError):
    """The isolated RepNet subprocess/environment could not be reached or failed."""


class FilteringError(HybridError):
    """The Butterworth band-pass filter could not be applied (invalid band
    for the video's sample rate, too few samples, all-NaN signal, ...)."""


class EstimatorError(HybridError):
    """A classical rate estimator (CWT, autocorrelation, FFT, peaks) could
    not produce a result (no in-band energy, fewer than 2 peaks, ...)."""


class FusionError(HybridError):
    """No usable candidate estimate was available to fuse into a final CPM."""
