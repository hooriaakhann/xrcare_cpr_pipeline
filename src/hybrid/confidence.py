"""Confidence Calculation (Phase 11).

Every branch already computes its own confidence inline where it's derived
(Phases 2-10) -- this module doesn't recompute anything. It collects those
already-normalized [0,1] values into one place for Phase 12's fusion, and
documents how each one is derived (the authoritative computation lives in
each branch's own module; this is the consolidated summary):

- **MediaPipe**: fraction of video frames with a successful hand detection
  (`MediaPipeVideoResult.detection_rate`, Phase 2).
- **CoTracker**: mean, across frames, of the fraction of tracked points
  currently visible (`CoTrackerVideoResult.valid_ratio_per_frame`, Phase 3).
- **Ego-motion**: mean, across frames, of the per-frame RANSAC inlier ratio
  (0.0 wherever the transform failed validity checks --
  `EgoMotionVideoResult.frames[i].confidence`, Phase 4).
- **Optical flow**: fraction of frames whose flow was not flagged unstable
  (`OpticalFlowVideoResult.frames[i].valid`, Phase 6).
- **CWT**: dominant scale's share of total in-band wavelet power
  (`CwtEstimate.confidence`, Phase 9) -- structurally lower than the other
  three classical estimators (Morlet's broader spectral peaks spread energy
  across more scales than FFT's discrete bins); see PROGRESS.md Phase 9.
- **Autocorrelation**: normalized autocorrelation value at the dominant lag
  (`AutocorrelationEstimate.confidence`, Phase 9).
- **FFT**: squashed spectral SNR, `snr/(snr+1)` (`FftEstimate.confidence`,
  Phase 9).
- **Peaks**: `1 - coefficient_of_variation` of inter-peak intervals
  (`PeakEstimate.confidence`, Phase 9).
- **RepNet**: the model's own combined period/periodicity score
  (`RepNetResult.confidence`, Phase 10) -- 0.0 when it found no periodicity
  it trusted.

MediaPipe/CoTracker/ego-motion/optical-flow confidences don't feed Phase
12's fusion directly (they already did their weighting work upstream, in
Phase 7's `tracker_weight`/`flow_weight`) -- they're collected here anyway
so the full per-video confidence picture is in one place for Phase 16's
diagnostics and Phase 13's evaluation tables.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hybrid.cotracker_tracker import CoTrackerVideoResult
from hybrid.ego_motion import EgoMotionVideoResult
from hybrid.estimators import AllEstimatesResult
from hybrid.exceptions import HybridError
from hybrid.mediapipe_roi import MediaPipeVideoResult
from hybrid.optical_flow import OpticalFlowVideoResult
from hybrid.repnet_branch import RepNetResult


class ConfidenceRangeError(HybridError):
    """A branch reported a confidence outside [0.0, 1.0] -- a bug in that
    branch's own normalization, caught here before it can silently skew
    Phase 12's fusion weights."""


@dataclass(frozen=True)
class BranchConfidences:
    video_id: str
    mediapipe: float
    cotracker: float
    ego_motion: float
    optical_flow: float
    cwt: float
    autocorrelation: float
    fft: float
    peaks: float
    repnet: float

    def as_dict(self) -> dict[str, float]:
        return {
            "mediapipe": self.mediapipe,
            "cotracker": self.cotracker,
            "ego_motion": self.ego_motion,
            "optical_flow": self.optical_flow,
            "cwt": self.cwt,
            "autocorrelation": self.autocorrelation,
            "fft": self.fft,
            "peaks": self.peaks,
            "repnet": self.repnet,
        }


def _validate_unit_interval(name: str, value: float, video_id: str) -> float:
    if not (0.0 <= value <= 1.0):
        raise ConfidenceRangeError(f"{video_id}: {name} confidence {value} is outside [0.0, 1.0]")
    return value


def collect_confidences(
    mediapipe_result: MediaPipeVideoResult,
    cotracker_result: CoTrackerVideoResult,
    ego_motion_result: EgoMotionVideoResult,
    optical_flow_result: OpticalFlowVideoResult,
    estimates: AllEstimatesResult,
    repnet_result: RepNetResult,
    video_id: str,
) -> BranchConfidences:
    ego_motion_confidence = float(np.mean([f.confidence for f in ego_motion_result.frames]))
    optical_flow_confidence = float(np.mean([f.valid for f in optical_flow_result.frames]))

    values = {
        "mediapipe": mediapipe_result.detection_rate,
        "cotracker": float(np.mean(cotracker_result.valid_ratio_per_frame)),
        "ego_motion": ego_motion_confidence,
        "optical_flow": optical_flow_confidence,
        "cwt": estimates.cwt.confidence,
        "autocorrelation": estimates.autocorrelation.confidence,
        "fft": estimates.fft.confidence,
        "peaks": estimates.peaks.confidence,
        "repnet": repnet_result.confidence,
    }
    for name, value in values.items():
        _validate_unit_interval(name, value, video_id)

    return BranchConfidences(video_id=video_id, **values)
