# Architecture

One development video (`data/split/<id>_development.mp4`) flows through the branches below.
Every arrow is a real module boundary in `src/hybrid/`; caching (Phase 0.5) sits transparently
in front of every expensive branch (MediaPipe, CoTracker, ego-motion, optical flow, RepNet) so
re-running downstream phases during tuning never re-runs them unless their own config changed.

```mermaid
flowchart TD
    V[Development video\nVideoReader, Phase 1] --> MP[MediaPipe hand/wrist\nPhase 2]
    MP --> ROI[Dynamic CPR foreground ROI]
    ROI --> CT[CoTracker points\nPhase 3]
    V --> BG[Background features]
    BG --> EM[RANSAC affine ego-motion\nPhase 4]
    CT --> COR[Ego-motion corrected\ntrajectory, Phase 5]
    EM --> COR
    V --> FLOW[Farneback optical flow\nPhase 6]
    ROI --> FLOW
    COR --> WAVE[Motion waveform fusion\nPhase 7]
    FLOW --> WAVE
    WAVE --> FILT[Butterworth band-pass\nPhase 8]
    FILT --> CWT[CWT]
    FILT --> AC[Autocorrelation]
    FILT --> FFT[FFT]
    FILT --> PEAKS[Peak detection]
    V --> REPNET[RepNet\nisolated TF venv, Phase 10]
    CWT --> FUSION[Confidence-weighted +\ndisagreement-aware fusion\nPhase 12]
    AC --> FUSION
    FFT --> FUSION
    PEAKS --> FUSION
    REPNET --> FUSION
    FUSION --> FINAL[Final CPM +\nconfidence score]
    FINAL --> COMPARE[Compare to GT\nPhase 13, post-prediction only]
```

## Module map

| Stage | Module | Result type |
|---|---|---|
| Video I/O | `hybrid.video_io` | `Frame`, `VideoReader` |
| Dev-set / GT | `hybrid.dataset` | `DevVideo`, `GroundTruth` |
| MediaPipe | `hybrid.mediapipe_roi` | `MediaPipeVideoResult` |
| CoTracker | `hybrid.cotracker_tracker` | `CoTrackerVideoResult` |
| Ego-motion | `hybrid.ego_motion` | `EgoMotionVideoResult` |
| Corrected trajectory | `hybrid.corrected_trajectory` | `CorrectedTrackerResult` |
| Optical flow | `hybrid.optical_flow` | `OpticalFlowVideoResult` |
| Motion wave fusion | `hybrid.motion_wave` | `MotionWaveResult` |
| Butterworth filter | `hybrid.filters` | `FilterResult` |
| 4 classical estimators | `hybrid.estimators` | `AllEstimatesResult` |
| RepNet | `hybrid.repnet_branch` (+ `src/repnet_env/`, isolated venv) | `RepNetResult` |
| Confidence collection | `hybrid.confidence` | `BranchConfidences` |
| Final fusion | `hybrid.fusion` | `FusionResult` |
| Dev-set evaluation | `hybrid.evaluation` | `VideoEvaluationResult`, `DevelopmentEvaluationResult` |
| Ablation comparison | `hybrid.ablation` | `AblationSummary` |
| Diagnostics | `hybrid.diagnostics` | `runs/development/<id>/*` |
| Entrypoint | `hybrid.run_development` | CLI, Phase 18 guard |

## What's deliberately not shown

Confidence flows alongside every arrow above (not drawn, to keep the diagram readable) —
CoTracker visibility and ego-motion confidence feed Phase 7's `tracker_weight`, optical-flow
validity feeds `flow_weight`, and all five CPM-producing branches' own confidences feed
Phase 12's fusion weights directly. See `hybrid.confidence`'s module docstring for the
complete list of how each is computed, and ADR 0004 for why fusion doesn't stop at
confidence alone.
