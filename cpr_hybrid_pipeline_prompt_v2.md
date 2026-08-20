# CPR Hybrid Compression-Rate Estimation — Development-Phase Implementation Prompt (v2)

> **v2 changelog:** adds project scaffolding (packaging, config schema, logging, exceptions, caching), a testing/CI phase, an experiment-tracking ledger, statistical rigor for evaluation, a documentation phase (README, ADRs, method card), and a session strategy for running Phases 1–21 as one continuous `/goal`-driven Claude Code session with a commit-and-push after every phase, rather than one session per phase. The core algorithmic phases are unchanged in intent — they now reference the shared infrastructure built in Phase 0.5.

---

## Dataset

I have finished preprocessing and splitting my CPR dataset.

I now want to implement the **hybrid CPR compression-rate estimation pipeline** on the **development videos only**.

My split videos are stored in:

`data/split/`

Development videos have filenames ending with:

`_development.mp4`

Test videos have filenames ending with:

`_test.mp4`

### Critical rule

For this phase:

**Use ONLY `_development.mp4` videos.**

Do not run, inspect for performance, tune on, benchmark, or use ground-truth results from `_test.mp4` videos.

The test videos must remain completely held out until the hybrid pipeline has been finalized and frozen.

I have already computed clip-level ground-truth CPM in:

`data/metadata/ground_truth_summary.csv`

Ground truth is **count-based active-segment CPM**:

`GT CPM = (30 × 60) / CPR-active duration`

Each usable video contains exactly 30 compressions.

If my renamed split filenames do not directly match the filenames in `ground_truth_summary.csv`, inspect the naming carefully and create a safe filename mapping. Do not silently assign the wrong GT to a video. If mapping is ambiguous, stop and tell me.

---

## Overall Goal

Build a hybrid CPR-rate estimation system combining:

1. **MediaPipe**
2. **CoTracker**
3. **RANSAC affine ego-motion compensation**
4. **Farneback optical flow**
5. **Motion waveform generation**
6. **Butterworth filtering**
7. **CWT**
8. **Autocorrelation**
9. **FFT**
10. **Peak detection**
11. **RepNet**
12. **Confidence-weighted fusion**

The final output for every development video should be:

- GT CPM
- MediaPipe-related motion estimate
- CoTracker motion estimate
- Optical-flow estimate
- RepNet estimate
- CWT CPM
- Autocorrelation CPM
- FFT CPM
- Peak-detection CPM
- Final hybrid CPM
- Confidence score
- Absolute error against GT
- relevant failure/quality indicators

Do not train MediaPipe, CoTracker, or RepNet from scratch. Use pretrained models/checkpoints.

The development dataset is used for **implementation, calibration, threshold tuning, and fusion tuning**, not neural-network training.

Beyond the CPM numbers, the finished project should read as an **engineered system** — tested, configured, logged, cached, and documented — not a collection of ad hoc scripts.

---

## Engineering Standards (apply to every phase below)

These are non-negotiable conventions the rest of this document assumes:

- **Config:** no hardcoded parameters; everything goes through a schema-validated config object (Phase 0.5).
- **Logging:** the shared `logging` setup only — no bare `print`; every run writes a console stream and a per-run log file.
- **Errors:** typed exceptions per failure mode, raised and logged at the orchestration level — never a silently-returned `None`.
- **Testing:** every estimator gets a synthetic-signal unit test before its output is trusted on real video; the full pipeline gets at least one integration test.
- **Caching:** expensive branch outputs (MediaPipe, CoTracker, RepNet) are cached per-video so tuning (Phase 15) doesn't re-run them for every parameter sweep.
- **Reproducibility:** git commit hash, config hash, and any RNG/RANSAC seed are recorded with every run.
- **Experiment ledger:** every development run is appended to a ledger with its config and metrics.

---

## Session & Continuity Strategy (Claude Code workflow)

Phase 0.5 is complete, verified, committed, and GitHub is connected (`hooriaakhan6-prog`). From Phase 1 onward this runs as **one continuous Claude Code session**, driven by `/goal`, rather than one session per phase:

- **`/goal`** keeps Claude Code working turn after turn until a completion condition is met, instead of stopping and waiting for "continue" after every phase. It's set once, at the start of this run, with the full Phase 1–21 condition (see the kickoff prompt that accompanies this document).
- **Commit *and push* at the end of every phase**, not once at the end. Each phase's commit lands on `origin` before moving to the next phase. This keeps GitHub as a live, phase-by-phase record — genuinely useful for a resume project, since the commit history becomes evidence of the engineering process — and means a dropped connection or crash loses at most one phase, not the whole run.
- **`CLAUDE.md` and `PROGRESS.md` stay the safety net even inside one long session.** Update `PROGRESS.md` at the end of every phase exactly as before: phase, decisions made, config values touched, "Next phase: Phase N+1." If the single session hits real trouble — degrading quality, getting stuck, context problems despite auto-compaction — the recovery path is unchanged: `/clear` (or a new `claude` invocation) and resume from that line. It's just a fallback now, not the default plan.
- **Turn on `auto` permission mode** (or `acceptEdits`) alongside `/goal` — `/goal` only removes the "wait for my next message" step; without hands-off permissions you'll still be approving every file write and command individually.
- **Bound the goal.** `/goal` has no built-in token budget, so the completion condition includes a turn cap. Check in any time with a bare `/goal` to see turns used, cost, and its last reasoning.
- **Delegate heavy read-only work to subagents** so it doesn't eat the main session's context — still worth doing in a single continuous session: Phase 20's pass over every `runs/development/<video_name>/` folder to draft the README/method card is a good candidate.
- **The real safety rails for an unattended run are the Phase 19 test suite and the exceptions list below** — a green `make test` per phase plus explicit "stop and ask" conditions, not the session boundary itself.

### Exceptions — stop and ask, even under `/goal`

- Anything that would touch a `_test.mp4` file
- Any ambiguous ground-truth mapping or dataset decision
- An environment/dependency conflict the Phase 0.5 virtualenv + subprocess isolation doesn't already cover
- Phase 15 parameter tuning: run one defensible-default pass, report dev MAE/RMSE, then stop and ask before doing a larger sweep
- Anything Phase 0's or Phase 18's rules already say to stop and tell the user about
- Any git push that fails (auth expired, conflict, etc.) — report it, don't force-push

---

## PHASE 0 — Inspect Before Coding

Before implementing anything:

1. Inspect the entire repository.
2. Inspect `data/split/`.
3. List all `_development.mp4` files.
4. List all `_test.mp4` files but do not run them through the algorithms.
5. Inspect:
   - `ground_truth_summary.csv`
   - preprocessing metadata
   - timestamp sidecars
6. Determine how renamed development filenames map to GT filenames.
7. Inspect current Python environment and installed packages.
8. Check whether:
   - MediaPipe is installed
   - PyTorch is installed
   - CoTracker is available
   - TensorFlow is available
   - RepNet code/checkpoint is available
   - SciPy / PyWavelets / OpenCV are available
9. Check for an existing git repository, `pyproject.toml`/`requirements.txt`, and Python version.
10. Check GPU/CUDA availability — CoTracker and RepNet both benefit from it.

If CoTracker and RepNet require conflicting dependencies, propose isolated environments rather than breaking the main environment.

Before writing code, show me:

- development videos found
- test videos found
- GT mapping
- existing dependencies
- missing dependencies
- GPU/CUDA availability
- proposed file structure
- proposed implementation sequence

Then proceed only after the architecture is clear.

---

## PHASE 0.5 — Project Scaffolding & Engineering Foundations

Set this up before any pipeline logic is written:

1. Initialize git (if not already) with a `.gitignore` covering `data/`, `runs/`, cache directories, `__pycache__`, and virtual environments. Do not commit raw video files or large caches.
2. `pyproject.toml` (poetry or uv) with pinned dependency versions. Separate the main pipeline deps from RepNet's, since they may need different PyTorch/TensorFlow versions.
3. A `Makefile` with at least: `make setup`, `make lint`, `make test`, `make run-dev VIDEO=...`, `make tune`.
4. Pre-commit hooks: `black`/`ruff` (or `flake8`), `isort`, `mypy` (can start lenient).
5. `config/default.yaml` + `src/hybrid/config.py` — a schema-validated config (e.g. Pydantic) covering every tunable parameter listed in Phase 15. The pipeline should load this once and pass it explicitly, not rely on hidden globals. Example skeleton:

```yaml
# config/default.yaml
mediapipe:
  min_detection_confidence: 0.5
  min_tracking_confidence: 0.5
cotracker:
  num_points: 40
  visibility_threshold: 0.6
  reinit_visibility_threshold: 0.4
ego_motion:
  ransac_reproj_threshold: 3.0
  min_inlier_ratio: 0.5
  seed: 42
optical_flow:
  pyr_scale: 0.5
  levels: 3
  winsize: 15
filtering:
  butterworth_low_hz: 1.0
  butterworth_high_hz: 3.0
estimators:
  cwt_freq_range_hz: [1.0, 3.0]
  fft_freq_range_hz: [1.0, 3.0]
  autocorr_lag_range_sec: [0.3, 1.0]
  peak_min_distance_sec: 0.35
  peak_prominence: 0.3
fusion:
  min_confidence_floor: 0.05
```

6. `src/hybrid/logging_config.py` — sets up a logger writing to console and to `runs/development/<video_name>/run.log`.
7. `src/hybrid/exceptions.py` — typed failure modes caught and logged at the orchestration level, e.g.:

```python
class HandNotDetectedError(Exception): ...
class TrackLostError(Exception): ...
class EgoMotionUnreliableError(Exception): ...
class OpticalFlowUnstableError(Exception): ...
class RepNetUnavailableError(Exception): ...
```

8. `src/hybrid/caching.py` — a per-video, per-branch cache keyed by video filename + a hash of the relevant sub-config, so tuning downstream parameters (Butterworth cutoff, fusion weights, peak prominence, etc.) doesn't require re-running MediaPipe/CoTracker/RepNet. The cache can reuse the same CSV/JSON artifacts already planned in Phase 16 — this layer just decides when to reuse vs. recompute them.
9. `src/hybrid/experiment_ledger.py` — appends one row per run/tuning iteration to `runs/experiment_ledger.jsonl` (or SQLite):

```json
{"timestamp": "2026-08-20T10:15:00Z", "git_commit": "a1b2c3d", "config_hash": "9f2e...", "changed_params": ["filtering.butterworth_high_hz"], "dev_mae": 3.8, "dev_rmse": 5.1}
```

10. `.github/workflows/ci.yml` skeleton — runs lint + unit tests (not full video runs) on every push. It's fine for this to be incomplete until Phase 19; just get the directory in place now.
11. Decide and document (as an ADR in Phase 20) whether CoTracker/PyTorch vs. RepNet/TensorFlow isolation will be **separate virtualenvs + subprocess** (simpler, matches the Phase 10 requirement) or **separate Docker containers** (more portfolio-visible, more setup cost). Default to virtualenv + subprocess unless there's time for Docker later — see Priority Tiers near the end of this document.
12. `CLAUDE.md` and `PROGRESS.md` at the repo root — see "Session & Continuity Strategy" above. These are the audit trail and the recovery path if a long continuous session ever needs to be resumed.

Show me the resulting skeleton project tree before moving to Phase 1.

---

## PHASE 1 — Shared Video / Timestamp Handling

Create one shared video-reading layer.

Requirements:

- Process the full-frame development video.
- Preserve real video timing.
- Do not assume every frame interval is exactly `1/FPS`.
- Use actual timestamps/PTS where possible because the source videos are VFR.
- Keep frame index + timestamp together throughout the pipeline.
- Use the Phase 0.5 logger for all diagnostic output, and raise the appropriate typed exception on unreadable frames or corrupt timestamps rather than returning `None`.

Each output signal should therefore be associated with:

`frame_index, timestamp_sec`

This will be important for later event-level analysis.

---

## PHASE 2 — MediaPipe Hand/Wrist Localization

Implement MediaPipe as the semantic localization branch.

Its job is NOT to calculate the final CPM by itself.

Use MediaPipe to detect:

- hands
- wrist landmarks
- hand bounding region
- visibility/confidence where available

Use it to generate a **dynamic CPR foreground region**.

The dynamic ROI should contain approximately:

- hand
- wrist
- part of forearm
- surrounding CPR motion area

Do not permanently crop the input video. Keep the full frame available for ego-motion estimation.

Save per-frame diagnostic information:

- hand detected yes/no
- wrist coordinates
- bounding box
- confidence
- missing detection flag

Also calculate:

- percentage of frames with successful hand detection
- longest detection gap

MediaPipe should also provide initialization locations for CoTracker.

Cache per-frame diagnostic output via the Phase 0.5 caching layer, keyed by video + MediaPipe config, so re-running downstream phases during tuning does not require re-running MediaPipe. Log (don't silently swallow) any detection gap that exceeds a configurable threshold.

---

## PHASE 3 — CoTracker Multi-Point Tracking

Use a pretrained CoTracker model.

MediaPipe should initialize multiple tracking points inside/around the hand/wrist/forearm region.

Do NOT rely on one wrist point.

Initialize a reasonable grid/set of points such as:

- hand points
- wrist points
- forearm points

Make the number of points configurable.

CoTracker should output:

- x trajectory
- y trajectory
- visibility/confidence
- valid/invalid state

for every tracked point over time.

Create a robust aggregate vertical trajectory from the visible points.

Possible strategy:

`median Y displacement of valid foreground points`

rather than mean, to reduce outlier effects.

Generate:

`tracker_motion(t)`

Also record:

- number of visible points per frame
- percentage of valid tracks
- track-loss periods
- reinitialization events

MediaPipe may periodically reinitialize CoTracker if tracking quality becomes poor. Do not hard-code a reinitialization frequency without making it configurable.

Cache the raw trajectory output. Raise `TrackLostError` (logged, not silent) when the visible-point ratio drops below a configurable threshold for longer than a configurable window.

---

## PHASE 4 — RANSAC Affine Ego-Motion Compensation

This is a critical part of the hybrid.

Estimate motion caused by the smart-glasses wearer/head separately from CPR motion.

Use stable **background features**.

Exclude the MediaPipe/CoTracker CPR foreground region when selecting background features so the compression motion itself does not influence camera-motion estimation.

A possible implementation:

1. Detect stable background features.
2. Track them frame-to-frame using suitable OpenCV feature tracking.
3. Match previous/current points.
4. Estimate a robust 2D affine transform using RANSAC.
5. Reject outliers.
6. Store affine transform and RANSAC inlier ratio.

Use an OpenCV affine estimator such as an appropriate RANSAC-based `estimateAffinePartial2D` or `estimateAffine2D`, and justify which one is selected (this becomes an ADR in Phase 20).

The transform should model camera/head:

- translation
- rotation
- small scale changes

Create diagnostics:

- number of background features
- number of matched features
- number of RANSAC inliers
- inlier ratio
- estimated translation
- estimated rotation
- transform-valid flag

If the transform is unreliable, do not blindly apply it — raise/flag `EgoMotionUnreliableError` and drive `ego_motion_confidence(t)` toward zero for that span rather than applying an untrustworthy transform.

Set and log an **explicit random seed** for RANSAC so runs are reproducible; record the seed in the run's `summary.json` and in the experiment ledger.

Produce an `ego_motion_confidence(t)`.

---

## PHASE 5 — Ego-Motion Corrected CoTracker Trajectories

Use the estimated affine transform to remove camera motion from the CoTracker foreground trajectories.

The goal is to move from:

`observed hand trajectory = CPR movement + camera movement`

toward:

`corrected hand trajectory ≈ CPR movement`

Generate:

`corrected_tracker_motion(t)`

Save both:

- raw tracker signal
- ego-motion-corrected tracker signal

so they can later be compared.

---

## PHASE 6 — Farneback Optical Flow

Implement dense Farneback optical flow.

Do not simply calculate whole-frame mean optical flow.

Use:

- the dynamic CPR foreground region from MediaPipe/CoTracker
- background information for camera-motion context

Preferably calculate optical flow after or together with affine camera compensation.

Extract the residual **vertical CPR motion**.

Generate:

`flow_motion_y(t)`

Store diagnostics such as:

- foreground vertical flow
- background/global motion
- corrected/residual vertical flow
- flow magnitude
- invalid/unstable flow flags

Keep the Farneback parameters configurable (config file, not hardcoded). Do not tune them on test videos.

Cache the flow signal per-video/per-config. Raise `OpticalFlowUnstableError` (logged) when the flow diagnostics indicate instability rather than silently passing bad data downstream.

---

## PHASE 7 — Motion Waveform Generation

At this point we should have at least two useful CPR motion signals:

1. `corrected_tracker_motion(t)`
2. `flow_motion_y(t)`

Normalize them robustly. Do not simply average raw values because their scales are different.

Possible methods:

- median centering
- robust standardization
- MAD-based scaling
- z-score only if appropriate

Then create a fused motion waveform:

`motion_wave(t)`

The weighting should depend on signal quality. For example:

- strong CoTracker visibility → higher tracker weight
- poor tracker visibility → lower tracker weight
- strong ego-motion/RANSAC failure → reduce affected motion confidence
- unstable optical flow → reduce flow weight

Keep raw signals, normalized signals, the fused signal, and per-frame weights for later debugging.

---

## PHASE 8 — Butterworth Filtering

Filter the fused motion waveform using a Butterworth band-pass filter.

The frequency band must be configurable, validated by the config schema (Phase 0.5) so `low < high` and both stay within the Nyquist limit implied by the video's effective sampling rate.

Do not blindly copy parameters from another project. Choose a physiologically/plausibly useful CPR range and explain the initial choice (this becomes an ADR in Phase 20), but keep it easy to tune on development videos.

Use appropriate zero-phase filtering when offline processing allows it.

Save unfiltered and filtered waveforms. Do not distort timing unnecessarily.

---

## PHASE 9 — Four Classical Rate Estimators

Use the filtered motion waveform to calculate CPM independently using FOUR methods.

**Testing requirement:** before trusting any of these four estimators on real video, write a unit test per estimator against a synthetic waveform of known frequency (e.g. a 1.67 Hz / 100 CPM sine wave, plus a noisy version and a version with a couple of dropped cycles) and assert the recovered CPM is within a small tolerance (e.g. ±2 CPM) of ground truth. These tests live in `tests/unit/test_estimators_synthetic.py` and must pass in CI (Phase 19) before Phase 13 development evaluation is trusted.

### Estimator 1 — CWT

Use Continuous Wavelet Transform to find the dominant periodic compression frequency.

Output: dominant frequency, CPM, spectral/wavelet confidence.

### Estimator 2 — Autocorrelation

Find the dominant temporal lag corresponding to repeated compressions.

Output: dominant lag, CPM, autocorrelation peak strength/confidence.

### Estimator 3 — FFT / Power Spectrum

Compute the frequency-domain spectrum within the allowed CPR band.

Output: dominant frequency, CPM, spectral SNR / peak ratio.

### Estimator 4 — Peak Detection

Detect individual compression cycles from the filtered waveform.

Use configurable peak prominence, minimum peak distance, and minimum height if necessary.

Output: detected peak timestamps, number of peaks, median inter-peak interval, CPM, peak regularity/confidence.

Do not assume that 30 peaks must be detected. The algorithm must produce its own result independently.

---

## PHASE 10 — RepNet Branch

Implement RepNet as a separate pretrained learned repetition branch.

Use the official/pretrained implementation/checkpoint where possible. Do not train RepNet on these development videos.

Test RepNet on an input that is appropriate for this project. Prefer to evaluate at least:

**Variant A** — Full CPR-active processed video.

**Variant B** — Stabilized / CPR-focused input if it can be created reliably without destroying timing.

Do not permanently alter the canonical dataset. RepNet preprocessing should happen dynamically inside the pipeline. Preserve temporal timing correctly when resizing frames for RepNet.

Extract: predicted repetition period, periodicity probability, predicted repetitions/count if available, RepNet CPM, RepNet confidence.

If the RepNet implementation requires a separate TensorFlow environment, keep it isolated and invoke it through a subprocess (per the Phase 0.5 decision) rather than forcing incompatible dependencies into the primary environment. Cache RepNet output per-video/per-config, since it's the most expensive branch.

---

## PHASE 11 — Confidence Calculation

Each branch/estimator should produce both a `CPM estimate` and a `confidence`.

Possible confidence cues:

**CoTracker** — visible point ratio, trajectory consistency, number of valid points.

**Ego motion** — RANSAC inlier ratio, transform stability.

**Optical flow** — motion consistency, residual foreground signal strength, global motion instability.

**CWT** — dominant wavelet energy / competing energy.

**FFT** — spectral SNR, dominant peak separation.

**Autocorrelation** — dominant autocorrelation peak strength.

**Peaks** — prominence, interval regularity, number of plausible peaks.

**RepNet** — periodicity probability, model confidence, period stability.

Normalize confidence values to a consistent range such as `0.0 – 1.0`. Document how each confidence score is calculated.

---

## PHASE 12 — Final Hybrid Fusion

Do NOT simply average every CPM estimate. Create a confidence-weighted fusion.

At minimum, candidate estimates include: CWT CPM, Autocorrelation CPM, FFT CPM, Peak CPM, RepNet CPM.

The classical estimators are based on the tracker + optical-flow fused motion signal, while RepNet provides an independent learned periodicity estimate.

A possible initial formulation:

`final_cpm = Σ(confidence_i × cpm_i) / Σ(confidence_i)`

but improve this if needed (justify the final formula as an ADR in Phase 20).

Also implement basic disagreement handling. For example, if CWT ≈ 100, autocorrelation ≈ 101, FFT ≈ 100, peaks ≈ 102, and RepNet ≈ 65 with low RepNet confidence, the 65 CPM result should not pull the final value strongly downward.

Store each estimate, each confidence, each fusion weight, final CPM, overall confidence, and estimator disagreement/spread.

---

## PHASE 13 — Development-Set Evaluation

After all branches work, run the complete system on **development videos only**.

Compare predictions to `ground_truth_summary.csv`.

For each video report: filename, person if available, GT CPM, CWT CPM, autocorrelation CPM, FFT CPM, peak CPM, RepNet CPM, final hybrid CPM, signed error, absolute error, confidence, runtime, FPS/processing speed if practical.

Calculate development MAE, RMSE, mean signed error, median absolute error, maximum absolute error.

Point estimates of MAE/RMSE alone are not very informative on a small dev set — Phase 22-equivalent statistical checks (see Phase 19/20 testing note and the ledger) add bootstrap confidence intervals and a significance test on top of these raw numbers before any "the hybrid helps" claim is made.

Remember: development metrics are for tuning/debugging. They are NOT the final generalization result.

---

## PHASE 14 — Ablation Comparison

I need to understand whether the hybrid actually helps. Generate development-set results for useful ablations such as:

**A** — MediaPipe wrist signal only
**B** — CoTracker trajectory only
**C** — CoTracker + affine compensation
**D** — Farneback optical flow
**E** — Affine-compensated Farneback
**F** — Tracker + flow fused motion
**G** — RepNet alone
**H** — Full hybrid

Do not overcomplicate this if a branch is not directly comparable, but save enough results to determine which components add value.

Produce a table like:

| Method | Development MAE | RMSE | Mean runtime | Notes |
|---|---:|---:|---:|---|
| MediaPipe | | | | |
| CoTracker | | | | |
| Affine + CoTracker | | | | |
| Flow | | | | |
| Affine + Flow | | | | |
| Motion fusion | | | | |
| RepNet | | | | |
| Full hybrid | | | | |

Where the dev set is small, also report a **bootstrap 95% CI** on MAE per row and a **paired Wilcoxon signed-rank test** on per-video absolute error comparing the full hybrid (H) against the best single-branch ablation, to back up (or temper) the claim that fusion helps.

---

## PHASE 15 — Parameter Tuning

Only development videos can be used for tuning.

Configurable parameters should live in the central `config/default.yaml` (Phase 0.5) rather than being scattered through scripts. Examples:

- MediaPipe confidence thresholds
- CoTracker point count
- CoTracker visibility threshold
- CoTracker reinitialization threshold
- RANSAC reprojection threshold
- minimum RANSAC inlier ratio
- Farneback parameters
- Butterworth lower/upper cutoff
- CWT scales/frequency range
- FFT frequency band
- autocorrelation lag range
- peak prominence
- minimum peak distance
- confidence thresholds
- fusion weights

Start from defensible defaults. Do not perform huge blind searches. Tune systematically and record every configuration/result.

Every tuning iteration must be appended to the experiment ledger (Phase 0.5) — config hash, which parameter(s) changed, resulting dev MAE/RMSE. This is what lets you later state "40+ tuning iterations, chose config X because Y" with evidence rather than memory.

Do NOT use test videos for selecting parameters.

---

## PHASE 16 — Save Diagnostics

For every development video, create a run folder such as:

`runs/development/<video_name>/`

Save useful artifacts, for example: `summary.json`, `signals.csv`, `cotracker_tracks.csv`, `ego_motion.csv`, `optical_flow_signal.csv`, `repnet_result.json`, `estimator_results.json`, `peaks.csv`, `run.log`.

Also generate diagnostic plots if practical: raw tracker motion, corrected tracker motion, optical-flow motion, fused motion waveform, Butterworth-filtered waveform, detected peaks, FFT spectrum, CWT-derived rate/confidence, per-estimator CPM comparison.

Keep outputs reproducible.

---

## PHASE 17 — Suggested Code Structure

Keep the implementation modular. A reasonable structure could be:

```text
.
├── CLAUDE.md                      # auto-loaded every Claude Code session
├── PROGRESS.md                    # per-phase hand-off log between sessions
├── pyproject.toml
├── Makefile
├── README.md
├── .pre-commit-config.yaml
├── .github/
│   └── workflows/
│       └── ci.yml
├── config/
│   └── default.yaml
├── docker/                        # optional — see Phase 0.5 ADR
│   ├── main.Dockerfile            # MediaPipe / CoTracker / PyTorch env
│   └── repnet.Dockerfile          # RepNet / TensorFlow env (isolated)
├── docs/
│   ├── architecture.md            # Mermaid diagram + system overview
│   ├── method_card.md
│   └── adr/
│       ├── 0001-affine-estimator-choice.md
│       ├── 0002-butterworth-band-selection.md
│       ├── 0003-env-isolation-strategy.md
│       └── 0004-fusion-formula.md
├── src/
│   └── hybrid/
│       ├── __init__.py
│       ├── config.py              # schema, loads config/default.yaml
│       ├── logging_config.py
│       ├── exceptions.py
│       ├── caching.py
│       ├── experiment_ledger.py
│       ├── video_io.py
│       ├── mediapipe_roi.py
│       ├── cotracker_tracker.py
│       ├── ego_motion.py
│       ├── optical_flow.py
│       ├── motion_wave.py
│       ├── filters.py
│       ├── estimators.py
│       ├── repnet_branch.py
│       ├── confidence.py
│       ├── fusion.py
│       ├── evaluation.py
│       ├── cli.py                 # optional — typer/click entrypoint
│       └── run_development.py
├── tests/
│   ├── unit/
│   │   ├── test_estimators_synthetic.py
│   │   ├── test_config.py
│   │   ├── test_caching.py
│   │   ├── test_gt_mapping.py
│   │   └── test_confidence.py
│   └── integration/
│       └── test_pipeline_end_to_end.py
├── runs/
│   └── development/<video_name>/
└── data/
    ├── split/
    └── metadata/
```

You may adjust this if there is a cleaner architecture, but do not put the entire pipeline into one giant Python file.

---

## PHASE 18 — Test-Safety Rule

Implement an explicit guard so that `run_development.py` only accepts files matching `*_development.mp4`.

If a `_test.mp4` file is accidentally passed during development/tuning, refuse to process it unless a separate future test script is explicitly used. Do not create/run that final test benchmark yet. The held-out test person must remain unseen.

---

## PHASE 19 — Testing & CI

Beyond the per-estimator synthetic tests from Phase 9:

- Unit tests for the GT filename mapping logic (Phase 0, including ambiguous-mapping detection), config schema validation (bad YAML should fail loudly), the caching layer (hit/miss/invalidation-on-config-change), and the confidence-normalization logic (Phase 11).
- One integration test that runs the full pipeline end-to-end on either (a) a short synthetic video with a rendered periodic motion pattern, or (b) a short trimmed real development clip checked into `tests/fixtures/` (only if small enough to commit; otherwise mark it `@pytest.mark.slow` and skip in CI).
- `pytest --cov=src/hybrid` for a coverage report — not aiming for 100%, but the estimator/fusion/confidence logic should be well covered.
- Wire the CI skeleton from Phase 0.5 to actually run `make lint` and `make test` (excluding slow/video-dependent tests) on every push, and make it green before calling development work "done."

---

## PHASE 20 — Documentation

- `README.md`: problem statement, the architecture diagram from "Final Expected Pipeline" below (also rendered as Mermaid in `docs/architecture.md`), setup/run instructions (`make setup`, `make run-dev VIDEO=...`), and a sample per-video output table.
- Short ADRs (`docs/adr/000N-*.md`, one paragraph each) for the decisions already flagged for justification: `estimateAffinePartial2D` vs `estimateAffine2D` (Phase 4), Butterworth band selection (Phase 8), CWT/FFT band choice (Phase 9), virtualenv+subprocess vs Docker for RepNet isolation (Phase 0.5/10), and the fusion formula beyond the naive weighted average (Phase 12).
- `docs/method_card.md`: intended use, known limitations (VFR video, single-hand assumption, smart-glasses viewpoint, small development set, not validated for clinical use), and a summary of observed failure modes from the Phase 16 diagnostics.

---

## PHASE 21 — Optional Stretch Polish (lower priority)

Only if time remains after Phases 0–20:

- A small Streamlit/Gradio viewer over the per-video `runs/development/<video_name>/` artifacts: waveform plots, ROI overlay, detected peaks — good for a demo GIF.
- CLI via `typer`/`click` (`hybrid run-dev`, `hybrid tune`, `hybrid ablate`) instead of hardcoded script paths.
- Dataclasses/typed result objects (e.g. `MediaPipeResult`, `CoTrackerResult`, `EstimatorResult`) per phase instead of loose dicts, for self-documentation and easier testing.
- Docker containers per environment (Phase 0.5 ADR) if virtualenv+subprocess isolation becomes painful.
- MLflow/W&B wiring on top of the experiment ledger (Phase 0.5) for a dashboard screenshot.

---

## Final Expected Pipeline

```text
Development smart-glasses video
              ↓
        MediaPipe Hands
              ↓
     Dynamic CPR foreground
              ↓
       CoTracker points
              │
              │
Full frame ───┼──→ Background features
              │             ↓
              │      RANSAC affine
              │             ↓
              │       Camera motion
              ↓             ↓
       Corrected tracker trajectory
              │
              ├───────────────┐
              │               │
              │          Farneback Flow
              │               ↓
              │        Residual CPR flow
              │               │
              └───────┬───────┘
                      ↓
               Motion waveform
                      ↓
               Butterworth filter
                      ↓
       ┌────────┬────────┬────────┐
       ↓        ↓        ↓        ↓
      CWT     AutoCorr   FFT     Peaks
       │        │        │        │
       └────────┴────┬───┴────────┘
                    │
                    │
Stabilized video → RepNet
                    │
                    ↓
           Confidence-weighted
                 fusion
                    ↓
              FINAL CPM
                    +
             confidence score
                    ↓
              Compare to GT
```

Render this as Mermaid in `docs/architecture.md` (Phase 20).

---

## Important Engineering Rules

1. Development videos only.
2. Test videos stay untouched.
3. Do not fine-tune pretrained neural networks yet.
4. Do not use GT as an input to the inference pipeline.
5. GT is only used after prediction for development evaluation/tuning.
6. Preserve real timestamps because videos are VFR.
7. Keep components modular.
8. Make parameters configurable.
9. Save intermediate signals so failures can be diagnosed.
10. Fail gracefully when MediaPipe, CoTracker, RANSAC, RepNet, or flow fails.
11. Do not hide failures by substituting GT or hard-coded 30-compression information into predictions.
12. The algorithm must estimate CPM independently.
13. Do not force the result toward the known 30-compression count.
14. Record software/model/checkpoint versions for reproducibility.
15. Prefer official pretrained model implementations/checkpoints where possible.
16. Do not overwrite original or processed dataset videos.
17. Every estimator (CWT, autocorrelation, FFT, peak detection) must have a passing synthetic-signal unit test before its output is trusted on real development video.
18. Expensive branch outputs (MediaPipe, CoTracker, RepNet) must be cached per-video; tuning (Phase 15) should not silently re-run them unless their own config changed.
19. All tunable parameters must be schema-validated and loaded from `config/default.yaml`, not hardcoded in scripts.
20. No bare `print` for diagnostics — use the shared logger; failures raise typed exceptions, never a silent `None`.
21. Every development run (and every tuning iteration) is appended to the experiment ledger with git commit hash, config hash, and RNG/RANSAC seed.
22. CI must be green (lint + unit tests) before development results are treated as final.

---

## Implementation Order

Do not attempt everything simultaneously — verify each phase's tests/sanity check before moving to the next, even running the whole order in a single continuous session under `/goal` (see "Session & Continuity Strategy" above). Implement and verify in this order:

0. Project scaffolding (repo structure, `pyproject.toml`, config schema, logging, exceptions, caching skeleton, pre-commit, CI skeleton)
1. Development-set loader + GT mapping (+ unit test)
2. MediaPipe (+ diagnostic caching)
3. CoTracker (+ caching)
4. RANSAC affine ego-motion estimation (+ seeded RNG, caching)
5. Corrected CoTracker trajectory
6. Farneback optical flow (+ caching)
7. Motion-wave fusion
8. Butterworth filtering
9. CWT (+ synthetic-signal unit test)
10. Autocorrelation (+ synthetic-signal unit test)
11. FFT (+ synthetic-signal unit test)
12. Peak detection (+ synthetic-signal unit test)
13. RepNet (subprocess isolation, caching)
14. Confidence calculation
15. Final fusion (+ integration test on synthetic/trimmed clip)
16. Development evaluation (+ bootstrap CIs, Wilcoxon test)
17. Ablation comparison
18. Parameter tuning (+ experiment ledger logging every run)
19. CI wiring + full test suite green
20. Documentation (README, Mermaid diagram, ADRs, method card)
21. Optional stretch polish (CLI, viewer, Docker, dashboard)

After each major stage, run a sanity check on one development video before applying it to the complete development set.

---

## Priority Tiers (if time-constrained)

**Must-have — do these even under a tight deadline:**
- Unit tests (synthetic signals) + CI
- Caching layer for MediaPipe/CoTracker/RepNet
- Config schema validation + centralized logging
- README + a handful of ADRs

**Nice-to-have — do if time remains:**
- Bootstrap CIs + Wilcoxon significance test on Phase 13/14 results
- Experiment ledger / MLflow or W&B dashboard
- Docker containerization
- Streamlit/Gradio diagnostic viewer, CLI, typed result dataclasses

---

## At the End, Give Me

1. Exact architecture implemented.
2. Files created.
3. Models/checkpoints used and their sources/versions.
4. Parameters/configuration.
5. Development results per video.
6. Development MAE/RMSE (with bootstrap CIs).
7. Individual estimator results.
8. Ablation results (with significance test vs. best single branch).
9. Runtime/performance.
10. Failure cases observed.
11. Which parameters were tuned and why.
12. Recommended frozen configuration for the eventual held-out test run.
13. Test coverage summary and CI status.
14. Cache hit-rate / speedup observed during tuning.
15. Experiment ledger summary (number of tuning iterations, best config and why).
16. Links to README, ADRs, and method card.

Do **not** run the held-out test videos yet.
