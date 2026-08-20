# Method Card — Hybrid CPR Compression-Rate Estimation

## Intended use

Estimates chest-compression rate (compressions per minute, CPM) from egocentric video
captured by a head-mounted smart-glasses camera during CPR, by combining classical
computer-vision motion analysis (hand/wrist tracking, ego-motion-compensated point tracking,
optical flow, spectral/temporal rate estimators) with an independent learned repetition
counter (RepNet), fused with confidence-weighted, disagreement-aware voting.

Built for **development and research use**: implementing, calibrating, and tuning the
pipeline on a small labeled development set, and evaluating whether the hybrid combination
outperforms any single component. It is explicitly **not**:

- A clinical or safety-critical device. No claim is made about accuracy sufficient for
  real-time CPR feedback, training certification, or any decision affecting patient care.
- Validated on the held-out test set at the time of writing — CLAUDE.md rules 1-2 keep that
  set completely unseen until the pipeline is finalized and frozen; every number in this
  repo's PROGRESS.md is a **development**-set number, explicitly not a generalization claim.

## Known limitations

- **Variable frame rate (VFR) source video.** Every stage that needs a uniform sample rate
  (Butterworth filtering, Phase 8) resamples explicitly rather than assuming `1/FPS` timing;
  every other stage keeps real per-frame timestamps. This adds real engineering surface
  (Phase 1, Phase 8) that a constant-frame-rate assumption would have avoided, in exchange for
  not silently distorting timing on the actual footage this pipeline was built for.
- **Single-hand assumption.** MediaPipe is configured for `max_num_hands=1` (Phase 2) — the
  dataset's compressions are single-hand-region motion from the wearer's point of view (hands
  stacked, tracked as one region). A two-independent-hands scenario is untested.
- **Smart-glasses (egocentric, head-mounted) viewpoint specifically.** Ego-motion compensation
  (Phase 4) assumes camera motion is wearer head movement, modeled as a similarity transform
  (ADR 0001) — a reasonable model for head motion, not necessarily for a handheld or
  fixed-tripod camera.
- **Small development set.** 6 development videos (`video1/2/3/4/9/10`), GT range 69.23-105.88
  CPM. Every dev-set statistic (MAE, RMSE, bootstrap CIs, the Phase 14 Wilcoxon test) is
  computed over just 6 videos — CLAUDE.md's own framing applies directly: "development metrics
  are for tuning/debugging, not the final generalization result." A 6-video bootstrap CI is
  wide by construction; a 6-video paired Wilcoxon test has very little statistical power. Both
  are still reported, honestly, rather than omitted or oversold.
- **RepNet is applied zero-shot.** It's a general-purpose, class-agnostic repetition counter
  (Dwibedi et al., CVPR 2020), pretrained only (CLAUDE.md rule 4) — never fine-tuned on CPR or
  egocentric footage specifically. Phases 10/12 both observed it running noticeably less
  accurately than the four classical estimators on real dev videos; this is expected for an
  out-of-domain zero-shot application, not a bug in the branch itself (see PROGRESS.md Phase
  10).
- **Ego-motion/optical-flow branches are per-frame-pair, not globally consistent.** Phase 4's
  RANSAC fit is independent for every consecutive frame pair (no persistent long-lived track,
  no bundle adjustment); Phase 5's cumulative composition of those independent estimates has no
  loop closure. Fine for this dataset's short clips (~17-30s); flagged in PROGRESS.md Phase 5
  as a real limitation if this pipeline is ever pointed at much longer footage.

## Observed failure modes (development diagnostics)

Summarized from real per-video diagnostics generated during development (Phases 4/6/9/10/12,
`runs/development/<id>/` once Phase 16 has run) — not a claim of exhaustive failure-mode
coverage, just what was actually observed and is worth watching for:

- **Per-video variation in ego-motion reliability.** `video1_development.mp4` logged 17
  unreliable-transform warnings (RANSAC inlier ratio dipping as low as 0.36) across ~600
  frames; `video3_development.mp4` logged zero across ~509 frames. None individually exceeded
  `max_unreliable_span_sec`, so no run failed, but this is real evidence that background-feature
  tracking difficulty (likely scene texture/lighting-dependent) varies meaningfully between
  subjects/sessions even within this small dev set.
- **A dilution failure mode in dense optical flow, found and fixed during Phase 6:** a plain
  median of foreground-region flow was dominated by the majority-static pixels in the
  deliberately generous CPR ROI and read near-zero motion for an entire video, even though
  real ~8px/frame motion was present (cross-validated against CoTracker). Fixed via
  motion-percentile filtering (Phase 6 PROGRESS.md) — recorded here because the same failure
  mode (a large ROI's aggregate statistic getting swamped by a static majority) is a plausible
  risk anywhere a future change widens a foreground region without re-checking this.
- **RepNet's domain mismatch is consistent, not a one-off.** Observed on both spot-checked dev
  videos (Phase 12 PROGRESS.md): noticeably larger error than any classical estimator, with
  self-reported confidence that's moderate rather than near-zero — exactly the scenario Phase
  12's disagreement-aware weighting (ADR 0004) was designed to discount, and was observed doing
  so on real fusion output.
- **CWT's confidence reads structurally low even when its CPM is accurate** (Phase 9/11
  PROGRESS.md) — a property of Morlet wavelets' broader spectral peaks vs. FFT's discrete bins,
  not a sign of poor estimation. Worth remembering when reading a per-video confidence table:
  low CWT confidence alone doesn't mean the CWT estimate was wrong.
