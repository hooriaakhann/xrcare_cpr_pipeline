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
- **The ego-motion branch (Phase 4/5) is not globally consistent and can drift.** Phase 4's
  RANSAC affine fit is independent for every consecutive frame pair (no persistent long-lived
  track, no bundle adjustment); Phase 5's cumulative composition of those independent estimates
  into an absolute per-frame correction has no loop closure, so small per-frame errors can
  compound over a clip. Fine for this dataset's short clips (~17-30s); flagged in PROGRESS.md
  Phase 5 as a real limitation if this pipeline is ever pointed at much longer footage.
  **The optical-flow branch (Phase 6) does not share this risk, and not by luck:** it cancels
  camera motion with its own independent, per-frame mechanism (median background-region flow
  subtracted from median foreground-region flow, both from the same Farneback field) rather
  than reusing Phase 4's RANSAC transform at all. Because that subtraction is a frame-to-frame
  differential recomputed from scratch every frame -- not an absolute position built by
  composing prior frames the way Phase 5's correction is -- there is no state for a bad
  estimate to corrupt going forward. Confirmed on video2 (Phase 14/15 investigation): its
  isolated ego-motion-corrected-tracker signal is 45.5 CPM off GT (114.7 vs 69.2, a RANSAC-drift
  failure), while its optical-flow branch and the fused hybrid output both stay accurate.

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
- **Catastrophic ego-motion drift (video2) is a real but isolated failure mode, not typical.**
  Re-checked the isolated `C_cotracker_affine` ablation arm across all 6 dev videos under the
  frozen config: video2 is a clear outlier at 45.5 CPM absolute error (65.7% relative) — nearly
  5x worse than the next-worst video. video9 (9.6 CPM, 10.1%) and video10 (8.2 CPM, 8.2%) show
  moderate divergence; video1 (4.2 CPM), video3 (0.8 CPM), and video4 (1.7 CPM) are clean. The
  full fused hybrid output stays close to GT across all six regardless (0.19-3.30 CPM absolute
  error) — including video2 — confirming Phase 7/12's confidence-weighted fusion is generally
  containing single-branch ego-motion weakness across the dev set, not just recovering one known
  bad case. This isolated-branch fragility is unresolved at the algorithm level (no loop closure
  / re-anchoring implemented, see the limitation above) and shipped as-is into the frozen config
  for test evaluation — the fusion architecture is the mitigation, not a fix to the branch
  itself.
