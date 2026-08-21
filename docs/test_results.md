# Held-Out Test-Set Results

This is the pipeline's one and only run against the held-out test set, per CLAUDE.md rules 1-2.
It ran once, under a frozen, tagged config, with no changes made in response to what it
reported. Everything above this document (Phases 0-21) was built and tuned on the development
set alone; nothing below reflects any adjustment made after seeing these numbers.

**Frozen config:** git tag `frozen-for-test-v1`, commit `5f22756faf03d5924e022c3410b59be115393bf6`,
`config/frozen.yaml`, `config_hash=e1768d77f65e`. Run via `hybrid.run_test`, which loads that
file by explicit path — see `src/hybrid/run_test.py` and PROGRESS.md's Phase 15 entry for how
that config was arrived at (development-set tuning only, before this file existed).

## Per-video results

| video | GT CPM | CWT | Autocorr | FFT | Peaks | RepNet | Final | Signed err | Abs err | Confidence |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| video5 | 90.00 | 93.04 | 94.74 | 96.16 | 94.74 | 95.34 | 95.24 | +5.24 | 5.24 | 0.615 |
| video6 | 85.71 | 83.32 | 78.26 | 77.14 | 81.82 | 10.90 | 76.99 | -8.72 | 8.72 | 0.296 |
| video7 | 90.00 | 97.90 | 94.74 | 99.00 | 94.74 | 61.97 | 94.37 | +4.37 | 4.37 | 0.466 |
| video8 | 94.74 | 93.04 | 94.74 | 94.90 | 94.74 | 70.50 | 93.66 | -1.08 | 1.08 | 0.558 |

## Dev vs. test, side by side

| | Dev (n=6, frozen config) | Test (n=4, frozen config) |
|---|---:|---:|
| MAE | 1.778 (95% CI [0.883, 2.680]) | 4.851 (95% CI [2.117, 7.632]) |
| RMSE | 2.117 | 5.562 |
| mean signed error | +0.317 | -0.047 |
| median AE | 1.833 | 4.805 |
| max AE | 3.304 (video9) | 8.721 (video6) |

**Test MAE is 2.7x higher than dev MAE; test RMSE is 2.6x higher than dev RMSE.** Stated
plainly, not softened. The two 95% CIs overlap only slightly at the edges (test's lower bound,
2.117, sits just under dev's upper bound, 2.680) — at n=4/n=6 that overlap means this isn't
statistically airtight proof of a large population-level gap, but the point estimates, and
every individual test video's error, tell a consistent story in the same direction. This is a
real, documented generalization gap, not a rounding artifact, and it is reported here rather
than acted on.

## video6: a shared-signal problem, not four independent failures

All four classical estimators (CWT, autocorrelation, FFT, peak detection) landed low for
video6 (77.1-83.3 CPM against a GT of 85.71) and, on the surface, that agreement might read as
four independent methods converging on the same wrong answer -- weak evidence the true rate is
actually lower. It isn't independent evidence. `run_estimators_on_video` (`estimators.py:204`)
hands all four estimators the exact same array, `filter_result.filtered_signal` -- one shared,
already-fused-and-filtered motion waveform. CWT/autocorrelation/FFT/peak-detection are four
different mathematical readings of one input, not four separately-derived signals. Their
agreement here says the shared upstream signal (Phase 7's tracker+flow fusion, Phase 8's
Butterworth filtering) was itself off for this video, not that four independent lines of
evidence happened to coincide. That points at the motion-signal stage as the likely source of
video6's error, not at estimator choice.

RepNet's near-total miss on video6 (10.90 CPM vs. GT 85.71) is a separate, independent failure
-- RepNet runs on raw video frames, not the fused waveform -- and is consistent with the
domain-mismatch pattern already documented in `docs/method_card.md` (Phases 10/12): a
meaningfully wrong answer at a confidence (0.47) that isn't low enough to be self-evidently
discounted.

## New limitation: confidence does not reliably track accuracy

Two pieces of evidence from this run, not present (at this magnitude) in the dev-set results:

- **video5 had the highest confidence of any test video (0.615) and the second-worst error
  (5.24 CPM).** Every branch, including RepNet, agreed tightly with every other branch
  (93.0-96.2 CPM) -- high inter-branch agreement drove the confidence score up -- but the whole
  cluster was consistently biased above GT. Confidence here measured agreement, not
  correctness; the branches were confidently wrong together.
- **RepNet's video6 collapse carried a confidence of 0.47** -- moderate, not near-zero -- while
  missing GT by nearly 75 CPM. The same pattern Phase 12's disagreement-aware fusion (ADR 0004)
  was built to discount, observed again here at a more extreme magnitude than anything seen in
  development.

Read together: a high overall confidence score is evidence of inter-branch agreement, not a
guarantee of accuracy, and should not be treated as one by anything consuming this pipeline's
output.

## Errors are not systematically directional

Test-set mean signed error is -0.047 -- essentially zero -- despite individual signed errors
ranging from -8.72 to +5.24. The errors did not cancel out because they were small; they
canceled out because they pointed in different directions on different videos (video5 and
video7 over-estimated, video6 and video8 under-estimated). That pattern -- near-zero aggregate
bias alongside high individual-video variance -- is consistent with increased variance on novel
footage rather than a systematic directional bias in the pipeline itself. A biased pipeline
would show a nonzero mean signed error even on a small sample; this one doesn't, which is
mildly reassuring, but it doesn't offset the magnitude of the individual-video errors above.

## What this does and doesn't mean

The development-set numbers (Phases 13-15) were always framed as tuning/debugging signal, not
a generalization claim -- this is that generalization claim, and it's weaker than the dev
numbers alone would suggest. Per the freeze rule, none of the findings above have been, or will
be, acted on by adjusting `config/frozen.yaml`, the fusion formula, or any estimator. A future
iteration that wants to address the video6-style shared-signal failure mode, or the
confidence-doesn't-track-accuracy finding, would do so as new, explicitly-scoped work -- re-frozen
under a new tag, re-run once, on a full understanding that this test set has now been seen and
can no longer serve as a clean held-out check.
