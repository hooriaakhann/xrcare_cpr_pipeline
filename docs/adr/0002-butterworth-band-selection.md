# ADR 0002: 1.0-3.0 Hz (60-180 CPM) as the shared frequency band

**Status:** Accepted (Phase 8; reused as-is for Phase 9's four estimators)

**Decision:** The Butterworth band-pass filter (`filtering.butterworth_low_hz/high_hz`)
and all four classical estimators' own search bands (`estimators.cwt_freq_range_hz`,
`fft_freq_range_hz`, and the inverse-mapped `autocorr_lag_range_sec`) default to the same
**1.0-3.0 Hz (60-180 compressions/minute)** range, taken directly from the Phase 0.5 example
config rather than re-derived per phase.

**Why:** this dataset's ground truth spans 69.23-105.88 CPM (`data/metadata/
ground_truth_summary.csv`, all 10 videos) — the chosen band brackets that with wide margin
on both sides rather than being fitted to it. That headroom is deliberate: CLAUDE.md rules
1-2 forbid tuning on or inspecting the held-out test set, so a band narrowed to exactly what
the *development* videos happen to contain would risk silently excluding a legitimate
compression rate in the test set. 60 CPM is a conservative lower clinical bound (well below
any AHA-guideline-compliant rate); 180 CPM is a generous upper bound accounting for
faster-than-guideline compressions. Using the *same* band everywhere (filter and all four
estimators) also means a single Phase 15 tuning pass can reason about one number, not five
independently-drifting ones.

**Consequence:** every estimator's Nyquist validation (`0 < low < high < sample_rate/2`) is
checked against this same band; at this dataset's ~30fps native rate (Nyquist ~15Hz) there's
no risk of the upper edge (3Hz) ever approaching the limit, so no per-video adjustment is
needed. If this pipeline is ever pointed at footage with a much lower effective frame rate,
Phase 8/9's runtime Nyquist checks (not just the config schema's `low < high`) are what would
actually catch it.
