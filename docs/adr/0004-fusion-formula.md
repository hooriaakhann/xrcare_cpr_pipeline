# ADR 0004: confidence x agreement double-weighting, not plain confidence-weighted average

**Status:** Accepted (Phase 12)

**Decision:** The final CPM (`src/hybrid/fusion.py::fuse_estimates`) is not the spec's
starter formula `Σ(confidence_i × cpm_i) / Σ(confidence_i)` applied directly. Each
candidate's fusion weight is `confidence_i × agreement_weight_i`, where
`agreement_weight_i = 1 / (1 + |cpm_i - center| / disagreement_scale_cpm)` and `center` is
the **confidence-weighted median** (not mean) of the usable candidates.

**Why:** the spec's own worked example is the reason — a candidate (RepNet) with a CPM far
from the group's consensus should not be trusted just because its self-reported confidence
happens to be middling rather than near-zero. Confidence alone doesn't capture *disagreement*
with the other independent estimates; agreement alone would unfairly penalize a genuinely
confident estimator that disagrees for a legitimate reason (e.g. detecting a real tempo
change others missed). Multiplying both captures "trust this estimate" from two independent
angles rather than one. The center is a **median**, not a mean, specifically for its
robustness to a single outlier: using a mean would let the very outlier being evaluated pull
the center toward itself, weakening the disagreement signal exactly when it matters most.
Confidence-weighting the median (rather than a plain unweighted median) lets higher-trust
estimators have more influence on tie-breaking without reintroducing the outlier-sensitivity
of a mean.

**Consequence, observed on real data (video3_development.mp4, PROGRESS.md Phase 12):**
RepNet's raw confidence (0.662) alone would have given it a non-trivial say in the average;
factoring in that its estimate (92.66) sat far from the other four's consensus (~105) dropped
its effective fusion weight to 0.285 — the mechanism visibly doing its job on a real video,
not just the synthetic unit test. `disagreement_scale_cpm` controls how many CPM of deviation
from center is treated as "meaningfully disagreeing" before the discount kicks in materially.

**Update (Phase 15):** the initial default of 10.0 was a first-pass, untuned value. A targeted
sweep over {5.0, 10.0, 15.0, 20.0} on the development set showed a clean, monotonic trend —
smaller values (more aggressive disagreement discounting) improved both MAE and RMSE at every
tested point, with 5.0 the best of the four (dev MAE 1.976 -> 1.778, RMSE 2.199 -> 2.117).
Adopted as the new default. See PROGRESS.md's Phase 15 sweep-results entry for the full
per-video breakdown and the reasoning for stopping at 5.0 rather than sweeping smaller values
still (the approved sweep scope was these 4 values, not an open-ended search).
