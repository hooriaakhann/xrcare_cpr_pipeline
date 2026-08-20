# ADR 0001: `estimateAffinePartial2D` over `estimateAffine2D` for ego-motion

**Status:** Accepted (Phase 4)

**Decision:** Ego-motion compensation (`src/hybrid/ego_motion.py`) fits a RANSAC-robust
**similarity transform** — translation + rotation + uniform scale, 4 degrees of freedom —
via `cv2.estimateAffinePartial2D`, rather than the full 6-DOF affine transform
(`cv2.estimateAffine2D`, which additionally allows independent x/y scale and shear).

**Why:** the motion being modeled is a smart-glasses wearer's head/camera movement between
consecutive frames — physically, that's translation, rotation, and (at most) a small
uniform scale change from head bob toward/away from the scene. It has no physical mechanism
that would produce independent horizontal vs. vertical scaling or shear. Fitting the full
6-DOF model to background feature correspondences would let those 2 extra degrees of freedom
absorb noise and outlier matches (RANSAC's job) as if they were real camera motion, which
would translate into worse — not better — separation between camera motion and the CPR
compression motion (the whole point of Phase 4). The similarity model is also more
numerically stable with fewer correspondences, useful since it's re-fit independently for
every consecutive frame pair rather than accumulated across a long-lived track.

**Consequence:** the estimated transform per frame pair is fully described by
`(translation_x, translation_y, rotation_rad, scale)` — see `FrameEgoMotion` — which is also
exactly the representation Phase 5's cumulative-composition math (complex numbers,
`z' = scale·e^{iθ}·z + translation`) needs; a full 6-DOF affine would have required carrying
2x3 matrices through that composition instead of four scalars.
