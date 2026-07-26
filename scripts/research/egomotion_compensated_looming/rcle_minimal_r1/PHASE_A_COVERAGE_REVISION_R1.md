# RCLE-Minimal Phase A Coverage Revision R1

Status: `FROZEN_BEFORE_FORMAL_R1_RUN`

This is the single implementation-only revision permitted by the R0
`REVISE / VALID` result. It preserves:

- protocol SHA-256
  `d20e77f3ea5f7ac55376006f1d14feb0ffb5daffd10a42792912fb89cdb1b502`;
- all 20 seeds, 2520 trial IDs, frame rates, motion conditions and degradations;
- every numerical Kill Gate A and abstention threshold;
- all R0 trial rows, including 101 `NOT_EVALUABLE` rows, and the original
  receipt SHA-256
  `14ed23e38bacc913207aaa56903a7b2cd3bebe52631338c4760f02dc5c2041ca`.

## Frozen implementation changes

1. Forward LK remains the R0 call with the frozen window, pyramid and
   termination parameters.
2. Backward cycle verification runs levels `3, 2, 1, 0`, initializes each
   backward search at the original previous point, and retains the smallest
   finite forward-backward error for each point. The frozen `<= 1.0 px`
   acceptance threshold remains unchanged.
3. Each fixed-grid cell uses `cv2.estimateAffine2D` only to select a robust
   correspondence consensus. Its RANSAC reprojection threshold is the already
   frozen `0.75 px/frame` residual threshold; `maxIters=2000`,
   `confidence=0.99`, and `refineIters=10` are fixed implementation controls.
4. The affine velocity model is refit by normalized least squares on consensus
   points. The original gates remain unchanged and are applied to that same
   consensus: support `>=12`, hull fraction `>=0.10`, condition number
   `<=1000`, and median residual `<=0.75 px/frame`.
5. No feature detector, grid, common-cell, pair-fraction, statistic, gate,
   generator, seed, trial or replacement rule changes.

The formal R1 matrix may run once after focused tests and R0 receipt regression
pass. A failing R1 is retained as the terminal implementation result; it is not
followed by threshold lowering, matrix changes, seed replacement or another
coverage revision.
