# TARO O1R R7 fresh confirmation result

Terminal: `TARO_O1R_R7_FRESH_CONFIRMATION_NOT_EVALUABLE_DUAL_CLASS_COVERAGE`.

The execution itself is valid: eight fresh parents, 170 frames, 170 sealed
DepthART candidates, and 1,530 query labels completed without source or
threshold reselection. Phase A read zero FARO/truth payloads.

The frozen positive-occupancy values are strong but descriptive because the
dual-class gate runs first: 1,147 occupied true positives, 13 misses, zero
false positives against definite clear, precision 1.000, recall 0.9888,
one-sided Wilson lower bound 0.9976, parent-macro occupancy coverage increase
+0.9876, and zero clear outputs.

The binding blocker is label coverage, not those effectiveness gates. FARO
produced 1,160 definite occupied labels but only one definite clear label on
one parent; the protocol requires at least 50 clear labels over four parents.
All 369 UNKNOWN queries lacked the required 3 m observed-forward proof, showing
that the existing surface-support clear label is structurally unsuitable for
this negative control.

No factor is promoted. The positive factor remains an experimental fail-safe
`OCCUPIED_OBSERVED/UNKNOWN` implementation, with `CLEAR_OBSERVED` forbidden.
The unique successor is a separately falsifiable FARO ray-space definite-clear
observability canary, followed by a new fresh clear-negative-control cohort.
