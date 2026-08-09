# Assistive Geometry R2 F0 synthetic factor geometry canary result

Terminal: `BLINDASSIST_ASSISTIVE_GEOMETRY_R2_F0_SYNTHETIC_FACTOR_GEOMETRY_CANARY_PASS`

F0 passed all ten conjunctive gates. The SHA-bound reducer exactly matched the expected tri-state geometry for all 22 positive analytic cases, rejected the one learned final-task shortcut negative control, and reproduced all 22 normal outputs deterministically.

The central anti-A0 result is clean: across 12 wide/open, side-obstacle, far-obstacle, missing-depth, degraded-support, weak/partial/blurred-boundary and scale-biased counterexamples, the reducer emitted no unsupported occupied band. Depth noise, scale uncertainty, support uncertainty and boundary blur each formed a frozen degradation ladder; every definite state either remained unchanged or became `UNKNOWN`, with no uncertainty-only `CLEAR_OBSERVED → OCCUPIED_OBSERVED` transition.

Positive obstacles still close correctly. `OCCUPIED_OBSERVED` requires lower-bounded obstacle evidence, guaranteed lateral overlap and an upper-bounded obstacle distance inside the horizon. Invalid global geometry, missing local evidence, incomplete boundary coverage and interval threshold crossings remain `UNKNOWN` rather than being silently folded into blocked.

This result closes only the mathematical/mechanical front door of `geometry_r2_interval_reducer_f0_v1`. It used zero learned models, zero real-dataset samples and zero training steps. It does not establish visual factor learnability or real-scene task utility.

The unique successor is `BLINDASSIST_ASSISTIVE_GEOMETRY_R2_F1_TRAIN_ONLY_FACTOR_LEARNABILITY_PROTOCOL_LOCK`. F1 execution authority remains `false`: a PASS permits only writing and freezing a separate pre-outcome F1 protocol, not data materialization, initialization, training or F2 activation. Teacher, temporal, mobile/device, Calibration, Confirmation, default-App, product and safety authority all remain false.
