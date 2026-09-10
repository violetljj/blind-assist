# MZ32: broaden responsibility supervision while preserving runtime scope

2026-09-10 EXPLORE, consumed Development. MZ31 finds1165 original TRAIN
disagreements, with at least20 examples and5 explicit sites for each branch's
correct class within each query. The MZ30 eligible subset has only168 examples
and lacks the RGB-correct class for both head queries. Hypothesis: this narrow
supervision teaches incomplete branch responsibility and causes transfer errors.

One change: supervise the same selector on all frozen TRAIN RGB/ToF sign
disagreements, including baseline-negative and geometrically supported queries.
Targets remain which branch sign matches the original event truth. Inputs are
unchanged, with no current truth/site/geometry-source labels in the selector.
This broader training domain does not expand the runtime correction mask:
only MZ5-positive, geometrically unsupported branch disagreements are editable.
Both agreeing branches, supported baseline decisions and MZ28 additions survive.

Use exactly the MZ30 shared268->32ReLU->1 model, seed123 initialization, original
7562TRAIN-frame normalization,1200x16 MZ20 batches, Adam0.001 and inverse class
frequency BCE formula. Recompute weights on the expanded eligible TRAIN target
counts; this is part of the supervision change. Keep all frozen feature/branch
inputs and the original oldDEV-only calibration algorithm: cutoffs>=0.5,
maximize removedFP with zero lost baselineTP, fewer switches then higher cutoff
as ties. Calibrated cutoff values may differ because model scores change; no
posthoc threshold/seed/step rescue or second fit is included.

Compare MZ5, MZ28, frozen MZ30 and this candidate on the exact same normal and
wrong-local-visual views. Report all FP/FN, complete frames, lost/gained task
bits, trained-pole coverage, actual supervised query exposures and runtime.
A decisive improvement over MZ30 requires placement totalFP<=20, zero loss of
MZ28 true positives on all normal cohorts, unchanged MZ28 additions, no newFP
versus MZ28, and trained-pole coverage>=48/49 clean/stress. Zero-loss oldDEV
calibration alone cannot satisfy this requirement. Report raw tradeoffs and
retain a limited challenger/component if the complete gate fails.

Gate before fit: MZ31 coverage receipt passes, original baseline branch parity
and MZ30 independent audit pass; exact original initialization/normalization
and batches are available. One1200-step fit only. Save expanded TRAIN masks,
initial/final model, losses, scores, calibration, predictions, metrics and hashes.
Independently verify target/mask coverage, invariant frozen inputs/initialization,
normalization/weights, calibration optimum and task counts. Mechanical failures
retain evidence. No new source, protected EVAL, device, temporal or safety claim.
Output: artifacts.local/work/mz32-expanded-responsibility-20260910/run-v1/.
