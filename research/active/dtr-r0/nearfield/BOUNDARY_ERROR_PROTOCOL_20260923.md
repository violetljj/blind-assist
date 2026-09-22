# Boundary-error source audit

2026-09-23 EXPLORE diagnostic, authorized after the contact-boundary review.
Reuse consumed contact-boundary outputs and their original 1728-frame source;
preserve the 864/288/576 whole-layout split, both failed packages, A/LOCAL and
UNKNOWN. No fit, inference, threshold search, new capture or successor run.

Question: do the observed metric errors exceed the ambiguity of the supplied
binary supervision, and is boundary-near native support present in the current
readouts? This is attribution, not a new algorithm competition or information
theoretic ceiling for RGB+ToF.

One CPU saved-output/native-geometry diagnostic (TASK_NOT_GPU_SUITABLE):

1. For both axes and all splits derive the sharp monotonic label bracket from
   the entire 4-width by 9-horizon training-query grid. The labels at evaluation
   images are evaluator-only hypothetical constraints, never training labels.
   Report bracket widths, censoring, minimax half-width, and identical full-grid
   signatures with true boundaries separated by >10cm. These signatures exclude
   images and do not imply observational indistinguishability.
2. Retain frozen first-positive curve predictions and original within-5cm
   denominators. Separately count predicted left/interior/right censoring.
   Test whether the *interval containing the predicted crossing*, including
   curve discretization, intersects the label bracket. Report errors outside
   that bracket, remaining ambiguous errors, and errors on training layouts.
   Reproduce original metrics and test exact-truth curve quantization.
3. For all576 evaluation images reproduce original64-zone ToF byte-for-byte
   from native depth and fixed sensor noise identities. Compute first possible
   contact from native pixel points, all zone-lattice points, and actually
   returned winning-bin contributors. Also select the closest eligible point
   to each true boundary, independently of the first-contact minimum. Compare
   both with the full-box boundary at5cm; retain early/disagreeing native values,
   missing support and censoring. Support classes use closest-point agreement.
   Native support is a privileged diagnostic, not a public predictor or proof
   that a full obstacle boundary is observable. Finite positive boundaries only;
   sampled point silence cannot establish empty space.
4. For the closest boundary-near returned point, report the width/horizon span
   compatible with its full zone and frozen public distance interval, and with
   that point's exact angle plus the same interval. Use r=.1+3(.01+.02Z), with
   Zlow=max(.1,Z-r); restrict to the fixed cross-section and layer. Width support
   stops at h=3; horizon support retains and flags the candidate tail beyond3m.
   Selecting
   the point/zone uses evaluator depth. These are single-return support spans,
   not confidence intervals or full-scene boundary bounds. Record interval
   exclusion of the actual point, since Gaussian noise is unbounded.
5. Inspect cached public features: verify the384ToF suffix against public tokens,
   hash exact RGB/ToF/feature duplicates, and report matched lateral-pair RMS
   feature distances. Nonzero differences cannot prove learnable metric signal;
   no similarity threshold, probe fit or nearest-neighbour predictor is selected.

Write all per-image/layer/axis evidence, split summaries, input/output hashes,
and a result. Use synthetic edge cases and independently reproduce old boundary
counts and the same-axis label brackets. A useful decision distinguishes
label underspecification from errors already violating available constraints;
unresolved feature/optimization causes remain unresolved. If provenance or
parity fails, stop with a preserved failure receipt. Mechanical correction may
repair the diagnostic, never refit or change consumed predictions. Stop after
  this audit, document the narrow next mechanism question and deliver scoped Git.

Mechanical review before interpretation: v1 remains preserved. Version2 separates
first-contact minima from closest-to-truth support. Fixed decimal query knots
and sweep coordinates are canonicalized to6decimal places before an exact
open-lower interval-intersection test; this reconciles float32 encodings without
eroding real overlaps by a tolerance. Truth and saved scores are unchanged.
Version2 finished computation but failed receipt serialization on a NumPy integer;
its complete rows and partial result remain. Version3 normalizes NumPy scalar
types for JSON; this is an output repair with unchanged scientific inputs.
