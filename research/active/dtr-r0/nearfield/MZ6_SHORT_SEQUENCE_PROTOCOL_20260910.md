# MZ6: bounded causal evidence pilot

2026-09-10 EXPLORE, fixed before the new sequence model outputs. The preceding
zero-fit [regression diagnostic](mz6_regression_diagnostic.py) reproduces the
27 MZ5 exact regressions: 23 far rows, 26/32 error bits with one correct branch,
6/32 with both branches wrong. This does not prove information absence. Retain
MZ5's low-error baseline and test whether recent observed returns can help when
current weak foreground returns disappear. Do not retrain or tune thresholds.

## Source and budget

One frozen Willow UE capture: four scenarios (approaching head bar, laterally
entering body obstacle, thin pole with background in the same angular region,
and lateral exit), each paired with a target-absent control; 25 samples each,
200 total. Nominal 12 Hz and a maximum two-frame history. Camera HFoV100deg,
640x360, level, eye1.7m above floor; native axial depth supplies simulated sensor
packets and separate visible-support labels. These are settled posed frames,
not real-time walking. World poses and object identities stay evaluator-side.
Mechanical failures may be repaired with preserved attempts; no source redesign
after model outcomes, budget extension, additional fits or weight/threshold sweep.

## Three methods on exactly the same observations

1. **CURRENT:** frozen compact MZ5, unchanged RGB features and generic8x8 packet.
2. **MEAN3:** arithmetic mean of current and up to two previous CURRENT logits,
   reset at each clip. Zero threshold throughout.
3. **COMPATIBLE:** same fixed MZ5 readout after conditional packet completion from
   up to two previous *raw* observations. Never feed reconstructed packets back
   into the history. No future RGB, native depth, truth, object IDs, exact pose,
   or fitted motion enters this algorithm.

COMPATIBLE is a small, falsifiable range-hypothesis transport recipe, not a
claim that optical flow resolves within-zone return ownership. Track visual
corners by forward/backward pyramidal LK flow on 320x180 RGB. Require roundtrip
error<=1.5px, photometric error<=20, and at least3 valid correspondences from a
previous zone into a current zone. Past source zone must have two returns
separated by>=0.30m; current destination exactly one return. The past farther
return must agree with the current return within `0.05+1.5*age_seconds` metres.
The nearer hypothesis must remain at least0.15m nearer than the current return.
Prefer freshest history, then greatest match count; use that past observed
near range, leave the current background range intact, and rerun the fixed ToF
branch. Displacement allowance is a declared hypothetical bound, not measured
ego motion or object velocity. At most two samples may carry stale range.
If correspondence is unavailable, make no completion. Candidate ranges are
region hypotheses, not independently localized depth or clearance evidence.

## Single artificial pressure condition

Evaluate clean packets and one seed107 artificial missing-return variant. A zone
is eligible only when it reports two ranges separated by>=0.30m and the nearer
radial bin occupies<=15% of its valid pixels. Delete the nearer return with
probability0.5, move the retained background to slot0, and mark slot1 invalid.
This uses geometry only inside the sensor simulator; eligibility/deletion masks
are not model inputs. It is a low-area first-bin stress, not a measured thin-pole
response or VL53L8CX emulation. Report eligibility and actual deletion counts;
zero eligible source opportunities means pressure NOT_EVALUABLE, not robustness.

## Evaluation and decision

Report all four event TP/FP/FN/TN, event and frame denominators, wrong-far,
cross-body, negative-control activations, and invalid/UNKNOWN coverage for each
source/pressure/method. Use native visible support>=3 pixels per BODY/HEAD
near/far query, with full-depth validity tracked separately. These labels do not
certify unseen-space clearance or continuous swept collision truth.

For each clip/event positive episode, report first valid positive index, first
detected index, delay in nominal samples/time, missed episodes and positive
retention. After eligible positive-to-negative transitions, report time to the
first two consecutive silent samples, uncleared and right-censored episodes.
Also report BODY/HEAD union events for range-transition warning continuity.
Use evaluator poses only for descriptive first-hit distance where meaningful.
No nominal-time number is measured end-to-end alert latency.

Retain the candidate only if it recovers at least one far TP or improves a first
effective hit versus CURRENT under pressure, does not lose any far TP in either
condition, adds no wrong-far/cross-body or negative-control false activations,
and does not worsen first-hit or clearance delay. Report all tradeoffs even if
exact accuracy increases. With no gain, retain CURRENT and stop this recipe;
with insufficient source/positive/dropout opportunities, report NOT_EVALUABLE.

One mechanism falsifier: rerun COMPATIBLE using fixed zero-displacement
correspondence, leaving all RGB consistency checks and raw history unchanged.
It diagnoses whether spatial movement correspondence adds value; it is not a
fourth tuned contender. Validate causal prefix invariance and clip reset using
small constructed observations. Preserve source, checkpoint, input/output hashes
and all original baseline alerts. Release the task-owned UE/Zen processes.
