# Frozen spatial branch: fresh region and apparent-size falsifier

Mode: EXPLORE, prospective controlled Development. No fitting or threshold search.

Question: does the retained JOINT spatial branch distinguish HEAD near/far in
new geographic regions when target projected size no longer distinguishes them?
This tests a target-size shortcut, not metric depth recovery or safety.

Frozen inputs: B checkpoint SHA256
`db39ccfcb9f3d1c8f3722711bab314ff0cf2da708e00d91f7fb263e6fec0ece0`;
JOINT `6458b02269a79b278bd639d74c9fbcf8cf9af1a3c926409c4743d85eeec46cde`;
normalization `81434e980c19206ee94327e67f2a39181fd798e4819f7a64fb928eab17ae4452`;
selection `f7989b0befdc69d472ab900aa5522bac65e3103a7afc2b4d0b684da3379e5671`.
Spatial event and query-presence threshold 0.5; retained BODY/HEAD thresholds
0.03946792706847191 / 0.5113814473152161. Fixed RGB 256x144,
100 degree horizontal FOV, 1.70 m camera height, zero pitch/roll.

## Source and intervention

Budget: two new Big City regions, dense_candidate_05 and dense_candidate_06;
240 pairs / 480 frames, equally divided between ordinary and size-matched arms.
Their 300 m source rectangles are outside all ten prior TRAIN/DEV/EVAL rectangles.
Map, distant assets and fixture primitives are shared; this is geographic
novelty, not complete visible-instance independence.

Source-only floor scouting and empty-view review precede cohort selection.
Prefer 30 admitted sites per region, four pairs per site (two appearances per arm).
Sites vary background and camera yaw; obstacle lateral offsets vary across pairs.
No model outputs may enter site selection or source qualification.

Source engineering note before cohort generation: the initial 90 m floor grid
in dense05 yielded only one eligible explicit-road site, while dense06 yielded
30. A 210 m grid is therefore probed inside the SAME frozen 300 m dense05 region;
region identity and the 240-pair budget are unchanged. The initial insufficient
site proposal is retained. Generic collision-bottom surfaces remain excluded.

Ordinary pairs translate the same assembly from near to far. Matched pairs
scale the target cuboid about the camera optical center by far/near distance,
including its lateral/vertical offset and dimensions. Its analytic projected
corners therefore coincide. Grounded supporting posts remain grounded and outside
the body corridor; their context can provide spatial evidence. This intervention
does not remove every contextual, shading, support or rasterization cue.
For a fixed site/appearance, both arms deliberately share the near endpoint;
the paired results are correlated controls, not 240 independent environments.

Native scene depth supplies visible query-count truth. Isolated native target
depth, captured after model RGB, binds target identity and projected size.
Both projected target width and height must match within 5% (or one pixel when
larger). Native floor, HEAD-only/exclusive intended range and visible target
support must qualify before inference. All planned cases and failures remain in
receipts. No performance-based replacement, recapture, retraining or threshold
changes. Mechanical failures may be repaired with unchanged model and geometry.
If fewer than 90% of planned pairs qualify in either arm, report NOT_EVALUABLE
for the intended test; no promotion. Do not silently reduce denominators.

## Four reported metrics and decision

Report BASE and JOINT separately for ordinary, matched and combined cohorts:

1. Near hit: positive native HEAD-near query cells detected at presence >=0.5,
   numerator and native-positive denominator (comparable to previous 92.67%).
2. Near-to-far confusion: near endpoints asserting the HEAD-far event at >=0.5,
   divided by admitted near endpoints, including BOTH as confused.
3. Pair-correct: BOTH endpoints have the exact two-bit HEAD near/far state,
   divided by admitted complete pairs. Each pair contributes once.
4. Original alert parity: exact retained BODY+HEAD decisions versus a separate
   frozen B forward on every captured RGB, with unchanged thresholds.

Promote only as an independent research spatial component if ordinary AND
matched strict pair-correct are each >=80%, with source coverage above and
100% original alert parity. Otherwise retain the previous controlled result,
record the failed falsifier, and do not expand to an ordinal head in this run.
All HEAD-positive sampling cannot estimate HEAD false alarm rate.

Historical correction: 1/2250 is 0.044444%, not 4.4%. The previous 1350/1500
(90%) was frame-exact near/far state, not strict pair-correct. Recomputing
unchanged old predictions gives JOINT 618/750 (82.4%) strict pairs; BASE 0/750.
The earlier result supports useful decoupling, but does not isolate representation
conflict from the changed decoder capacity and optimization.

Stop after this fixed test and scoped delivery. Retain payloads, failed attempts,
hashes and receipts; release task-owned capture processes and jobs.

## Source terminal and remaining requested diagnostic

The fixed capture completed 480 frames. Native admission retained 106 ordinary
and 106 matched pairs (88.33% each); seven sites contain additional BODY support.
All target identity, visible extent, size matching and HEAD-exclusive-range checks
passed. The original 90% source gate is NOT_EVALUABLE and is NOT relaxed.
The original protocol bytes are retained as
`artifacts.local/work/body-query-fresh-size-20260909/frozen-protocol-v1.md`.

Before ANY model predictions were read, remaining delivery was narrowed to a
zero-fit diagnostic of these 212 admitted pairs, within the user's requested
200-300-pair size. No data replacement or change to model, normalization, spatial
threshold, native labels or metric definition. The original primary experiment
cannot pass or promote; an explicit diagnostic flag reports the requested four
metrics without overriding that failed source gate. This subset is consumed
Development after scoring and cannot be relabelled fresh confirmation later.
