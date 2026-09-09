# Frozen JOINT fails the fresh region and fixture diagnostic

2026-09-09 EXPLORE. Do not promote the spatial branch or start an ordinal head.
The 212 admitted-pair diagnostic gives JOINT zero strict correct pairs in both
ordinary and apparent-size-matched arms. Retained original alerts remain exactly
unchanged on all 480 captured frames. The previous controlled result remains
valid within its original source/fixture scope; it has not established robust
spatial transfer.

[Protocol and source disposition](BODY_QUERY_FRESH_SIZE_PROTOCOL_20260909.md),
[geometry generator](body_query_fresh_size_spec.py),
[native admission and frozen inference](body_query_fresh_size_eval.py).

## Four core metrics

Native source coverage failed the original primary gate: 106/120 pairs in each
arm, 88.33% versus the predeclared 90%. That NOT_EVALUABLE terminal is preserved.
Before model access, remaining delivery was explicitly restricted to a zero-fit
diagnostic on the 212 admitted pairs, within the user's requested 200-300 pairs.
It cannot override the failed source gate or yield a promotion. No samples were
replaced and no model, normalization, threshold or native label was changed.

| Arm and model | Near query hit | Near-to-far confusion | Strict pair-correct | Original alert parity |
| --- | ---: | ---: | ---: | ---: |
| Ordinary BASE | 0/318 (0%) | 57/106 (53.77%) | 0/106 (0%) | 212/212 |
| Ordinary JOINT | 113/318 (35.53%) | 26/106 (24.53%) | 0/106 (0%) | 212/212 |
| Size-matched BASE | 0/318 (0%) | 58/106 (54.72%) | 0/106 (0%) | 212/212 |
| Size-matched JOINT | 116/318 (36.48%) | 26/106 (24.53%) | 0/106 (0%) | 212/212 |

Combined JOINT: near hit 229/636 (36.01%), near-to-far confusion 52/212 (24.53%),
strict pair-correct 0/212, retained alert parity 424/424 admitted frames.
The separate frozen B forward also verifies parity on ALL 480/480 captured
frames, including source-excluded cases. Partial query recovery over BASE does
not satisfy the spatial task or the requested 80% strict-pair criterion.

Near hit counts positive native HEAD-near query cells. Confusion counts near
endpoints asserting FAR, including BOTH. A pair is correct only if BOTH endpoints
have the exact two-bit near/far HEAD state. No asserted range remains UNKNOWN,
never CLEAR. All-HEAD-positive sampling cannot estimate HEAD false-alarm rate.

Zero strict pairs does not mean every endpoint is wrong: the correctly classified
near endpoints have no correctly classified FAR partners. In the matched arm,
many of those far endpoints remain NEAR. This is compatible with unresolved
appearance/scale association, but is not a unique causal identification.

## Source, intervention and interpretation

Two new Big City rectangles are geographically outside all ten previous
TRAIN/DEV/EVAL regions: dense_candidate_05 is at least 638.83 m from an old
rectangle, dense_candidate_06 at least 575.04 m. They share the original map,
asset library and potentially distant visible instances. Sixty distinct source
positions have balanced cardinal views and at least 6 m spacing within a region.

The initial dense05 90 m floor grid yielded one eligible explicit-road position.
A 210 m floor grid inside its unchanged 300 m rectangle found 30 positions;
the failed first proposal is retained. All 60 empty views were visually reviewed
and passed camera-point native floor checks. This is source engineering, not
admission of actual walking routes or dense swept clearance.

The fixed collection contains 240 pairs / 480 frames. Twenty-eight pairs across
seven sites were excluded solely because native background BODY support violates
the frozen HEAD-only scene condition. None failed native target identity,
HEAD-exclusive-range, visible target extent or apparent-size matching. Native
labels and every rejected frame remain preserved; exclusion did not use scores.

Matched pairs scale the target bar around the optical center by far/near distance,
including its lateral and vertical offset and all three dimensions. Analytic
projected corners match within 1.14e-13 pixels; actual isolated and visible native
target extents pass the frozen pixel tolerance. Grounded posts retain contextual
and thickness cues. This removes target projected-extent differences, not all
possible scale or scene shortcuts. Both arms share the same near geometry per
site/appearance; the 240 pairs are correlated controls, not independent scenes.

An important limitation is that the ordinary arm also changes the fixture from
the old training assembly: the new bar is 1.20/1.35 m wide, with two short posts
and default material, while an inspected old crossbar has a 2 m member, 3.3 m
uprights, clamps, feet/rear rails and explicit painted-metal materials. Thus the
failure combines new geography with new assembly/appearance. It cannot identify
geography alone, nor prove that pixel size alone caused the failure. The ordinary
arm already collapses, so this run has no strong ordinary baseline from which to
isolate an additional apparent-size penalty.

The decision is to retain original B alerts and the previous scoped JOINT result,
without a broader spatial-structure claim. A future attribution check should
hold the original fixture fixed while separating background and assembly changes;
none was launched here. No ordinal training, rescue fit or threshold sweep ran.

## Validation and retained evidence

Model/normalization/selection hashes match the frozen protocol. Source capture
and world verification pass on both regions. Admission binds original specs
(only worker map-path rewriting is allowed), actual RGB/depth capture poses,
native target component/mesh identity, floor probes and native visible counts.
All 536 returned files were hash verified before inference. GPU admission ran on
the RTX 3060 worker; frozen inference ran on RTX 5060 Laptop CUDA in 5.71 seconds,
excluding model startup. No training occurred.

The surprising collapse prompted a small consumed-source interface regression.
An eight-image batch exceeded the old 2e-6 numerical tolerance (JOINT 2.18e-5).
Reusing the original aligned 32-image batch passed the UNCHANGED tolerance:
JOINT maximum probability error 1.19e-7, BASE exactly zero. The failed small-batch
attempt is retained. This supports correct model/preprocessing wiring; it does
not turn the new source diagnostic into a successful transfer result.

An independent no-inference audit reproduces all four metrics, reconstructs
range events with float64 count convolution (maximum discrepancy 2.24e-7 with
identical threshold decisions), and reconstructs BASE alerts from its six-cell
counts with exact decision parity. Its full receipt is
`independent-fresh-metric-validation.json` under the durable root below.

Delivery packages the unchanged native admission functions in
`body_query_fresh_native_labels.py` to avoid importing unrelated collection WIP.
AST parity is verified in `packaging-parity.json`. The exact executed inference
source remains hash-bound in `executed-inference-v1/`; only the packaged import
path changes. No fresh inference was rerun for packaging.

Old-number audit: 1/2250 is correctly 0.044444%. The previous 90% was frame-exact
1350/1500, while strict both-endpoint correctness was 618/750 (82.4%), not 90%.
See the corrected [historical result](BODY_QUERY_CONTEXT_DISTANCE_RESULTS_20260909.md).

Durable root: `artifacts.local/work/body-query-fresh-size-20260909/`.
Key records: `cohort-freeze.json`, `frozen-protocol-v1.md`,
`returned-full-capture-v1/admission-v1/`, `inference-v1/`,
`historical-aligned32-interface-regression.json`, and `worker-final-release.json`.
Raw native/isolated arrays remain in the same task's worker `full-capture-v1`;
primary retains RGB, native-count arrays, manifests, hashes, logs and failures.
All task-owned UE processes, scheduled tasks and Zen listeners are gone; capture
temporary directories were removed. No shared runtime, checkpoint or cache was deleted.
