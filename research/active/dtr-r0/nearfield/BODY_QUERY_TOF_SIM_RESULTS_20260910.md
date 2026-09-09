# Static fusion and rotational jitter simulation results

2026-09-10 EXPLORE. Hypothetical single-zone response, consumed static sources,
not measured VL53L1X behavior or fresh confirmation. Original source gate failures
remain recorded. No fitting, new capture, model inference or alert replacement.

## Static three-way contrast

[Frozen protocol](BODY_QUERY_TOF_FUSION_SIM_20260910.md). All600 frames generated;
530 admitted endpoints scored, including every invalid simulated observation.
HEAD-near here is an event metric, not the historical three-count-query metric.
For primary15degree footprints, all three response hypotheses have the same
reported spatial metrics (they are not three independent confirmations).

| Cohort | RGB near hit | Threshold near hit | Fusion near hit | Strict pairs RGB / threshold / fusion |
| --- | --- | --- | --- | --- |
| Original fixture | 40/53 | 36/53 | 51/53 | 29 / 25 / 32 of53 |
| Simplified ordinary | 42/106 | 89/106 | 95/106 | 0 / 71 / 0 of106 |
| Size matched | 42/106 | 82/106 | 94/106 | 0 / 63 / 0 of106 |

Fusion adds11,53,52 true events and zero false events respectively. Valid packets
are77/106,174/212,169/212. Existing RGB wrong-far-on-near errors remain3/53,
26/106,26/106; false-near-on-far remain8/53,3/106,37/106. Positive-only fusion
preserves these conflicts, so near recall gain is not complete attribution repair.
Threshold's strong strict-pair result in simplified scenes also prevents claiming
that fusion is generally better. The27degree sensitivity adds zero spatial events.
All600 original B alerts are identical. No HEAD false-alarm rate is evaluable on
this all-HEAD-positive source; no real temporal trigger-distance jitter is measured.

Decision: retain positive near support as an ideal simulation component. The
predeclared recall/no-added-error criterion passes, but no network expansion,
replacement of RGB, real-hardware claim or general fusion superiority follows.

## User-requested rotational jitter

[Frozen protocol](BODY_QUERY_TOF_JITTER_20260910.md). Both arms share the same
25-step range noise/dropout on every source frame. Rotating sensor: pitch2degrees
at2Hz, yaw1degree at1Hz, plus fixed seed23 +/-0.25degree axis perturbation over2s.
Origin, scene, body frame and saved RGB remain fixed; exact sensor pose is assumed.
These are illustrative assumptions, not measured gait. All three return laws yield
the same aggregate support metrics below.

| Common-denominator metric | Stationary | Rotational jitter |
| --- | --- | --- |
| Valid observations | 10518/13250 (79.38%) | 10518/13250 (79.38%) |
| Correct near support per step | 5256/6625 (79.34%) | 4256/6625 (64.24%) |
| Near endpoints supported at least once | 265/265 | 265/265 |
| Fused near event hit per step | 5896/6625 (89.00%) | 5437/6625 (82.07%) |
| Adjacent fused spatial-state flips | 1114/12720 (8.76%) | 1302/12720 (10.24%) |
| Added false spatial events | 0 | 0 |

The denominator is repeated observations of530 endpoints, not13250 independent
scenes. Any-time coverage is already saturated in the stationary control. Jitter
loses1000 near-support observations per law and gains none. Shared-valid scalar
ranges change0,0,592 times for nearest, greatest-area, farthest laws respectively;
validity is identical. Thus the primary cost is conservative body-region
association, not measured sensor dropout. The rotated enclosing box is conservative
and may abstain more than tight shell extrema. This experiment cannot separate
that approximation cost from exact cone containment without another check.

Decision: in this bounded simulation, no coverage benefit and more spatial-output
instability. Preserve jitter as a diagnostic control; do not infer that real gait
is universally harmful. Missing translation, new occlusion, changing RGB/body
pose and measured inertial uncertainty limit the conclusion. Original B alerts
remain600/600 unchanged; reported flips concern spatial evidence only.

## Evidence and engineering checks

Artifacts: `artifacts.local/work/body-query-tof-fusion-sim-20260909/`:
`packets-v1/receipt.json`, `eval-v1/result.json`, `jitter-v1/receipt.json`,
`jitter-v1/evaluation.json`, hashed sparse packets and observation/prediction arrays.
Source depth is hash-bound; target identities/masks/truth never enter measurement
or support. Evaluator truth is used only for scoring.

Worker RTX3060 Laptop CUDA: static packet generation1.88s, jitter13.06s.
CPU scalar scoring verifies all90000 support states from range/validity/pose,
common530 endpoint inclusion and original alerts. Seven artificial geometry tests
(four packet-law, three rotation) pass; these are engineering checks only.
The jitter launcher first failed before source processing because deterministic
CuBLAS workspace configuration was missing; process-local configuration fixed it.
An evaluator attempt was stopped to replace repeated archive decompression with
one load. Both engineering attempts are recorded; scientific settings unchanged.
