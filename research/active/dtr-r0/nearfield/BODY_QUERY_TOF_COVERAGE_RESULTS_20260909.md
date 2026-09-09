# Thin HEAD bars remain mixed inside fixed ToF-sized footprints

2026-09-09 EXPLORE. All600 captured frames contain visible target AND other
visible surfaces in both centered ideal footprints. No frame satisfies the
predeclared95% geometric target-dominance screen. This does NOT demonstrate
VL53L1X failure: weak/partial targets can have a response, but geometry alone
does not establish which return or validity the hardware would report.

[Protocol](BODY_QUERY_TOF_COVERAGE_20260909.md),
[CUDA coverage operator](body_query_tof_coverage.py),
[observation contract](RGB_TOF_OBSERVATION_20260909.md).

| Admitted visible-source frames | Diagonal footprint | Median target fraction | Mixed frames | Dominated frames |
| --- | ---: | ---: | ---: | ---: |
| Original fixture/background translation |27deg|8.82%|106/106|0/106|
| Original fixture/background translation |15deg|16.06%|106/106|0/106|
| Simplified fixture |27deg|15.43%|424/424|0/424|
| Simplified fixture |15deg|28.11%|424/424|0/424|

All-frame counts are120/120 and480/480 mixed respectively, with zero dominance
and zero target absence. Narrowing the footprint increases bar coverage but does
not make these sources homogeneous. Missing native support is retained (up to
54.2% in an individual footprint); it is not converted to known free/far space.
Surface ranges are evaluator geometric distributions, never synthetic sensor
returns. A nominal4m cutoff cannot model signal strength, aliasing or validity.

The user confirmed that work currently assumes simulated hardware only. Continue
with explicitly idealized observation/response models and sensitivity studies;
do not wait for physical hardware or label hypothetical returns as calibrated
VL53L1X measurements. No fusion accuracy is reported from this coverage pass.

## Implemented positive-evidence geometry

[ToF body support](tof_body_support.py) is a pure engineering primitive for the
declared coincident, aligned optical axis at1.7m. It encloses ALL possible return
directions in the ideal angular square, including a bounded radial error, and
adds a BODY/HEAD near/far event only when that whole region lies inside the
corresponding swept query volume. It uses no target identity, RGB label or depth
map. Failure to establish containment remains UNKNOWN. Fusion is positive-only:
a near return cannot erase an existing visual FAR event or assert clearance.

Under the explicit hypothetical +/-0.05m error bound, the sufficient measured
range intervals for HEAD_NEAR are approximately0.184-0.846m at27deg and
0.181-1.568m at15deg. These are GEOMETRIC intervals, not sensor detection ranges,
calibrated FoV guarantees or hardware precision. At a valid1m reading,15deg
supports HEAD_NEAR while27deg remains UNKNOWN. A2.6m reading remains UNKNOWN
for HEAD assignment at15deg. Real device use needs actual extrinsic and angular
response calibration; the current function does not claim arbitrary sensor pose.

Four [focused tests](test_tof_body_support.py) pass: invalid-return preservation,
narrow-cone positive support, wide/far ambiguity, and independent sampled radial
shell rays against the analytic bounds. A visual FAR event is preserved when
ToF adds NEAR. This is an implemented association rule, not a measured gain over
RGB-only. Mixed-return behavior remains the next simulation variable.

## Evidence and lifecycle

The worker read600 scene and600 isolated arrays, checking each against the saved
native SHA256. Two footprints per frame give1200 records. CUDA RTX3060 Laptop
coverage took10.53seconds; no new capture, RGB inference or training occurred.
Independent local arithmetic verified92x92 and50x50 raster footprints, fraction
partition sums, record hash and aggregate counts. The remote synchronous process
ended successfully and its task invocation file was removed. No UE, task, port
or persistent worker reservation was created by this pass.

Durable root: `artifacts.local/work/body-query-tof-coverage-20260909/`.
`run-v1/rows.json`, `run-v1/result.json`, `validation.json` and
`ideal-cone-engineering.json` contain data, hashes, backend, timing and limits.
Keep existing B alerts and scoped JOINT evidence; neither the failed primary
source gates nor the lack of fresh transfer confirmation is changed here.
