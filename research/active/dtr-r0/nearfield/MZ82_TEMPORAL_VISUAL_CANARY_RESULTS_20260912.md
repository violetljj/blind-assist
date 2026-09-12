# MZ82 explicit temporal visual collision canary

Decision: `EXPLICIT_TEMPORAL_VISUAL_CANARY_NO_LOW_FP_TAIL`.

The causal five-frame RGB motion recipe does not meet the predeclared low-FP
canary on the consumed MZ77 Development sequence. At no more than five added
false query bits, the three 5-klux typical profiles rescue **0 / 0 / 2 TP**.
The canary required at least 10 in one profile; the corresponding 25%-of-ToF-FN
targets are 22 / 25 / 28. Do not escalate this recipe into a learned temporal
RGB expert on the strength of MZ77.

## Fixed canary

[Protocol](MZ82_TEMPORAL_VISUAL_CANARY_20260912.md): no model fitting, future
frames, native depth, object ID, target mask, world pose, or evaluator truth is
used by the predictor. Fixed dense optical flow on 320x180 grayscale frames
produces outward flow, radial expansion, image-space TTC, and causal five-frame
stability in overlapping BODY and HEAD corridors. Fixed near/far TTC intervals
use the same query endpoints as MZ80. `features.npz` and its predictor receipt
were sealed before evaluator labels were opened.

The oracle cutoff is deliberately forbidden for deployment. It estimates only
whether this already-consumed source contains a high-precision score tail.

## Result

| MZ79 profile | ToF FN | Required 25% rescue | Oracle at added FP <=5 | At <=10 FP | At <=20 FP | At <=40 FP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 5 klux, 88% typical | 88 | 22 | **0 TP / 0 FP** | 4 / 10 | 12 / 20 | 36 / 38 |
| 5 klux, 54% typical | 98 | 25 | **0 / 0** | 5 / 10 | 14 / 20 | 40 / 38 |
| 5 klux, 17% typical | 110 | 28 | **2 / 1** | 7 / 10 | 16 / 20 | 45 / 38 |

The only <=5-FP rescue is two HEAD_NEAR bits on the wall clip; the one added
false bit is HEAD_FAR on that same wall clip. It does not recover the head-bar
or thin-pole clips at this operating point. The ordering begins to recover more
true bits only when false bits rise almost one-for-one: at 10 FP it recovers
4--7 TP, and at 20 FP it recovers 12--16 TP. That is not the selective tail
needed for an observability rescue.

This is slightly different from MZ81's exact 0/0/0 ceiling, but not a decision
reversal. Two source-specific wall bits in the most collapsed ToF profile are
far below the declared 10-TP canary and fail all three 25% rescue targets.

## Interpretation and stop

Time is present in the pixels, but this explicit corridor flow/looming/TTC
representation still mixes collision evidence with ego-induced scene expansion
and near/far or height attribution. The result supports the information-level
diagnosis: adding a causal history alone is insufficient to create a useful
high-precision rescue on this source.

Close this explicit MZ82 recipe and keep it as a negative control. Do not tune
its corridor, flow parameters, TTC weights, stability window, or cutoff on MZ77;
do not automatically train ConvGRU/video variants. A temporal RGB route may be
revisited only with source-separated motion data and a mechanism that changes
the missing information, such as measured ego-motion compensation.

Per the user-declared stop point, shift the main research bet toward physically
orthogonal observability: ToF + radar for range/radial velocity, with IMU as
ego-motion infrastructure, or a stronger depth sensor. Dual identical ToF remains
a spatial-sampling experiment and is not the primary response to ambient-light
range collapse. This priority change is a research decision, not hardware proof
or authorization to make deployment/safety claims.

## Evidence and verification

Canonical evidence is
`artifacts.local/work/mz82-temporal-visual-canary-20260912/run-v2/`. It contains
the sealed feature arrays, predictor receipt, posthoc result, source snapshot,
and hash receipt. CPU execution used OpenCV 4.10.0 and took under four seconds.
Three focused tests pass, including clip-boundary history isolation, no-future
score construction, FP-budget accounting, and missing-TTC handling. Python
compilation and `git diff --check` pass. This is consumed controlled-simulation
Development evidence, not source-separated confirmation, natural video, measured
sensor physics, latency, alert quality, user benefit, deployment, or safety
evidence.
