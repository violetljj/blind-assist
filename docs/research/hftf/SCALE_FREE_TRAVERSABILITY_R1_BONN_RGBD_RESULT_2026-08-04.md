# Scale-Free Traversability R1 Bonn RGB-D Result

Date: 2026-08-04

Terminal: `SCALE_FREE_TRAVERSABILITY_R1_NOT_EVALUABLE_SOURCE_SUPPORT`

The frozen evaluator processed 192 public registered RGB-D frames from two Bonn
source sequences, and the independent ledger validator returned `PASS` with no
failures. The candidate executed on 100% of frames. However, the first sequence
produced valid sensor-truth scores on only `47/97 = 48.45%`, below the frozen
50% source-support gate. The round therefore fails closed as not evaluable.

Practical-use decision:
`RETAIN_FOR_DEVELOPMENT_DIAGNOSTIC_REGRESSION_AND_COUNTEREXAMPLES`. The terminal
limits the formal accuracy claim; it does not require discarding the 192-frame
ledger, 19 matched recommendations, source-coverage evidence, or implementation.

| Sequence | Truth score coverage | Truth directions | Candidate recommendations | Directional agreement* | Exact decision agreement |
|---|---:|---:|---:|---:|---:|
| `bonn_person_tracking` | 48.45% | 10 | 7/10 | 7/7 | 85.71% |
| `bonn_person_tracking2` | 51.58% | 13 | 12/13 | 12/12 | 90.48% |

`*` The observed 19/19 directional matches and zero left/right opposite errors
are diagnostic only. Because the precondition failed, they are not an admitted
accuracy estimate and cannot support the positive replication terminal. No
coverage threshold, missing-depth rule, sampling choice, or candidate parameter
was changed after outputs were read.

Both sequences were consumed by earlier project experiments. Their disclosed
roles are `PROJECT_CONSUMED_DEVELOPMENT` and the narrower
`OPERATOR_UNSEEN_EXTERNAL_REPLICATION`, because the R0 operator was frozen from
phone sessions without using these sequences. They are not globally fresh or
Confirmation evidence.

The immutable local evidence is under
`artifacts.local/evidence/hftf/scale-free-traversability-r1-bonn-rgbd-consumed-20260804/`.
Result SHA-256 is
`4AA991C6CEDFC70937F4DDD3D87E0C918C1FED592132BBFC25FB207326A53E71`;
frame-ledger SHA-256 is
`786EBBB776325E1444904B91E727293FEB2E44D87FE21262945E48F15709ABDA`.
This result does not authorize clearance, distance, alerts, safety, App
integration, or production behavior.
