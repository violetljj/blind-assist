# MZ95: pointwise authority recovers risk but releases useful vetoes

Decision: `COVERAGE_GATE_NOT_MET`; `NEGATIVE_CONTROL` for standalone negative
authority replacement. Frozen code `8ecfb84d` and
[protocol](MZ95_COVERAGE_AUTHORITY_20260912.md), one run without tuning.

| Regime / method | TP | FP | FN | F1 | False segments | Fragments |
|---|---:|---:|---:|---:|---:|---:|
| ideal matched_hold | 457 | 41 | 34 | .9242 | 8 | 27 |
| ideal coverage B | 461 | 57 | 30 | .9138 | 10 | 27 |
| proxy matched_hold | 333 | 151 | 158 | .6831 | 29 | 49 |
| proxy coverage B | 349 | 159 | 142 | .6987 | 27 | 43 |

Same 1,920 frames and491 positives per regime. B rescues16/20 proxy conflict FN,
but loses6/8 conflict TN; outer hold adds another2 FP, for8 additional FP overall.
Ideal rescues4/9 FN while losing16/82 TN (66 retained). No baseline TP is lost.
Proxy missed events remain1; ideal remains0. These paired counts show both useful
recovery and the cost of releasing correct vetoes. The no-extra-FP gate fails.

Only negative ToF authority changes; Radar admission/hysteresis stay frozen.
Range/bearing point compatibility does not establish coverage of the uncertain
Radar cone, same-object identity, height or clear space. B cannot remove existing
FP by construction: it releases vetoes. Missing ToF returns remain UNKNOWN,
never clearance evidence. Two focused tests passed, including all-true coverage
parity with matched_hold and causal/permutation checks. CPU scalar work only.

Do not promote standalone B or retune consumed conflict cohorts. The separately
authorized fresh MZ96 reports B and A+B with the same parameters, so interaction
benefits can be distinguished from this standalone failure.

Evidence: `artifacts.local/work/mz95-coverage-authority-20260912/run-v1/`.
