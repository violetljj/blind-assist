# Future occupancy field A2 result

Date: 2026-08-03

Terminal: `FUTURE_OCCUPANCY_FIELD_A2_WINDOW_LOSO_FAIL`

The conforming execution covered all 17 consumed TUM windows and 3,522 known
0.5-second future band x horizon opportunities. Five of six frozen gates
passed. The learned field false-positive rate was 16.28%, above the fixed 15%
ceiling. No class weight, probability threshold, feature, or gate was changed.

| Arm | Brier | Log loss | Recall | FPR | MCC |
|---|---:|---:|---:|---:|---:|
| HOLD | 0.12805 | 0.44540 | 77.45% | 11.70% | 0.66035 |
| CV | 0.17061 | 0.62065 | 77.12% | 21.09% | 0.56017 |
| CA | 0.25074 | 1.11285 | 66.59% | 28.27% | 0.38348 |
| fixed IMM | 0.16293 | 0.55575 | 75.67% | 20.63% | 0.55050 |
| learned future field | **0.10178** | **0.32704** | **87.86%** | 16.28% | **0.71678** |

The learned field reduced Brier score by 20.51% and log loss by 26.57% versus
the best geometric baseline, reached ECE 0.03987, and had the highest MCC and
recall. This is evidence of useful causal future signal, but it does not pass
the predeclared operating gate and cannot be frozen for fresh confirmation.

An earlier command accidentally supplied the old rpy report without the frozen
UniDepth confidence feature. It admitted only 11/17 windows and 2,349
opportunities, so it was non-conforming and `NOT_EVALUABLE`; it was replaced by
the already established confidence-bearing rpy artifact, SHA-256
`3749895880A01A414FB6D94DA5B2ED188C811E01844E046E0B9B23AAE2220DED`.
No outcome-dependent code or parameter changed between those executions.

This exact 27-feature model family is closed. A separately frozen successor may
add the A1 2D-corridor evidence source, which independently showed a 3.45% FPR,
but may not rescue A2 through class-weight or threshold search.

Conforming machine report SHA-256:
`8339BB8D92138E5A2F02EB748AAD007D1A5C979C52C20705720D94D9C7C56F5A`.
