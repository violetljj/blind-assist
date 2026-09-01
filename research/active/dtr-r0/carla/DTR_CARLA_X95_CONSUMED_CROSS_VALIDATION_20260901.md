# DTR CARLA X95 consumed cross-validation

Date: 2026-09-01

Decision: `DTR_CARLA_X95_CONSUMED_CROSS_VALIDATION_GATE_NOT_MET`

## Question

Can one trainable credentialed hazard-state model replace the accumulated
X73-X94 event lifecycle without losing the frame-level collision evidence that
X94 retains?

## Structural change

X95 consumes only the causal credential ledger already sealed in X94
predictions. It adds no perception model, geometry threshold, route threshold,
weather prior, or evaluator input. A logistic emission model is fit on ten
cohorts and applied to the held-out eleventh cohort. A fixed transition mask
then decodes five states:

`CLEAR / ONSET_PENDING / ACTIVE_MEASURED / ACTIVE_OCCLUDED / RELEASE_PENDING`.

Absence cannot create risk from `CLEAR`. `ACTIVE_OCCLUDED` requires an already
active state, the same held parent, the same valid issued-plan receipt, no
current release, and elapsed time within the inherited X24 `0.60 s` hold
window. Current release evidence has precedence.

The first replay showed that requiring the missing frame's logistic emission
to remain positive made the transition prior ineffective. X95 v2 corrected
that structural error: an already credentialed active state can enter
`ACTIVE_OCCLUDED` from the transition prior alone. No coefficient, feature,
probability cutoff, or duration was tuned after the first outcome. The v2
result below is the terminal Development result.

## Result

All eleven C26/C27/C28/C32/C34/C35/C36/C37/C39/C40/C41 sources and truths were
already consumed before X95 was designed. Leave-one-cohort-out here is an
internal Development partition, not fresh confirmation.

| Arm | Precision | Recall | Frame F1 | Event F1 | False segments | Segments/min | Fragment gaps |
|---|---:|---:|---:|---:|---:|---:|---:|
| X94 | 94.62% | 77.99% | 85.51% | 79.28% | 23 | 5.23 | 79 |
| Plain full-dropout forward-fill | 94.71% | 81.32% | 87.51% | 79.28% | 23 | 5.23 | 76 |
| 0.60 s hysteresis | 87.25% | 90.98% | 89.07% | 80.73% | 21 | 4.77 | 17 |
| Logistic emission only | 94.11% | 80.90% | 87.00% | 63.31% | 51 | 11.59 | 92 |
| **X95 v2 constrained state model** | **96.53%** | **66.12%** | **78.48%** | **83.02%** | **18** | **4.09** | **109** |

Every arm detected all `44/44` scripted contact events before contact. Median
lead was `2.80 s` for X94 and `2.75 s` for X95; P10 lead was `2.50 s` and
`2.40 s`, respectively. Median clear latency was `0.00 s` for both.

Against X94, X95 v2 produced:

- `-225 TP / -39 FP / +225 FN`;
- `-7.02 pp` frame F1;
- `+3.74 pp` Event F1;
- `-5` false-alert segments;
- `+30` fragment gaps.

The event-level signal is real within this consumed replay: fixed credential
transitions make alerts quieter while retaining all scripted events. It is not
enough to retain X95 as the unified successor. The model discards too many
positive frames and fragments positive intervals. The added frame-F1
non-regression check therefore fails even though the event-F1, precision,
event-recall, and five-state-exercise checks pass.

The strongest new diagnostic is simpler: plain full-dropout forward-fill
recovered `63` additional true-positive frames for only `2` additional false
positives, improving frame F1 by `2.00 pp` over X94 while leaving event metrics
unchanged. This consumed result does not authorize replacing X94. It identifies
multi-frame full dropout as the next decision-changing fresh falsifier and
shows that X95 should not be tuned further on these eleven cohorts.

## Evidence

- Replay output:
  `artifacts.local/evidence/dtr-carla-x95-consumed-cross-validation/x95-v2-cv-20260901-211602`
- Summary SHA-256:
  `18358B69B6864F5AF3EA3922F4E8F5FF31D386A64640EBA579E0F55D3EDCCF53`
- Fold models SHA-256:
  `8916C4C3500127802888BD56E75411B369FB2C038985499248832E2184A35CAB`
- Predictions SHA-256:
  `3AEE0D4A011F2837D52D1B1045621E5064891732D4CB53A72876CBACB9D59488`
- X95 predictor SHA-256:
  `A1C05578361435E3CC81948E67BA65734C16443FEA66D6BAA1C20DD14DCC9455`
- Runner SHA-256:
  `25A1522FE1E8DE602F3E122818A4E0F0178B5F9DF5CF7390F1A4F7A475C3FEDC`

## Claim boundary

This is consumed, post-hoc synthetic Development. The held-out folds are not
fresh because the feature topology and stop decision were formed after all
eleven source outcomes were available. X95 is an event-layer challenger and
inherits bottom-up geometry and credentials from X94; it does not prove that a
learned model replaced X24-X73 perception or collision geometry. X73 retains
the latest complete source-disjoint confirmation authority. No real-sensor,
natural-distribution, deployment, reliability, user-benefit, or safety claim
is supported.
