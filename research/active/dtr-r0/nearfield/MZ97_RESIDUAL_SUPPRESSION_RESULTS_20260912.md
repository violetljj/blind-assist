# MZ97: residual rejection disabled; A recall gain repeats, A+B superiority does not

Decision: `RESIDUAL_DISABLED_NO_VALIDATION_GAIN`, `NEGATIVE_CONTROL` for this
fixed post-alert residual replacement. No active threshold meets the frozen
validation constraints while removing FP, so the shipped experimental decision
is exactly A. No test-driven threshold adjustment or second fit configuration.
Keep A/A+B as Development challengers; no default-App or retained-core promotion.

[Protocol](MZ97_RESIDUAL_SUPPRESSION_20260912.md) and implementation were frozen
before fitting; final pre-run code `c65ca7c6`. The fresh source uses seed97013,
128 scene instances x40 frames, split80/16/32 scenes. MZ96 data are not reused
for training, validation or test. These are new instances of shared templates,
not unseen-family transfer, physical Radar RF or human-body collision evidence.

## Fresh rule replication

Test32 scenes /1,280 frames has305 positives, including56 future-only positives.
Different positive counts from MZ96 mean its absolute scores are not a temporal
performance curve; all comparisons below are paired on this same new panel.

| Method | TP | FP | FN | F1 | Future-only recall | False segments | Fragments | Missed events |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| matched_hold | 231 | 167 | 74 | .6572 | 21/56 =37.5% | 40 | 34 | 0 |
| range-only3.6 | 234 | 208 | 71 | .6265 | 22/56 =39.3% | 45 | 33 | 0 |
| horizon A | 254 | 169 | 51 | .6978 | 36/56 =64.3% | 38 | 33 | 0 |
| coverage B | 238 | 221 | 67 | .6230 | 21/56 =37.5% | 26 | 28 | 0 |
| A+B | 272 | 211 | 33 | .6904 | 41/56 =73.2% | 28 | 19 | 0 |
| residual DISABLED | 254 | 169 | 51 | .6978 | 36/56 =64.3% | 38 | 33 | 0 |

A adds27 TP and loses4; removes60 FP and adds62. Net TP+23/FP+2, no lost
baseline event or added shared-event delay. Future-only recall gain repeats,
but simultaneous FP reduction from MZ96 does not. A changes multiple admission
conditions; this is not isolated attribution to constant-velocity prediction.

A+B adds18 TP and42 FP versus A, loses no A TP/event, and has no added shared-event
delay. Its F1 is lower despite fewer segments/fragments and higher recall.
Thus the claim that A+B consistently beats A is not supported by this replication.
Standalone B again lowers F1. A remains the more defensible primary challenger,
with uncertainty about both frame false alerts and cross-scene variability.

Paired1,000-draw episode-bootstrap95% F1 difference intervals:
A-minus-matched_hold [-.0025,.0976]; A+B-minus-A [-.0464,.0295]. Both include0.
This small shared-template panel supports a directional recall finding, not a
settled population-level superiority claim. Residual-minus-A is exactly[0,0]
because the selected filter is disabled, not because it has zero uncertainty
about an active learned effect.

## Why validation selected no-op

Training uses only1,311 A-positive frames from the80 training scenes:401 false
alerts and910 true alerts. A fixed XGBoost predicts false-alert score using the
unchanged34 causal observable features. The filter has no authority outside A's
final alerts and does not modify A's temporal state.

Validation A has145 TP,119 FP and24 correctly detected future-only frames.
Of the101 predeclared thresholds,96 remove at least one FP. None meets all
constraints. Among those96,80 fail overall98% TP retention,83 fail future-only
98% retention,2 lose an event,70 exceed delay,45 increase false segments and77
increase truth-event fragments. These are overlapping counts, not distinct causes.
With24 future-only A TP, retaining98% requires retaining all24 in this finite panel;
that discreteness is disclosed, and the floor was not relaxed after seeing results.

The highest threshold that removes any FP is0.95. It retains all145 TP and24
future-only TP and removes1 FP, but false segments rise30 to31: frame rejection
splits an existing false-alert interval. This fails the frozen event constraint.
It is a stored validation diagnostic, not an alternate test policy or a threshold
selected after test. DISABLED preserves all A outputs and has no claimed FP benefit.

No-op is the intended fallback, not an execution failure. This experiment rejects
the current fixed post-alert filter/grid under its constraints; it does not prove
all residual learning is impossible or the features contain no useful information.
The result also does not authorize relaxing gates or testing new temporal filters
on these now-consumed outcomes.

## Coverage, evidence and execution

For comparability the common no-ToF/no-alert UNKNOWN marker is761 for matched_hold,
736 for A,732 for A+B,736 for residual; positive UNKNOWN47/24/23/24 respectively.
Predictions also store explicit residual-abstention and policy-UNKNOWN masks so
rejected A alerts are never presented as clearance. Selected no-op has0 abstentions,
policy UNKNOWN736 and positive policy UNKNOWN24. F1 includes all test frames.

Ghost-present scene slice: A23TP39FP1FN F1.5349; A+B23TP57FP1FN F1.4423.
Ghost-absent: A231TP130FP50FN .7196; A+B249TP154FP32FN .7281. These are whole-scene
slices, not attribution of individual errors to ghost returns. Residual equals A
in both. No ghost flag is an input feature.

Unchanged MZ96 collector SHA256
`ca6be8d11d64b90e28638d40247dfb1a57241289cbb9a4fe0d70ed0cce2e8315`;
fresh spec SHA256
`c236a8d102f2cd7fd3e3219746cae4cc4ba14538d767c2a436760c6f2e933411`;
raw SHA256
`68e2fbd32e33e6e1afd9981379eec4b0aecd2f32b85bd08b9ab7d7ded9b690a5`.
UE native geometry recorded28,130 ToF ray hits and6,278 Radar visibility hits.
Native work7.562s, full capture lifecycle25.953s. Raw contract/shape/count/split and
three payload hashes passed; task editor/Zen processes were released.

Equivalent train-candidate fits measured CPU0.174s versus actual CUDA0.352s;
CPU retained as `CPU_FASTER_MEASURED`. One configuration, one fit per backend,
no predictive backend selection. Comparison3.363s. These are host training
measurements, not edge inference timing. Four focused tests cover subset/no-op,
safe validation rejection, and a short-event loss hidden by aggregate retention.
Independent pre-run review added explicit abstention reporting before execution.

Evidence under `artifacts.local/work/mz97-residual-suppression-20260912/`:
source-v1, capture-v1 native/raw/evaluator/receipt/process-release,
capture-validation.json, and comparison-v1 split payloads, model, selection grid,
sealed predictions/dependency hashes, results, gain importance, backend and receipt.
These durable task-owned outputs remain for reproducibility; no live capture,
training or paid allocation remains. MZ96 artifact-local dependency is reused.

The authorized bounded experiment is complete. Retain the negative result and
the replication evidence; no automatic successor, threshold relaxation or refit.
