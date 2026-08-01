# RISKSEG-R1 P0 soft dense adapter audit contract

## Decision

`RISKSEG-R1 P0` is authorized as a pre-output-locked, benchmark-only diagnostic.
It will rerun the three frozen PIDNet-S INT8 checkpoints and the hard truth-mask
input through one isolated soft dense corridor adapter. It does not train a new
model, edit the frozen R0 adapter/event/risk chain, or change the default App.

The 30 parent events are already consumed. Five bucket-stratified,
session-disjoint outer folds are therefore only
`POST_CONSUMPTION_NESTED_DEVELOPMENT_DIAGNOSTIC`; nesting does not make the
cohort fresh, held out, independently confirmatory, or suitable for promotion.

## Why P0 exists

R0 discarded the four-channel model output at argmax, emitted connected
components with `confidence=1`, and forwarded only one component. P0 isolates
that information-path hypothesis:

1. retain all four INT8 scores and their quantization metadata;
2. apply a stable four-way softmax without calling the values calibrated risk
   probabilities;
3. preserve unknown, top-two margin, normalized entropy, and derived-known
   coverage as diagnostics;
4. pool obstacle and boundary evidence densely in the frozen trapezoid
   corridor's left/center/right thirds;
5. emit an off-chain `proxy_alert` from a finite pre-locked adapter and threshold
   grid, without creating a `Detection` or entering `RiskAnalyzer`.

The hard 256×256 truth masks are converted to one-hot values and scored by the
same family. The accurate name is
`CROSSFITTED_CONSUMED_ORACLE_INPUT_FAMILY_REFERENCE`: it is an optimistic
reference conditional on this adapter family, not a mathematical upper bound
for learned segmentation, the App, or safety.

## Frozen equations and search space

For each pixel:

```text
p = softmax(dequantized four-channel INT8 logits)
known = 1 - p_unknown
evidence = known * (p_obstacle + boundary_weight * p_boundary)
frame_score = max over zones(
    lateral_weight[zone] * mean(top fraction of evidence in zone)
)
```

The only boundary weights are `0.25, 0.5, 1.0`; top fractions are `0.0025,
0.01`; lateral profiles are center-only `[0,1,0]` and center-dominant
`[0.5,1,0.5]`. Thresholds are `0.050..0.950` at `0.025` intervals. Unknown is
not silently added to the alert score; it remains visible in the output
diagnostics.

Within each bucket, parent-event IDs are sorted and assigned round-robin to five
outer folds with frozen offsets `0,0,3,3` for blocker, boundary, normal, and
parallel-curb buckets. Every outer fold therefore has exactly six events and
every inner set has 24. For each arm and outer fold, the 24 inner events select one
configuration and threshold by the exact deterministic rank in the JSON
contract; the six outer events are then scored once. Seed `20260801` remains the
decision seed. No event outcome may be used to expand the grid, change the
folds, or select another seed.

## Gates and claim ceiling

The oracle-input family reference eliminates the route unless its aggregated
out-of-fold result has at least the current YOLO's `13/16` positive hits, no
more than `6/14` false-alert events, and at least `5/16` cleared events.

If that gate passes, the learned route supports P1 design only when the fixed
decision seed and at least two of three seeds are no worse than their own old
R0 adapter on hits, false alerts, and clearance, with at least one strict
event-level improvement. P0 success does not promote a model or App. P1/P3
still requires a new session-disjoint event cohort before any confirmatory
comparison.

Terminals are frozen:

- `TRUTH_MASK_SOFT_ADAPTER_FAIL_CHANGE_ACTIONABILITY_LABELS`
- `RISKSEG_R1_P1_NOT_AUTHORIZED_SOFT_ADAPTER_SIGNAL_INSUFFICIENT`
- `RISKSEG_R1_P1_DESIGN_AUTHORIZED`

Any identity, determinism, finite-value, test, or independent-validation failure
closes before interpretation as `HOLD_P0_CONTRACT_NOT_READY`.

The executable implementation is frozen at baseline commit
`0ac48b67dbc009ff1d1bfb0b214e80b057309eee`; the JSON contract binds the
producer, scorer, validator, and test SHA-256 values. The first soft output may
be opened only after the follow-up contract-lock commit.

### Pre-output technical amendment

The first locked execution reached the end of the first model's in-memory
feature extraction, then failed because the `ai-edge-litert` Python
`Interpreter` has no public `close()` method. No feature trace, event report, or
event outcome was written or opened. The unsupported resource-release call was
removed; equations, grids, folds, inputs, gates, and all scientific variables
remain unchanged. A new producer SHA and lock commit are required before retry.

Pre-output review then found that the original per-bucket modulo assignment
would have produced outer-fold sizes `8,8,6,4,4`, contradicting the stated
24/6 contract. No trace or event outcome had been written or opened. The split
was amended to frozen bucket offsets that produce `6,6,6,6,6`; this is recorded
as a scientific split amendment, not hidden as a code fix. The same review also
added runtime implementation/receipt checks, explicit per-tensor quantization
checks, fail-closed atomic publication into a new evidence directory, stronger
interval/fold tests, and an independently implemented 120-frame feature-canary
plus full event rescoring validator. Adapter equations, parameter grid, cohort,
YOLO reference, and outcome gates did not change.
