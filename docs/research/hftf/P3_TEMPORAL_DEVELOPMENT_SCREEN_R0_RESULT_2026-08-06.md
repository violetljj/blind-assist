# P3 Temporal Development Screen R0 Result

## Terminal

```text
P3_TEMPORAL_DEVELOPMENT_SIGNAL_MIXED
```

This result is `DEVELOPMENT_SIGNAL_ONLY`. It is not holdout, generalization,
product, safety, deployment, A5S, QNN/HTP, Android, or cadence evidence.

## Frozen execution

- A2 checkpoint SHA256: `8464C67D33010EF0F3225B0CDF9ACFFB2C6581150C3B2F42F45C73522A6CC0E9`
- P3 selected checkpoint SHA256: `1BD9DF8CB194B3BA505E619CFF3EAB77CBEBFF34610B8227A5E35D488CD3B639`
- Best epoch: `2` of `3`
- Best validation composite: `1.8395359937590783`
- Validation scope: `99` four-frame clips from exactly `3` independent parents
- Bootstrap and p-values: not used
- Sealed holdout: not opened

## Aggregate development comparison

| Metric | A2-392 | P3 temporal | Direction |
|---|---:|---:|---|
| Raw AbsRel median | 0.343624 | 0.309262 | improved |
| Scale-aligned AbsRel median | 0.098739 | 0.088786 | improved |
| Clearance MAE (m) | 0.502215 | 0.458115 | improved |
| Clearance-delta MAE (m) | 0.399643 | 0.393390 | improved 1.56%, below frozen 5% gate |
| False-clear rate | 0.176744 | 0.141860 | improved |
| Transition exact agreement | 0.304153 | 0.387205 | improved |
| Transition Macro-F1 | 0.135923 | 0.196441 | improved |
| Valid-to-UNKNOWN rate | 0.200000 | 0.177907 | improved |
| External abstention rate | 0.000000 | 0.089226 | increased |

The lightweight temporal head's direct clearance-delta MAE was `0.296833 m`,
but its transition exact agreement (`0.232323`) and Macro-F1 (`0.086459`)
were weak. Two of three parents improved on clearance delta, while parent
`437528` regressed from `0.311387 m` to `0.412999 m`.

## Decision

The route shows useful development signal in depth, clearance, false-clear,
valid-to-UNKNOWN, and frame-derived transition metrics. It does not satisfy the
frozen continuation rule because:

- aggregate clearance-delta improvement was only `1.5646%`, below `5%`;
- the evaluator reported `quality_guardrails_passed=false`;
- the signal was not directionally consistent across all three parents;
- the direct temporal transition head remained below the fixed majority-class
  exact-agreement baseline.

Therefore P3 remains development-only and must not proceed to A5S W8A16,
QNN/HTP conversion, Android/device testing, cadence search, or canonical
replacement. No threshold, loss, clip length, seed, class weight, validation
parent, or checkpoint may be retuned using this consumed result.

## Execution qualification

This run does not constitute a clean realization of the originally intended
single-training activation. Attempts `05`, `06`, `07`, and `08` each completed
three epochs while producer/trainer/evaluator interface defects were repaired.
The repairs did not intentionally change the frozen loss, thresholds, clip
length, class weights, data roles, or checkpoint-selection rule, but the
trainer constructs the temporal head before applying the frozen random seed.
Consequently the repeated candidates were not guaranteed to share the same
head initialization.

The `MIXED` terminal is retained as the deterministic evaluator output for
attempt `08`, but its authority is limited further to implementation-diagnostic
development evidence. It must not be represented as a contract-compliant
unique-run development screen, and no additional rerun may be used to rescue
or select among these consumed attempts.

## Evidence bindings

- Protocol SHA256: `0B8BA45DBB2848FE71F207E6BA64BB1382B3CC34F05DD504994BB77FFAEA11C9`
- Activation bindings SHA256: `D2DBC7E10052E4D733BA6D3F6B5EF3F995AF04B8CE75F5EE413E325C75C959F2`
- Training result SHA256: `A2B4F2F849F4C61969AE559FD72786D44B9FC64F66D363DA82B144A1BF028F14`
- Prediction ledger SHA256: `6235AD1EEABA6C18E5A1CF3623611DB5292B066830D7CB843F05A92A25961F29`
- Development result SHA256: `79CA2A97227C26E34526CB9F26E9ABE663F2820D3754D30E54C316988CE3C40C`

Ignored evidence root:

```text
artifacts.local/evidence/hftf/p3-temporal-development-screen-r0/
```
