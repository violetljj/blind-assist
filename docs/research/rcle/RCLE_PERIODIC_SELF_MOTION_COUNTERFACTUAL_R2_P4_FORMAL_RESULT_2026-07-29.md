# RCLE periodic self-motion counterfactual R2 — P4 formal result

Date: 2026-07-29
Terminal: `INTERVENTION_NOT_EVALUABLE`
Validation: `VALID`
Execution state: `COMPLETE_PRE_R3_TERMINAL`

## Answer

P4 reached its frozen formal terminal at the response-blind manipulation gate.
The selected global blur intervention passed all eight `block×motion`
subgroups, but the selected global low-texture intervention failed four of the
eight required subgroups. The contract therefore requires termination before
any formal R3 pair-core call. This is a completed fail-closed P4 result, not an
incomplete 480+16 run and not an R3 scientific outcome.

No strength was retuned, no seed was replaced, and the R3 threshold,
three-pair rule, PairState, six factorial arms and two positive-guardrail arms
were not changed.

## Frozen formal preparation

- W8 scheduler amendment: user-authorized `8 workers`, observed
  `OpenBLAS=18` and `OpenCV=1` per worker; scientific locks unchanged.
- Formal identity lock: `480 MAIN + 16 GUARD = 496` exact sequence identities,
  `298,592` frame identities and `298,096` ordered-pair identities.
- One-shot activation:
  `P4_FORMAL_EXECUTION_AUTHORIZED / ONE_SHOT`.
- Response-blind check: `80 clusters × 2 motion levels × 16 frame positions =
  2,560` frame-state checks.
- Required gate: for each degradation and each of eight `block×motion`
  subgroups, at least `18/20` sequences must pass.

## Manipulation results

| block | motion | blur pass | low-texture pass | subgroup result |
| --- | --- | ---: | ---: | --- |
| ADVIO_13 | static | 20/20 | 19/20 | PASS |
| ADVIO_13 | periodic 6DoF | 20/20 | 17/20 | FAIL |
| ADVIO_14 | static | 20/20 | 19/20 | PASS |
| ADVIO_14 | periodic 6DoF | 20/20 | 19/20 | PASS |
| ADVIO_15 | static | 20/20 | 20/20 | PASS |
| ADVIO_15 | periodic 6DoF | 20/20 | 14/20 | FAIL |
| ADVIO_17 | static | 20/20 | 17/20 | FAIL |
| ADVIO_17 | periodic 6DoF | 20/20 | 17/20 | FAIL |

All blur subgroups passed. Low texture failed in four subgroups, so the global
formal manipulation gate failed. Per the frozen precedence,
`INTERVENTION_NOT_EVALUABLE` is reached before clean-arm, positive-guardrail,
max-t or mechanism-support analysis.

## R3 and outcome firewall

- formal arms started: `0`
- formal arms completed: `0`
- main R3 pair-core calls: `0`
- guardrail R3 pair-core calls: `0`
- outcome analysis performed: `false`
- sequence16 / Android / realtime access: `false`

The missing 480+16 ledgers are intentional under the prerequisite terminal.
Running them after this failure would violate the formal contract and would not
turn this result into valid evidence.

## Evidence and receipts

- activation lock SHA-256:
  `803c8fe7774268725924a74a55c68374e6f12e72d5d002b5634394021c2620fc`
- formal identity lock SHA-256:
  `d30bb5f9fe572d2abcea65e4da8bb894e1106c72d467783e70e91e99d82a9f7e`
- manipulation producer receipt SHA-256:
  `d039224a4d5348b32dc8a3b65b604ee2a53c84ae44d96055efbccaf54b887aa1`
- manipulation independent validation receipt SHA-256:
  `0ffe02fdbd078de8204ab1f6108df5f2da4c40f37f126998901eee13a0b6f888`
- formal pre-R3 result SHA-256:
  `5cedd16ba0acee530e1551bc82ea87c27617ae7740476e33c5b404ea2327e6f4`
- independent terminal receipt SHA-256:
  `772384324861c37b4abe19f9bc7fae230752778b91cafc924321c6816afa9e06`
- independent terminal receipt:
  `validated=true`, `errors=[]`,
  `P4_PRE_R3_RESULT_VALID / INTERVENTION_NOT_EVALUABLE`

Local evidence entry:

```text
artifacts.local/evidence/rcle_periodic_self_motion_counterfactual_r2/p4_formal/
```

## Interpretation boundary

This result says the frozen low-texture intervention did not satisfy its
response-blind formal generalization gate. It does not say that R3 succeeded or
failed, does not compare motion and quality mechanisms, and does not provide
natural-video, Android, product or safety authority.
