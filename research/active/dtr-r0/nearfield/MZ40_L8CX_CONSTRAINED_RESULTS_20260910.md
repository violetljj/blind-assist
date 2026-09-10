# MZ40: unresolved-return constraints expose a selector transfer gap

2026-09-10 EXPLORE. The fixed sensitivity experiment is complete. A midpoint
merge loses two MZ37 true alerts; missing close-return zones lose eighteen.
The latter also exposes eight true MZ28 alerts removed by the frozen MZ35
selector. Retain this diagnostic and the unchanged ideal comparators; prioritize
matched restricted-observation training and an observable-quality-aware selector
before claiming that new object variety solves this failure.

## Scope and identity

The [frozen protocol](MZ40_L8CX_CONSTRAINED_PROTOCOL_20260910.md) uses the same
400 attempted / 380 admitted MZ36 frames, 76 admitted groups, 1,520 known query
bits (304 positive / 1,216 negative) and 80 UNKNOWN bits. No exclusion changed.
RGB, weights, bank, cutoffs, normalization and labels are unchanged. This is
consumed controlled Development with old asset families and regions.

This comparison retains a configured two-target assumption. On the same 4,234
sub600mm dual-return zones in 346 frames, MERGE_CLOSE emits a float32 midpoint
and invalid second slot; DROP_CLOSE makes both slots invalid. Neither midpoint
weights nor the missing-zone rate are measured device physics. The other 659
dual-return zones remain ideal. Default one-target/strongest operation and
weak-target detectability, noise, ambient conditions and timing are untested.

## Task outcomes

Cells are **FP / FN / exact frames**. FP and FN count query bits; exact requires
all four queries correct in a complete admitted frame. Denominator stays 380.

| Method | Ideal | MERGE_CLOSE | DROP_CLOSE |
|---|---:|---:|---:|
| RGB only | 23 / 14 / 348 | 23 / 14 / 348 | 23 / 14 / 348 |
| ToF only | 5 / 17 / 362 | 3 / 32 / 349 | 28 / 118 / 265 |
| MZ5 | 6 / 10 / 365 | 7 / 14 / 362 | 9 / 21 / 354 |
| MZ28 | 6 / 2 / 372 | 8 / 4 / 369 | 9 / 10 / 364 |
| MZ30 | 4 / 2 / 374 | 5 / 4 / 371 | 3 / 23 / 359 |
| MZ35 | 3 / 2 / 375 | 4 / 4 / 372 | 2 / 18 / 363 |
| MZ37 | 3 / 0 / 377 | 4 / 2 / 374 | 2 / 18 / 363 |

MZ37 versus its ideal outputs:

- MERGE_CLOSE: loses 2 HEAD_FAR true bits on crossbars, adds 1 HEAD_FAR false
  bit on an oblique rod; no recovered true bits or removed false bits. Three
  exact frames are lost. Exact groups decrease from 73 to 70 of 76.
- DROP_CLOSE: loses [1, 5, 4, 8] true bits in BODY_NEAR, BODY_FAR, HEAD_NEAR,
  HEAD_FAR order, removes one HEAD_FAR false bit, and adds no true or false
  bits. One exact frame is gained and fifteen lost. Exact groups fall to 62.
  Seventeen missed bits are on crossbars and one on a cabinet.

RGB scores remain exactly unchanged in both arms. Under DROP_CLOSE, MZ37
still removes 21 RGB false bits and gains 9 RGB true bits, but loses 13 true
bits that RGB alone detected. Overall exact frames therefore hide a real
false-alarm / missed-alarm tradeoff.

## Mechanism exposed by the saved outputs

Under ideal and merged packets the MZ35 negative selector removes zero MZ28
true bits. Under DROP_CLOSE it removes eight true bits [1, 2, 2, 3] while
removing seven false bits [1, 0, 2, 4]. Thus MZ28 has 10 misses and 9 false
bits; applying MZ35 produces 18 misses and 2 false bits. This is a direct
within-arm action comparison, not an attribution of all errors to the sensor.
MZ37 restores no bits in that arm. Its earlier ideal-input success does not
establish recovery when geometric support and branch responsibilities shift.

The existing models and availability bank were trained on ideal packets.
This run cannot separate information loss from input-distribution mismatch.
It supports testing matched training and preventing missing ToF support from
being treated as sufficient evidence to suppress an RGB alarm. It does not
establish an absolute RGB veto or a calibrated confidence threshold.

## Observation coverage

| Packet | 0 / 1 / 2 return zones | Valid slots | Frames with no ToF returns |
|---|---:|---:|---:|
| IDEAL | 16592 / 2835 / 4893 | 12621 | 34 |
| MERGE_CLOSE | 16592 / 7069 / 659 | 8387 | 34 |
| DROP_CLOSE | 20826 / 2835 / 659 | 4153 | 41 |

Each arm has 24,320 zones. Invalid measurements stay zero plus invalid
flags; sensor unavailability is not obstacle absence. Even fully ToF-missing
admitted frames remain scored, because RGB is available. The prior 20 excluded
frames / 80 unknown truth bits remain separate. No new abstention output or
clearance interpretation is introduced.

## Verification, resources and evidence

- First 16 ideal packets: every original saved array and MZ37 restoration
  array matched exactly before changed inference. No tolerance was widened.
- Independent audit: 48,640 arm-zone scalar changes, 31,920 known model-query
  decisions, paired exact/event changes, all regional/site/family and group
  scores, RGB hashes/features, packet identity and frozen asset hashes pass.
- GPU executions: 16 parity frames plus 2 x 380 changed frames, zero fitting.
  MERGE_CLOSE took 14.199s and DROP_CLOSE 12.052s including setup/preprocessing;
  these are desktop CUDA timings, not Android performance. Packet preparation,
  saved-output scoring and scalar audit are CPU TASK_NOT_GPU_SUITABLE.
- No new RGB/depth captures or duplicated raw source, no permanent dense
  feature cache, and no original dataset deletion. Models release CUDA in
  finally blocks. The scoped GPU processes completed.

Canonical evidence: `artifacts.local/work/mz40-l8cx-constrained-20260910/`:
`packets-v1/{receipt,result}.json`, `parity-v1/receipt.json`,
`parity-check-v1/{receipt,result}.json`, each arm's inference receipt,
`score-v1/{receipt,result}.json`, `score-v1/scored.npz`,
`score-v1/changes.json` and `audit.json`.

The experiment stops after the two registered arms. Diverse nonvegetation
and vegetation object replacement remains useful for testing shape coverage,
but needs paired rendering, actual-surface labels and declared constrained
observations. No hardware, natural-scene, protected, deployment or safety
claim is supported here. See the [observation contract](VL53L8CX_OBSERVATION_CONTRACT_20260910.md).
