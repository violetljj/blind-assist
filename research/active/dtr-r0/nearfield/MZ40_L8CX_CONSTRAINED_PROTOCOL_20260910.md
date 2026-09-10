# MZ40: frozen-model sensitivity to unresolved close returns

2026-09-10 EXPLORE. Question: how much of the frozen task behavior depends on
separately reporting the sub600mm pairs identified by MZ39? This is the first
executable limited-observation step toward the user-selected VL53L8CX target,
not a full physical emulator. Previous ideal evidence remains unchanged.

## Fixed interventions and baselines

Use exactly the380 admitted MZ36 frames and400-attempted/80-excluded-query-bit
denominator. RGB, truth, identities, calibration, model weights, thresholds,
normalization and the original ideal-trained bank remain frozen.
No new data, training, bank rebuild, threshold selection or protected access.

The common close mask is two valid stored returns with float64 subtraction
`r1-r0 < 0.600m`. This matches MZ39 exactly:4234zones in346frames.

- MERGE_CLOSE: on this mask only, first slot becomes float32((r0+r1)/2),
  second slot becomes0/invalid. Midpoint is computed in float64 then cast.
  This is an equal-weight unresolved-return proxy, not an intensity-weighted
  sensor estimate; the midpoint may lie between actual surfaces.
- DROP_CLOSE: on the same mask only, both slots become0/invalid. This is a
  missing-observation sensitivity comparator, not a measured failure rate.
- All other ranges/validity flags are unchanged. Both modes retain659separated
  dual-return zones; surviving returns are still ideal and not hardware-certified.

Expected0/1/2return-zone counts are16592/7069/659 for MERGE_CLOSE and
20826/2835/659 for DROP_CLOSE. Ambiguity is saved for analysis but is not supplied
as an untrained extra predictor feature. No fabricated device status values.

Compare RGB-only, ToF-only, MZ5/MZ28/MZ30/MZ35/MZ37. Frozen RGB and old ideal
outputs provide the baseline. MZ37 uses its original positive cutoff and action
rule; this run does not change its earlier17TP/2FP outcome or zero-new-FP failure.

Before changed-packet inference, run the external-packet adapter on the first
16 manifest frames with their original ideal packet and require exact frozen
logits/task decisions. Preserve mechanical failure evidence; no tolerance
widening. Only after parity passes, run each changed arm once on380frames.

## Evaluation and interpretation

Native labels stay in the evaluator. The predictor reads only RGB, packets,
calibration and frozen model assets. Packets use finite0 plus invalid masks;
invalid ToF means UNKNOWN availability, not negative obstacle truth. Do not
remove admitted frames because ToF becomes unavailable: RGB may still provide
evidence. Preserve20previously excluded frames and80UNKNOWN event bits.

Report TP/FP/FN/TN by query, exact complete frames, paired true/false gains and
losses versus ideal and RGB, source/family/site/group counts, packet0/1/2target
coverage, and fully ToF-unavailable frames. Independent scalar scoring must
reconcile counts; all packet/model/input identities and untouched components
must match. Source receipt and every new artifact are hashed.

If correct alerts or exact frames decline, retain sensitivity evidence and
prioritize matched training on a declared restricted observation. An FP drop
alone is not a gain if true alarms or coverage collapse. If no degradation,
retain finite robustness to these proxies only; strongest selection, weak-target
detectability, reflectance, ambient conditions, noise and timing remain untested.
Neither outcome separates information loss from a model's ideal-training mismatch.

## Budget and source

One compact CPU packet preparation,16parity-frame executions,2x380 frozen GPU
frame executions, and saved-output scoring/audit. No raw RGB/depth duplication
or permanent dense feature cache. Release task-owned models/processes after use.
Stop after these fixed arms; no added noise/dropout sweep or retraining here.

Documentary constraint: [UM3109 Rev12 section4.10](https://www.st.com/resource/en/user_manual/um3109-a-guide-for-using-the-vl53l8cx-lowpower-highperformance-timeofflight-multizone-ranging-sensor-stmicroelectronics.pdf).
It gives600mm minimum dual-target separation; it does not prescribe our midpoint
or whole-zone invalidation. See [the observation contract](VL53L8CX_OBSERVATION_CONTRACT_20260910.md)
and [MZ39](MZ39_L8CX_READOUT_AUDIT_RESULTS_20260910.md).
