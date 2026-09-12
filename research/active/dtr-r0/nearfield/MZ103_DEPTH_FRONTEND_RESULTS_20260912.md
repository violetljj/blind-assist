# MZ103: exact spatial evidence exposes the alert-state tradeoff

Date: 2026-09-12. EXPLORE, consumed MZ101/MZ102 Development only.
Frozen code/protocol: `80cee494a3e536f9457ee254d0dabf196a6998ca`.
Status: native diagnostic complete; conditional model stage not run. Global
registration remains pending the unrelated historical ASE input fingerprint error.
No baseline or default-App promotion; existing retained core is unchanged.

## Result

Replacing SGBM with evaluator-only native depth makes current geometric support
exactly match all 1,152 BODY/HEAD query labels in these 576 rendered frames.
The unchanged alert state then introduces all remaining native errors. Accurate
spatial evidence has headroom; final frame FN alone was a confounded gate for it.
The frozen gate nevertheless **failed** and is not retrospectively changed.

| Panel / method | TP | FP | FN | F1 | Missed events | Entirely false sessions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MZ101 ToF | 148 | 6 | 76 | .7831 | 0/28 | 0 |
| MZ101 SGBM + ToF | 202 | 25 | 22 | .8958 | 0/28 | 2 |
| MZ101 native + ToF reference | 196 | 8 | 28 | .9159 | 0/28 | 0 |
| MZ102 ToF | 126 | 8 | 96 | .7079 | 2/28 | 0 |
| MZ102 SGBM + ToF | 200 | 15 | 22 | .9153 | 0/28 | 0 |
| MZ102 native + ToF reference | 194 | 8 | 28 | .9151 | 0/28 | 0 |

Descriptive pooled final comparison: SGBM union **402/40/44**, native union
**390/16/56**. Native removes both entirely false sessions, loses no events,
and has at most 0.25 s extra paired first-correct delay. Its FN increases by 12,
so the declared no-more-FN condition fails. This is not a model negative result.

Before the alert state, the same comparison is:

| Current geometric support, pooled | TP | FP | FN |
| --- | ---: | ---: | ---: |
| ToF | 308 | 0 | 138 |
| SGBM + ToF | 435 | 62 | 11 |
| Native + ToF reference | **446** | **0** | **0** |

These rows score current support, not delivered alerts. They cannot replace the
final-alert rows above or establish measured sensor performance.

The MZ102 `small_head` slice improves from 13TP/0FP/5FN to 16/0/2; both events
remain detected, and maximum first-correct delay falls from 1.00 to 0.25 s.
Thin-left/right final TP changes from 67/68 to 60/68 on MZ101 and 69/72 to
64/72 on MZ102; all thin events remain detected and all native misses are their
first-frame confirmation cost. Occluded-thin MZ102 stays 36/40 TP.
Pooled BODY changes 184TP/16FP/14FN to 174/8/24; HEAD changes 218/24/30 to
216/8/32. Per-family, per-part and full event details remain in panel summaries.

## What the attribution proves

The post-outcome audit checks exact arrays, with no policy change:

1. `native_union_support > 0 == GT` for every query on both panels.
2. `native_final == hysteresis(GT)` under the unchanged two-on/two-off state.
3. All **56 native FN** are exactly the first frame of the 56 GT intervals.
   Twelve are left-censored at episode start, so their 0.25 s delay is an
   initialization measurement, not a claim about observed physical entry.
4. All **16 native FP** are exactly the first frame after an observed GT exit;
   current native support is absent, but the fixed state holds the warning.
5. Native loses 21 old TP and gains 9 different TP, net -12. All 21 lost TP are
   GT entry frames. Twenty follow immediately preceding false SGBM support;
   one follows an already activated false warning held across a missing-support
   frame (`cross_hit_textured` HEAD). Thus these old entry detections were
   enabled by geometrically false pre-entry evidence, not accurate earlier depth.

The geometry labels and native depth share controlled UE scene geometry. Exact
agreement establishes this harness's input/readout consistency; it is not an
independently measured 100% perception result. The task is a current forward
corridor, not future contact, and a pre-entry false label is interpreted only
under that frozen task definition.

## Decision and the next useful intervention

Retain this as a diagnostic component: improve the **raw spatial evidence** and
measure alert-state costs separately. Another surface-length rejection rule is
unsupported. Neither this result nor an ideal-depth oracle justifies removing
confirmation/holding for noisy measured depth; SGBM currently has 62 raw FP.

A later pretrained-depth experiment should declare two distinct questions before
scoring: can RGB-only depth remove spatial artifacts while preserving real small
support, and what event delay/false-warning duration does the unchanged state
then incur? A subsequent evidence-dependent confirmation policy would require
observable reliability, tested on actual predicted depth, not native truth.
Do not require a better frontend to preserve warning timing that depended on
false pre-entry support. Do not silently relax the present frozen gate either.

FoundationStereo preparation obtained official source revision
`6e8806816b533e4d13ddbb95ffa907b797060a62` and checked the research-use license.
No checkpoint/configuration was downloaded: the official Google Drive config
request hit a download quota. No alternate mirror was used. No model inference,
training or compatibility success is claimed. Root stopped preparation when the
native gate failed; all task-owned acquisition processes are released. Source,
isolated download dependencies and the acquisition receipt remain as resumable
research material, with no automatic restart. The base Python environment was
not changed.

## Verification and evidence

Both original predictions **and support counts** reproduce exactly; capture
receipt hashes, spec identity, unchanged GT, native object bounds and excluded
background/floor geometry were checked. Independent audit confirms every
attribution above, including the one carried false warning. CPU replay took
12.63 s for geometry/scoring (`TASK_NOT_GPU_SUITABLE`); no GPU model was run.

- [Frozen protocol](MZ103_DEPTH_FRONTEND_PROTOCOL_20260912.md)
- [Runner](run_mz103_depth_frontend.py) and [posthoc audit](audit_mz103_lifecycle.py)
- [Native summary](../../../../artifacts.local/work/mz103-depth-frontend-20260912/native-v1/summary.json)
- [Lifecycle attribution](../../../../artifacts.local/work/mz103-depth-frontend-20260912/lifecycle-attribution.json)
- [Independent audit](../../../../artifacts.local/work/mz103-depth-frontend-20260912/independent-native-audit.json)
- [Acquisition receipt](../../../../artifacts.local/work/mz103-depth-frontend-20260912/acquisition-receipt.json)
- [Pending registration](../../../../artifacts.local/work/mz103-depth-frontend-20260912/pending-registration.json)

The supported registration command returned the existing row-303
`input_fingerprint does not match input_refs` error. No historical row was
rewritten, no validator bypassed, and no successful ledger registration claimed.
The pending record records the intended diagnostic role, source, terminal and
artifacts; completing global metadata awaits recovery of that unrelated input.
