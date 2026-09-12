# MZ99: angular information limits route decisions; contact semantics remain separate

Decision: `ANGLE_INFORMATION_BENEFIT_ON_CURRENT_ROUTE`. Retain a diagnostic
component only. There is no deployable algorithm change or champion promotion.
Correcting only Radar bearing meets all predeclared R/CURRENT_ROUTE conditions:
FP753 to559 (-25.76%), original TP retained980/987 (99.29%), and zero lost
previously detected route intervals. Net TP rises987 to1016; paired changes
are36 added TP,7 lost TP,255 removed FP and61 added FP. Maximum extra interval
onset delay is0.1s. R F1 difference is+0.0614, with descriptive paired-episode
bootstrap95% interval[+0.0342,+0.0909]. This identifies a material limitation
of the simulated angular channel, not an achievable calibration improvement.

[Frozen protocol](MZ99_ANGLE_INFORMATION_PROTOCOL_20260912.md) used one new UE
capture: seed99013,128 episodes x40 frames,5120 total (nominal8.533 minutes).
All frames are diagnostic, with no training or parameter selection; inherited
split metadata are unused. Generator families overlap earlier panels. Source and
capture freeze was committed bef53615 before capture; comparison/evaluator freeze
was committed ebbf142e2b88f6d85b384c5a7679ae3f24a7204a before predictions or outcomes. One full capture and one
comparison succeeded; no outcome-driven retry, sweep or successor was launched.

## What changed, and what did not

Measured packets are shared. Only valid Radar angles are replaced by evaluator-only
pre-noise sensor-relative bearings:4960 real-actor and855 persistent-ghost slots.
All755 transient returns retain their sampled angles. Ghosts remain present and
are corrected to their own simulated virtual locations. Original measured tuple
ordering and truncation, range, Doppler, visibility, packet/valid flags, ToF and
IMU errors are identical. No ground-truth position, identity or velocity enters
the predictor. This is an angle-information intervention, not a full pose oracle
or universal upper bound on possible algorithms.

Four frozen methods run on each input: baseline matched_hold; R current raw Radar
admission; R+F adds fitted-current admission; A additionally allows fitted future
entry. All retain existing matching, ToF selection, hysteresis and outer hold.
R still uses the old angular wedge; it was not redesigned for fixed body width.
All eight predictions and both inputs were hashed before label scoring. Baseline
and A separately pass exact parity checks against their original implementations.

## Three targets, scored separately

CURRENT_ROUTE uses actual UE object footprints against a fixed0.6m wide forward
strip z=[0.2,3.6]m. BODY_1S uses continuous relative-CV footprint intersection
with an evaluation body rectangle0.6m wide x0.4m deep over[0,1]s. There are1183
route-positive frames,239 body-positive frames including68 current-contact and
171 strictly future-contact frames. Legacy wedge/center GT remains unchanged:
1417 positive frames,310 legacy future-only. These are separate task definitions;
new-target numbers cannot be joined to historical accuracy curves. Body dimensions
are simulation choices, with no height, anthropometry or real safety validation.

### Current fixed-width route

| Input | Method | TP | FP | FN | F1 | Future TP/positive |
|---|---|---:|---:|---:|---:|---:|
| measured | baseline | 963 | 837 | 220 | 0.6457 | - |
| measured | R | 987 | 753 | 196 | 0.6753 | - |
| measured | R+F | 995 | 771 | 188 | 0.6748 | - |
| measured | A | 1009 | 888 | 174 | 0.6552 | - |
| angle_corrected | baseline | 967 | 734 | 216 | 0.6706 | - |
| angle_corrected | R | 1016 | 559 | 167 | 0.7368 | - |
| angle_corrected | R+F | 1018 | 564 | 165 | 0.7363 | - |
| angle_corrected | A | 1019 | 604 | 164 | 0.7263 | - |

### Body contact within one second

| Input | Method | TP | FP | FN | F1 | Future TP/positive |
|---|---|---:|---:|---:|---:|---:|
| measured | baseline | 143 | 1657 | 96 | 0.1403 | 127/171 |
| measured | R | 138 | 1602 | 101 | 0.1395 | 124/171 |
| measured | R+F | 139 | 1627 | 100 | 0.1387 | 125/171 |
| measured | A | 139 | 1758 | 100 | 0.1301 | 125/171 |
| angle_corrected | baseline | 142 | 1559 | 97 | 0.1464 | 126/171 |
| angle_corrected | R | 141 | 1434 | 98 | 0.1555 | 125/171 |
| angle_corrected | R+F | 141 | 1441 | 98 | 0.1549 | 125/171 |
| angle_corrected | A | 141 | 1482 | 98 | 0.1515 | 125/171 |

### Unchanged legacy target

| Input | Method | TP | FP | FN | F1 | Future TP/positive |
|---|---|---:|---:|---:|---:|---:|
| measured | baseline | 1091 | 709 | 326 | 0.6783 | 126/310 |
| measured | R | 1182 | 558 | 235 | 0.7488 | 194/310 |
| measured | R+F | 1190 | 576 | 227 | 0.7477 | 194/310 |
| measured | A | 1214 | 683 | 203 | 0.7326 | 204/310 |
| angle_corrected | baseline | 1099 | 602 | 318 | 0.7049 | 126/310 |
| angle_corrected | R | 1178 | 397 | 239 | 0.7874 | 169/310 |
| angle_corrected | R+F | 1179 | 403 | 238 | 0.7863 | 169/310 |
| angle_corrected | A | 1219 | 404 | 198 | 0.8020 | 201/310 |

The BODY_1S result is a semantic mismatch diagnostic, not a collapse relative to
an old matched-target benchmark. R strict future-contact recall changes only
124/171 (72.5%) to125/171 (73.1%); R F1 remains0.1395 to0.1555 because many existing
route alerts occur without contact in the next second. A has exactly125 strict
future-contact TP under both inputs. Corrected angles therefore do not by themselves
turn these route-oriented outputs into selective imminent-contact warnings.

Do not pool target results: even though corrected R improves legacy F1, it loses
51 old legacy TP and adds47; legacy future TP falls194 to169 and maximum onset
delay increases0.8s. The favorable CURRENT_ROUTE gate is not a universal recall
retention result. A's strict future increment over R+F is14TP/117FP for measured
CURRENT_ROUTE, and1TP/40FP after correction. On BODY_1S, that same increment adds
zero TP and131FP measured, zero TP and41FP corrected. These fixed diagnostic
comparisons do not select a new policy from this panel.

## Evidence paths and remaining errors

Source attribution records the selected sensor, current/hysteresis/outer-hold
stage, and the set of qualifying return provenance kinds at its latest support
frame. This is not a unique physical cause or the original activation cause.
Mixed real/ghost/transient support stays mixed. A real-return tag does not mean
that return belongs to the particular GT hazard; a ghost-supported TP can be a
coincidental general warning when a real hazard exists elsewhere.

For R/CURRENT_ROUTE, pure real-actor Radar support paths have336FP measured and
148FP corrected; their current-evidence subset changes251 to105. Pure ghost Radar
paths instead change180 to201FP (current136 to160). Mixed paths are separate.
Thus angle correction helps route localization but does not eliminate geometrically
consistent ghosts. Corrected R still has559FP: ToF direct90, ToF temporal62,
ToF outer hold37, and370 across Radar/mixed-support paths. Direct and temporal
ToF paths are unchanged; outer-hold source allocation can change with fusion state.
These counts locate unresolved errors, not warrant a new filtering rule by themselves.

## Warning sessions and contact timing

A shared causal presentation layer retains sessions for three negative frames,
closes on the fourth and issues one prompt per session start. It never changes
the algorithm output and resets at episode boundaries. An already-active session
at a truth interval's start counts when determining subsequent repeated prompts.
This carried-session edge case was repaired and tested before scoring.

| Input/method | Starts | Starts/min | Route-false starts | Wholly route-false sessions | Repeats in route intervals |
|---|---:|---:|---:|---:|---:|
| measured/R | 101 | 11.84 | 72 | 57 | 1 |
| measured/A | 123 | 14.41 | 95 | 80 | 0 |
| angle_corrected/R | 90 | 10.55 | 61 | 46 | 1 |
| angle_corrected/A | 87 | 10.20 | 58 | 43 | 1 |

All eight input/method combinations cover12/12 evaluable object-contact events
with an active warning in the[contact-1,contact-0.5]s window. Every one is a carried
session: zero fresh starts in that timely window. There are14 right-censored
future contacts and zero left-censored contacts. The timely result is saturated
and offers no evidence of incremental predictive ability, timely new notification,
source-specific identification or realistic user response. A general warning may
cover multiple objects. Under BODY_1S, measured R has101/101 starts outside the
positive window but only81 wholly false sessions; early starts and wholly irrelevant
sessions must not be conflated. New contact-target FP need not be useless route
awareness. Short scripted episodes and12 evaluable contacts limit timing inference.

## Retained conclusion and bounded next decision

Keep the current algorithm core unchanged. Retain explicit route versus contact
contracts, paired angular diagnostic and causal warning-session evaluation.
The evidence supports improving observable angular localization for route relevance;
it does not yet establish an algorithm that achieves the corrected-input scores.
Any future work must use only available observations and keep route awareness
separate from urgent contact timing. A proposed contact-specific readout would need
its own task-matched baseline and a panel where warning timing can distinguish
methods; the present saturated contact result cannot justify promotion. No new
model, coverage rule, threshold sweep or successor was run in this task.

## Verification and reproducibility

-11 capture/geometry/session tests PASS; independent pre-scoring review found the
 carried-session counting issue, which was fixed before code freeze and scoring.
-Input pairing checks,128x40 roster, all16 raw fields, slot provenance and exact
 baseline/A parity passed. Capture and comparison receipt hashes, prediction seal
 and imported implementation hashes verified after completion.
 Independent result review verified13 comparison payload hashes,17 implementation
 hashes and4 sealed input/prediction payloads, with no further validity defect.
-Capture UE work6.890s; total owned lifecycle49.422s. Comparison CPU5.558s;
 no training, GPU or paid worker. Owned UE/editor/Zen descendants are released,
 with no survivors in process-release.json.
-Artifacts are retained for reproducibility under the canonical ignored junction:
 `artifacts.local/work/mz99-angle-information-20260912/`.
 Source: `source-v1/spec.json`; capture: `capture-v1/`; scores: `comparison-v1/`.
 `result.json` includes all task/frame/session/paired/timing/bootstrap metrics;
 `source-paths.json` preserves mixed support attribution; `contact-events.json`
 and `evaluation-labels.npz` preserve evaluators; `prediction-seal.json` and both
 receipts preserve evidence boundaries. These are synthetic UE visibility and
 measurement proxies, with no physical RF, hardware or user evaluation.

Capture receipt SHA256: `5e579b174199a0077c013f7e9e2c96e4d30460e553e547a94d3beee51f93f254`.

Comparison receipt SHA256: `aab2994776c72950c3174f26df3ccf7aeaa13d6be4ec94649f09c1a852bd5403`.
