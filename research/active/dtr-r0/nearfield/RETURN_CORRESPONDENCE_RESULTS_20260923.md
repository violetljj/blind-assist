# Perfect correspondence does not fix the unchanged interval readout

**STOP_BEFORE_STUDENT_INTERVAL_READOUT_OPPORTUNITY_FAILED.** On all 576 consumed
LOCAL transfer frames, perfect observed-contributor angular association plus
the original public distance intervals produces exactly the existing LOCAL
alerts. It retains 36/36 rescues, but removes **0/2 added false alerts**. The
1024-ray sampling control has the same decisions. Do not train a correspondence
student for this particular filter on the premise that association alone will
reproduce the previous native-XYZ result.

The [protocol](RETURN_CORRESPONDENCE_PROTOCOL_20260923.md) fixed the full-support
primary, sparse accounting control and native-witness parity before outcomes.
Its operative stop is: "If primary fails, stop before student training for this
filter." This is a negative control for that scoped readout opportunity, not a
negative result for a trained model or for all spatial RGB learning. No fit,
capture, model invocation, threshold/interval change or App modification occurred.

## Result and costs

All rows are the previously consumed, same-generator 576-frame transfer cohort,
48 clips, 256 known positive and 320 known negative frames, 32 positive events.
There is no new independent or natural-distribution confirmation.

| Readout | TP / FP / FN | Recall | Precision | FPR | False segments / sampled duration | Events |
| --- | --- | ---: | ---: | ---: | --- | --- |
| A current | 192 / 4 / 64 | 75.00% | 97.96% | 1.25% | 4 / 0.8 s | 32/32 |
| Frozen A OR LOCAL | 228 / 6 / 28 | 89.06% | 97.44% | 1.875% | 5 / 1.2 s | 32/32 |
| Exact contributors + public interval | 228 / 6 / 28 | 89.06% | 97.44% | 1.875% | 5 / 1.2 s | 32/32 |
| 1024-ray subset + public interval | 228 / 6 / 28 | 89.06% | 97.44% | 1.875% | 5 / 1.2 s | 32/32 |
| Previous native-XYZ witness | 224 / 4 / 32 | 87.50% | 98.25% | 1.25% | 4 / 0.8 s | 32/32 |

Both interval readouts preserve all LOCAL frame decisions and event first-alert
times. The native-XYZ control again loses four early detections, delaying two
events by 0.4 s and two by 0.2 s. All controls retain A by OR construction; this
is not newly learned A preservation. Both interval opportunity gates fail the
required removal of two added false alerts; the native-XYZ parity gate passes.

UNKNOWN remains 498/576 for every arm. The interval readouts alert on 156 UNKNOWN
frames and remain silent on 342, including 28 positive and 314 negative frames.
Silence is not confirmed free space. No occupancy mask is predicted; IoU is not
applicable to this support-readout diagnostic.

| Subgroup | Frames | Exact interval TP / FP / FN |
| --- | ---: | --- |
| Base appearance | 288 | 112 / 4 / 16 |
| Changed background | 288 | 116 / 2 / 12 |
| BODY | 288 | 122 / 6 / 6 |
| HEAD | 288 | 106 / 0 / 22 |
| BOUNDARY | 192 | 105 / 0 / 23 |
| INSIDE | 192 | 123 / 0 / 5 |
| OUTSIDE | 192 | 0 / 6 / 0 |

These are also exactly the LOCAL and sampled-interval subgroup counts. Across
288 appearance pairs, both interval arms retain LOCAL's 10 alert flips: two
positive losses, six positive gains and the two original false-alert removals
under changed backgrounds. All eight geometry-group results are preserved in
the output metrics; no subgroup is dropped or used for selection.

## What the previous native diagnostic had supplied

The previous [support result](LOCAL_SUPPORT_RESULTS_20260922.md) used native XYZ
to decide whether an observed contributor was actually inside the query. A
student predicting which ray contributed to a return would supply angular
association; it would still only have that zone's noisy distance interval.

For the two original added false frames, both winning query 4 (HEAD centre):

| Frame suffix, `local_transfer_body_suspended_solid_g00_base_outside_` | Full observed contributors with possible interval intersection | Fixed-grid contributors with possible intersection | Native contributors actually inside |
| --- | ---: | ---: | ---: |
| 06 | 73 | 6 | 0 |
| 07 | 45 | 4 | 0 |

Thus support remains even after removing every pixel that did not contribute to
the observed strongest bin. No background-only cleanup or perfect contributor
mask can remove these two alerts under the fixed "any possible contributor"
rule. The ambiguity includes distance along correctly associated rays. This
does not prove those 73/45 rays caused the HGB scores, nor that every possible
depth in the assumed interval is physically realized.

False-alert removal by a different classifier, a decision accepting some miss
cost, or newly justified joint angular/range evidence remains untested. Calling
such a rule "correspondence" would not make it this tested mechanism. The result
does not authorize narrowing intervals, substituting a midpoint/definite rule,
or training and selecting a rescue on these consumed outcomes.

## Reproducibility and inheritance

Sources: unchanged `ba-local-support-20260922-run` sealed lineage and diagnostic
counts, `ba-local-transfer-20260922-prepared` public ToF, and the saved evaluated
LOCAL rows. Native depth files, RGB arrays, old train/dev rows and models are
not consumed by this diagnostic. Original public intervals and identities are
verified, without a new simulator replay. The helper's teacher API is covered
only by synthetic tests; no student or training-label materialization was run.

Artifacts are rooted at `artifacts.local/evidence/ba-return-correspondence-20260923`:
the plan/source freeze and RunSpecs, sibling `-run` frame/metric/input/output
seals, and sibling `-audit/result.json`. Both governed science runs succeeded.
The first launcher attempt was rejected before execution because the new plan
was not in the master catalog. Scoped zero-copy registration fixed the missing
entry; diagnostic v2 used the same frozen source and criteria. v1 log/spec remain.

Seven synthetic tests pass. The separate audit passes **15,614 checks** over
all 576 frames, all contributor intervals and both ray sets, with 584 input
hash entries. It independently derives lattice projection, interval/slab
intersections, counts, frame decisions, false segments, event onset and gates.
It reuses the previously audited native-count control; it does not independently
replay native depth or certify a physical sensor model.

Actual CPU NumPy time: primary 7.27 s, audit 1.36 s, excluding launcher/catalog
overhead. `TASK_NOT_GPU_SUITABLE` is recorded for ragged-array evidence accounting.
No worker, simulator, training or other task-owned process remains running.

Retain the exact possibility-preserving association filter as **NEGATIVE_CONTROL**.
Keep original LOCAL as COMPONENT, its old-cohort cost failure and new-cohort
transfer result unchanged, and native support as a privileged diagnostic
COMPONENT. A and UNKNOWN retain their previous roles. Future work needs a
materially different decision mechanism or independently justified joint range
and angular evidence; no consumed point-count, grid, interval or cutoff sweep.
