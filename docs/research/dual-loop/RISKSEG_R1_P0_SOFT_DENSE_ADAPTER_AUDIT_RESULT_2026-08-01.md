# RISKSEG-R1 P0 soft dense adapter audit result

## Decision

`COMPLETE_VALID_NEGATIVE /
TRUTH_MASK_SOFT_ADAPTER_FAIL_CHANGE_ACTIONABILITY_LABELS`

Do not enter R1 P1 training with the current four-class target. Do not rescue it
with a larger segmentation model, another adapter grid, temporal confirmation,
component rules, or App integration. The next admissible research step is to
redesign event/actionability labels and then collect a new session-disjoint
parent-event cohort under a new frozen contract.

This conclusion is bounded to a post-consumption nested Development diagnostic.
It is not fresh confirmation, model promotion, App evidence, or safety
evidence.

## What P0 tested

The pre-output lock preserved the complete four-channel INT8 tensor through
softmax and dense corridor pooling. It removed the R0 information bottleneck
`argmax -> connected component -> confidence=1 -> take(1)` without changing the
R0 production adapter, event chain, risk rules, or default App.

Five exact six-event outer folds yielded one OOF result per parent event. The
30-event/30-session/1,920-frame cohort remained
`CONSUMED_DEVELOPMENT_ONLY`. The truth masks were hard class-ID masks converted
to one-hot values, so the reference is
`CROSSFITTED_CONSUMED_ORACLE_INPUT_FAMILY_REFERENCE`, not a mathematical model
or App upper bound.

## Gate result

| Arm | Positive hits | Blocker / boundary hits | False-alert events | Cleared events | Result |
|---|---:|---:|---:|---:|---|
| Current YOLO reference | 13/16 | — | 6/14 | 5/16 | Comparator |
| Truth-mask soft adapter | 14/16 | 8/8 / 6/8 | 12/14 | 4/16 | Truth gate fail |
| PIDNet seed 20260801 | 11/16 | 6/8 / 5/8 | 9/14 | 8/16 | Guardrails fail |
| PIDNet seed 20260802 | 12/16 | 4/8 / 8/8 | 7/14 | 9/16 | Guardrails fail |
| PIDNet seed 20260803 | 7/16 | 3/8 / 4/8 | 8/14 | 11/16 | Guardrails fail |

The truth-mask route passed only the hit floor (`14 >= 13`). It failed both
false alerts (`12 > 6`) and clearance (`4 < 5`). Its fold selection was fully
stable—every fold chose `bw0.25_top0.0025_center_dominant` at threshold
`0.525`—so this is not explained by fold-to-fold parameter instability.

The 12 oracle-input false alerts include six of seven parallel-curb negatives
and six of seven normal-walkable negatives. The two remaining oracle-input misses
are both boundary-level-change positives. Perfect four-class IDs plus this
pre-locked dense family therefore do not encode enough event actionability to
separate “present in corridor” from “should alert now / has cleared.”

All three learned seeds reduced false alerts relative to their old argmax
adapter, but only by trading away positive hits and, for seeds 20260801/20260802,
clearance. The fixed decision seed changed from `13 hits / 13 false / 14
cleared` to `11 / 9 / 8`; its common-hit median alert was also `+8` frames
later than YOLO. No seed passed all relative guardrails, so seed stability is
`0/3`.

## Validation and amendments

The final trace has 7,680 rows: 1,920 frames for each of three frozen PIDNet
checkpoints plus the truth-mask arm. Every learned row binds the raw INT8 output
SHA, quantization metadata, argmax SHA, full adapter score grid, unknown,
derived-known, margin, and normalized-entropy diagnostics.

The independent validator re-ran the first frame of every parent event for
every arm: 90 model canaries and 30 oracle canaries. It independently checked
raw tensor SHA, quantization, softmax/pooling values, all 7,680 identities,
fold selection, event aggregation, and the terminal. Validation passed.

Two pre-outcome amendments are explicit in the contract:

1. an unsupported LiteRT `close()` call was removed after no trace/report had
   been written;
2. review found the initial fold formula would produce `8/8/6/4/4`, so offsets
   were frozen to produce `6/6/6/6/6` before any event output was written or
   opened.

After report creation, the independent validator numerically reproduced the
result but omitted the non-computational `lateral_profile` name from serialized
candidate metadata. A separate validation-only amendment restored that uniquely
derivable name; no tensor, score, fold, threshold, metric, gate, report, or
trace changed.

Evidence identities:

- contract SHA-256:
  `1b72032383c0d674d7db67b5c031f7987e370276e154e2ea820b6b520773ef43`
- feature trace SHA-256:
  `a817b4b63fe61fe91864bc060ea1cc8007059cda4c983c11b0678ee466465260`
- report SHA-256:
  `697975b7468141c0d2e34af365b1732b06ed89a25084584808d2ad1b574c018d`
- validation SHA-256:
  `5ec2130fb4ff19e9af3b98d33ce1c14602933424551c5af1cd4364f658a8db37`

## Frozen next action

P0 closes the “current four-class mask contains enough actionability; only the
argmax adapter hid it” hypothesis. It does not prove segmentation is generally
useless.

The next proposal must change the supervision target before training—for
example, explicit intrusion/actionability, alertable-vs-passed state, or
boundary crossing geometry at the parent-event level—and must define how those
labels can be produced without reading model outputs. A new session-disjoint
event cohort is required before any confirmatory comparison. Until then:

```text
KEEP_YOLO_BASELINE
RISKSEG_R1_P1_NOT_AUTHORIZED
DEFAULT_APP_UNCHANGED
```
