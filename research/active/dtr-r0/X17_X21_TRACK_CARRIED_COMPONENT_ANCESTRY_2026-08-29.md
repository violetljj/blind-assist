# DTR X17-X21 track-carried component ancestry

## Decision

Accept `DTR_X21_TRACK_CARRIED_COMPONENT_ANCESTRY_GATE_MET` on the already
opened six-sequence Development cohort. X21 is the first arm in this route to
meet every frozen check:

| Arm | CONTACT | False segments | Event F1 | Median lead | Dropout recovery |
|---|---:|---:|---:|---:|---:|
| M1-PDC baseline | 4/6 | 25 | 22.86% | 2.291 s | 5/18 |
| X15 RGB-authorized continuation | 5/6 | 18 | 34.48% | 3.061 s | 2/18 |
| X17 tracked-instance continuation | 5/6 | 33 | 22.73% | 4.922 s | 14/18 |
| X18 X15-seeded track bridge | 5/6 | 33 | 22.73% | 4.922 s | 14/18 |
| X19 raw-X13-seeded track bridge | 5/6 | 31 | 23.81% | 4.922 s | 12/18 |
| X20 component/track gap closure | 5/6 | 16 | 37.04% | 3.785 s | 2/18 |
| **X21 track-carried component ancestry** | **5/6** | **11** | **45.45%** | **3.061 s** | **8/18** |

The frozen gate required CONTACT `>=5/6`, false segments `<=16`, Event F1
`>=35%`, median lead `>=2 s`, dropout recovery `>=5/18`, and fewer false
segments than PDC. X21 passes all six. Relative to PDC it adds one recalled
CONTACT, removes 14 false segments, raises Event F1 by 22.60 percentage points,
and adds `0.769 s` median lead. Relative to X15 it removes seven false segments,
raises Event F1 by 10.97 points, and raises dropout recovery from `2/18` to
`8/18` without losing CONTACT recall.

This is Development promotion only. Freeze X21 and move to a source-disjoint
confirmation cohort; do not tune the consumed six sequences.

## Structural result

X17 proved that learned instance-track persistence carries useful continuity:
it reached `14/18` dropout recovery and `4.922 s` median lead. It was not
selective enough. Once a YOLO11 track was authorized, its whole current mask
could admit unrelated X7 components, producing 33 false segments.

X18 changed only the seed to sealed X15 cells. The dense X15 continuation
seeded nearly the same live tracks and was score-equivalent to X17. X19 then
restricted the seed to recomputed raw X13 births, but still allowed mask-wide
absorption. It reduced false segments only from 33 to 31. Read-only attribution
found that 30 of X19's 31 false segments overlapped X17 and all 31 were started
by raw route entry, not lifecycle HOLD. The failure was track-level authority
amplification, not insufficient seeding quality.

X20 changed the admitted unit from an instance mask to exact ancestry
`(class_id, track_id, sealed-X7 component_id)`. A current X7 cell could continue
only when the same component ID remained inside the same live track mask. That
restored selectivity: false segments fell to 16 and Event F1 rose to 37.04%.
It did not improve induced dropout because exact current-X7 component support
disappeared in the target gaps.

X21 preserves X20's only legal entry: a raw X13 birth cell inside a current
YOLO11 track mask. The state stores that cell row and its origin component. On
the next frame it transports only that existing row with the frozen X14 ego and
velocity operator. The transported anchor must remain inside the same current
track mask. X21 never absorbs a new current-X7 cell from the mask. Image, pose,
track-ID, or current-mask break deletes the state immediately. Emitted rows
then use the unchanged X14 `0.50 s` continuation and unchanged X3 route,
lifecycle, and scorer.

That representation separates the two properties the earlier arms traded off:
X13 birth supplies decision selectivity; the learned instance track supplies
identity persistence; component ancestry prevents mask-wide authority growth.

## Sealed evidence

- Timeline: 4,811 frames across six sequences.
- X21 freeze SHA-256:
  `23be56f9bf471ce3dabc331b05cd120b4f8c0ac72a16028b3c12452a10e895e0`.
- X21 materialization: 301,483 X7 input-cell occurrences, 36,277 raw X13
  birth-cell occurrences, 226,899 authorized occurrences, 194,758 of them
  track-transported, and 1,764,048 occurrences after unchanged X14
  continuation.
- Predictions SHA-256:
  `f52b21c14672d6a91812e6e54649df5c549bd5c001dbbe8af0acd57b5a182863`.
- Materialization SHA-256:
  `96b8de339487474ddcb8cfb0022ac9c5dfb0f0ba3103ca8cf3aca3359bb4da40`.
- Result SHA-256:
  `7775fd16bc8fa3fb41ad005fe86d5d06731e10e6fdcbcf8701794c26cffaf690`.
- Frozen model: `yolo11n-seg.pt`, SHA-256
  `55ed65c56c91713d23e8402371c6c49a6fd84f257f7dce452e8d70e41dcbe152`.
- Backend: Python 3.11.9, OpenCV 4.10.0, verified CUDA execution.

Per-sequence X21 score:

| Sequence | CONTACT | False | Lead | Dropout recovery |
|---|---:|---:|---:|---:|
| huang-2 | 1/2 | 1 | 3.061 s | 3/6 |
| huang-basement | 1/1 | 1 | 4.205 s | 0/3 |
| huang-lane | 2/2 | 2 | 2.826 s | 3/6 |
| memorial-court | 1/1 | 5 | 18.245 s | 2/3 |
| meyer-green | 0/0 | 2 | n/a | 0/0 |
| tressider | 0/0 | 0 | n/a | 0/0 |

## Boundaries and next action

- The six sequences are consumed Development evidence. X21 has not passed
  source-disjoint confirmation.
- Labels are hashed for frozen provenance before prediction but are not parsed
  or used by materialization/prediction. Native truth is parsed only by the
  unchanged scorer.
- YOLO11 is an RGB representation and is not sensor-disjoint from X13 RGB
  motion.
- Induced dropout recovery is a frozen continuity guardrail, not proof of real
  physical occlusion robustness.
- This result does not establish real-device latency, product benefit, safety,
  or cross-source generalization.
- Do not sweep X13 confidence, YOLO model/class/confidence, tracker settings,
  component rules, continuation duration, route, lifecycle, or scorer on this
  cohort. The next legal action is one frozen source-disjoint X21 confirmation.
