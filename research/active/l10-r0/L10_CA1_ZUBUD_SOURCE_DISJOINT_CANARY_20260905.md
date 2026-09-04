# L10-CA1-C1 — source-disjoint commitment canary

Date: 2026-09-05
Authority: source-disjoint real-image commitment canary; not SEVN episode confirmation

## Frozen question

Keep the consumed 24 SEVN episodes, Triggered action policy, OCR, and candidate generation frozen. Test only whether the previously frozen 3D-Street-View center-scale DINO score can act as a ranking-independent, single-reference physical-instance witness before `COMMIT`.

The official ZuBuD query-to-building mapping supplies 115 source-disjoint real-image queries, 201 building identities, and five database views per identity. A strong proposal arm selects the top building by maximum global-DINO similarity over its five views. CA1 does not change that proposal: it compares the query only with the proposed building's fixed `view01` and returns `UNKNOWN` below the inherited threshold `0.611806`.

No ZuBuD threshold, crop, weight, view, or cohort selection was tuned after pixels were observed.

## Result

| Arm | Correct | Wrong | UNKNOWN | Commit precision | Coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| COMMIT_ALL | 110 | 5 | 0 | 95.65% | 100.00% |
| CA1_UNARY_GATE | 104 | 1 | 10 | 99.05% | 91.30% |

The frozen witness converted 4/5 wrong commits to `UNKNOWN` while retaining 104/110 correct commits:

- wrong-commit reduction: **80.0%**
- correct-commit retention: **94.5%**
- commit-precision change: **+3.40 percentage points**
- coverage change: **-8.70 percentage points**

The only surviving wrong commit is query 64 (`truth=13`, `proposal=199`, score `0.619501`). Six correct proposals also fell below the frozen threshold.

## Decision

The preregistered composite gate is **not met** because it required a 10-point absolute precision increase. The observed COMMIT_ALL precision was already 95.65%, so at most 4.35 points were available. This criterion cannot be rewritten after the result.

Therefore:

- preserve this as a strong positive mechanism signal that independent commitment authority can turn wrong decisions into `UNKNOWN`;
- do **not** promote this exact inherited threshold into a new SEVN episode under the preregistered decision rule;
- do **not** tune or rerun on these 115 queries;
- keep facade/portal continuity as the preferred CA1 evidence family for the next independently admitted cohort.

## Source admission note

The official ShopSign sample was inspected first but rejected: its released sample contains images and text boxes without an auditable same-sign pair mapping for the relevant image subset. Inferring pairs visually would make the tested appearance mechanism define its own truth. ZuBuD was admitted because its official query-to-building ground truth is explicit.

## Claim boundary

This result is evidence about building-instance commitment rejection on a source-disjoint provider. It does not establish address-token binding, facade or portal continuity, entrance identity, closed-loop SEVN improvement, arrival, handoff, access, traversability, user benefit, or mobility safety. The 24 SEVN episodes remain frozen Development/failure-analysis evidence.
