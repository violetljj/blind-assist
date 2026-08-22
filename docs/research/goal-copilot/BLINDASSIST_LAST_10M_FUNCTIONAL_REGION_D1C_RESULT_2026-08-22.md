# BlindAssist Last-10m functional-region D1C result (2026-08-22)

## Outcome

This development milestone did not authorize a fresh formal successor. It did establish a usable current-frame
metric-depth ground-plane representation and narrowed the remaining failure to semantic/functional candidate recall.
All collection, qualification, truth derivation, model execution, and adjudication were automated from public data.

The claim ceiling is:

`DEVELOPMENT_ONLY_SYNTHETIC_EXACT_DOOR_GROUND_CONNECTED_APPROACHABILITY_PROXY_NO_REAL_WORLD_ENTRANCE_TRAVERSABILITY_NAVIGATION_PRODUCT_OR_SAFETY_CLAIM`.

## Contract correction

The S2-S5 private evaluator contains exact `door` segmentation components, bboxes, and metric depth. It does not
contain open-aperture state, room connectivity, a navigation mesh, or proof that a door is an entrance. Therefore the
earlier wording `OPEN_APERTURE` was broader than the available truth. None of the results below are traversability
evidence.

## Automated cohort qualification

The public goal contract was frozen before truth as `帮我找一扇通向建筑内部的门` with canonical prompt `door`.
Environment selection and denominator qualification happened before RGB/provider access:

- Supermarket: remote random-access ZIP transport stalled; no roster or provider result (`NOT_EVALUABLE_TRANSPORT`).
- DesertGasStation D1B: `near=4, far=408`; failed the required 24/24 denominator before roster/provider calls.
- HongKong D1C: `eligible=1766, near=112, far=1654`; passed, then mechanically selected 24 near + 24 far cases.

D1C is development-only and cannot serve as independent confirmation after it was used for model/rule selection.

## Algorithm results

The deterministic constrained-RANSAC depth module fitted all 48 Office development frames. Against private
floor+carpet segmentation in the lower image half, aggregate precision was `0.94105`, recall `0.78209`, and IoU
`0.74556`. This supports the narrow statement that metric ground plane is observable on this synthetic development
data.

The completion gate was predeclared as zero false completion, at least eight opportunities, and correct completion
coverage at least 0.50. D1C had 24 opportunities.

| Provider / verifier | Correct | False | Coverage | Gate |
|---|---:|---:|---:|---|
| YOLOE functional mask | 6 | 16 | 0.250 | fail |
| YOLOE bbox diagnostic | 13 | 9 | 0.542 | diagnostic only |
| frozen context-v2 fusion | 11 | 6 | 0.458 | fail |
| SAM 3 `door`, confidence 0.50 | 6 | 0 | 0.250 | fail |
| SAM 3 proposal floor 0.10 | 13 | 1 | 0.542 | fail |
| best zero-false point in fixed 0.10:0.05:0.50 grid | 10 | 0 | 0.417 | fail |

SAM 3 materially improved precision, but the zero-false operating point did not achieve the required recall. The
observed 0.105 boundary is a development-derived candidate only; it requires a fully untouched D2 environment before
any successor can be authorized.

## Decision

`D1C_SAM3_FUNCTIONAL_REGION_DEVELOPMENT_ONLY_NOT_CONFIRMED`.

Do not rerun D1C as fresh evidence, do not restore AMRM/identity, and do not claim real-world entrance or navigation
success. The next permitted action is an automated untouched-environment confirmation of the frozen SAM 3 plus
metric-ground rule, followed by a fresh cohort only if that confirmation passes.
