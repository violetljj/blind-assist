# TARO O0R ARKitScenes direct Apple hybrid R4A result

## Outcome

Terminal: `TARO_O0R_DIRECT_APPLE_HYBRID_R4A_COMPLETE`.

The threshold-free policy
`DIRECT_WHEN_SOURCE_SUPPORT_AVAILABLE_ELSE_R1_BASELINE_V1` closes the direct
branch's extraction regression on the full 171-frame cohort. It selects direct
Apple SUPPORT only when the source-only plane is physically valid and otherwise
falls back to the existing R1 baseline. The policy recovers 36 extraction-
evaluable queries over baseline and loses none.

Support-height and support-normal parent-macro errors both improve on all 16
parents. This is meaningful algorithmic headroom, but it is still a
same-cohort retrospective replay. It is not fresh confirmation, a formal O0R
PASS, or deployment, product or safety evidence.

## Policy and leakage firewall

The policy has zero fitted parameters, zero thresholds and zero training steps.
Its sole selection input is the R4 Phase-A boolean
`source_support_available`, derived before FARO or truth outcomes are opened.
It does not read extraction evaluability, error, no-regret, knownness or any
other Phase-B metric.

Each hybrid query record is bound to the exact externally verified R4 record.
The public validator requires that external record, and summary construction
requires an equal-length, row-for-row R4 sequence. The runner verifies the
frozen R4 result, manifest, summary, query blob and sequence hashes before it
creates the R4A evidence root.

## Descriptive result

| Metric | Result |
|---|---:|
| physical frames / parents | 171 / 16 |
| queries | 1,539 |
| direct-selected queries | 1,422 |
| baseline-fallback queries | 117 |
| fallback-saved evaluable queries | 108 |
| hybrid extraction-evaluable queries | 1,530 |
| baseline extraction-evaluable queries | 1,494 |
| extraction recovered / lost vs baseline | 36 / 0 |
| height-and-normal no-regret vs baseline | 1,111 queries |
| support height improved vs baseline | 1,212 queries |
| support normal improved vs baseline | 1,114 queries |
| hybrid boundary-evaluable queries | 121 |
| hybrid known point-clearance queries | 8 |

The median of parent-level median support-height error reductions was
`+0.285509691243 m`; the support-normal reduction was
`+0.022961694045 rad`. Height improved on 16/16 parents, normal improved on
16/16, and all 16 parents were jointly positive on both metrics.

Hybrid support coverage had a median parent coverage of `1.0`. Fifteen parents
had complete coverage; parent `469607` retained `297/306` evaluable queries.

One limitation remains visible: hybrid knownness lost two baseline-known
queries and recovered none, leaving 8 known point-clearance queries. The policy
therefore advances the SUPPORT/extraction hypothesis but does not yet establish
a final clearance-query improvement.

## Decision and next falsifiable step

Freeze R4A as the only live direct-Apple SUPPORT policy. Do not fit a selector
or tune a threshold on the 16 observed eval parents.

The unique successor is
`DIRECT_APPLE_HYBRID_ADAPTER_FIT_CONFIRMATION_R5_AMENDMENT`: independently
confirm the already frozen zero-parameter policy on the eight disjoint
`ADAPTER_FIT` parents / 211 existing source frames. A new amendment is required
because the original role contract allowed those parents for uncertainty fit,
not DepthART candidate output or task-metric evaluation. Phase A must seal all
truth-blind model/source outputs before Phase B opens the same parents' FARO
truth. The existing 16-parent eval truth must not be enumerated or read.

## Validation and reproducibility bindings

- implementation commit: `50c676c9`;
- focused R4A tests: `9/9` PASS;
- combined related tests: `36/36` PASS;
- summary content seal:
  `E15C48A34F7CA503A60C39D69AFD01538B1021048B8B1B1363A774E2F26EFB06`;
- evidence manifest file SHA-256:
  `F9260F32BEEA9B5AB749F3D8F67DEC83C1623F159D3B9BFA7A2BEDD52BEDF309`;
- result file SHA-256:
  `F2030EE00B5C0F63B5807D5DF6E361B1828C45462D079F59A482A4C62CB71684`;
- evidence root:
  `artifacts.local/evidence/taro/o0r-arkitscenes-direct-apple-hybrid-r4a`.

The manifest accounts for four pre-manifest files and 579,454 bytes. All file
hashes, 1,539 externally bound query records and the exact canonical summary
replay were verified.
