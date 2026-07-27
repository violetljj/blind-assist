# RCLE RGB algorithm / CID-SIMS floor3_1 disjoint geometry-stratified holdout R0

Date: 2026-07-27

Protocol:
`RCLE_RGB_ALGORITHM_CID_SIMS_FLOOR3_1_DISJOINT_GEOMETRY_STRATIFIED_HOLDOUT_R0`

Terminal: `GEOMETRY_STRATIFIED_WINDOWS_NOT_EVALUABLE / VALID`

Authority: `SAME_SEQUENCE_DISJOINT_WINDOW_DEVELOPMENT_HOLDOUT_ONLY`

## Answer

This evidence version cannot answer whether the unchanged RGB algorithm
reproduces the Pairwise R0 direction separation in disjoint windows.

The geometry-only phase found seven eligible positive-approach windows and
zero eligible below-trigger-reference windows. The frozen exact
`2 positive + 2 below-reference` joint selection therefore had no feasible
set. The run stopped before selected-RGB identity creation:

- selected windows: `0`;
- selected RGB identity manifest: absent;
- RGB cache: absent;
- RGB pair ledger: absent;
- RGB member bytes read: `0`;
- RGB algorithm executed: `false`;
- threshold changes or replacement windows: `0`.

This is a valid `NOT_EVALUABLE`, not evidence that the RGB direction failed.
It also is not cross-source confirmation or performance qualification.

## Frozen geometry-only selection

The 20 s inspected interval was W0–W1. W2 was excluded as the frozen 10 s
adjacent guard band. Candidate windows were the fixed, non-sliding W3–W11
grid. Any selected windows would additionally need start times separated by
at least 20 s.

Role eligibility used the full fixed 299-pair denominator:

- positive: geometry coverage at least `0.80`, positive-band fraction at
  least `0.80`, and longest positive run at least `5.0 s`;
- below reference: geometry coverage at least `0.80`, below-reference
  fraction at least `0.80`, and longest below-reference run at least `5.0 s`;
- exact joint selection: two of each role, lexicographically earliest feasible
  index tuple; no partial run, sliding, replacement, imputation or gate change.

Only depth PNG, bound `groundtruth.txt` pose and source timestamps were used
to compute the candidate ledger. ZIP color-member identity metadata had been
available for archive bookkeeping, but no candidate RGB member bytes or RGB
algorithm outcomes were read.

## Candidate results

| Window | Frames / pairs | Geometry evaluable | Positive fraction | Below fraction | Longest positive run | Median signed radial expansion | Frozen role |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| W3 | 300 / 299 | 299 / 299 | 1.0000 | 0.0000 | 9.957 s | 0.538449/s | positive |
| W4 | 300 / 299 | 299 / 299 | 1.0000 | 0.0000 | 9.977 s | 0.436748/s | positive |
| W5 | 299 / 298 | 0 / 299 fixed | 0.0000 | 0.0000 | 0.000 s | — | identity-ineligible |
| W6 | 300 / 299 | 299 / 299 | 1.0000 | 0.0000 | 9.966 s | 0.155762/s | positive |
| W7 | 300 / 299 | 299 / 299 | 1.0000 | 0.0000 | 9.970 s | 0.407285/s | positive |
| W8 | 300 / 299 | 299 / 299 | 1.0000 | 0.0000 | 9.975 s | 0.515608/s | positive |
| W9 | 298 / 297 | 0 / 299 fixed | 0.0000 | 0.0000 | 0.000 s | — | identity-ineligible |
| W10 | 300 / 299 | 299 / 299 | 1.0000 | 0.0000 | 9.977 s | 0.330448/s | positive |
| W11 | 300 / 299 | 299 / 299 | 1.0000 | 0.0000 | 9.964 s | 0.470538/s | positive |

The seven eligible windows contributed `2093/2093` evaluable geometry pairs
with no geometry abstention. W5 and W9 were not decoded for geometry because
their source timestamp counts failed the frozen exact 300-frame identity
rule. They were not repaired, resized, shifted or used as replacements.
Their machine summaries record `geometry_abstention_count=299` against the
fixed expected denominator while leaving `geometry_abstention_reasons={}`.
This means identity-failed expected pairs were not evaluated; it must not be
read as 299 pair-level algorithm abstentions.

## Independent validation

The validator does not import the formal runner or RGB producer. It
independently reconstructed:

- W3–W11 timestamp identity and the inspected-plus-guard exclusion;
- geometry ledger pair identity, order, band assignment, run lengths,
  coverage, role aggregates and 2+2 selection feasibility;
- contract, implementation lock, immutable algorithm bindings, ledger,
  selection and result hashes;
- the absence of forbidden RGB identity/cache/ledger artifacts after the
  geometry `NOT_EVALUABLE` terminal.

Validation returned `VALID` with `errors=[]`. Because RGB never ran,
independent RGB cache/ledger/aggregate recomputation is correctly `false`.

## Learning record

### Observation

All seven identity-eligible disjoint candidates are sustained positive
approach under the frozen geometry rule; no below-reference role exists.

### Supported inference

The remaining `floor3_1` interval does not contain the within-sequence role
contrast required by this R0. More windows from the same fixed grid add
positive duration but not the missing low-reference stratum.

### Alternative explanations

- The exact 300-frame identity rule excludes W5 and W9; this R0 cannot infer
  their role.
- Source pose/depth synchronization and the global radial estimator remain
  imperfect measurements rather than calibrated collision truth.
- Temporal windows belong to one sequence and are not independent trials.

None authorizes changing the consumed R0 rule or reading W5/W9 RGB as rescue.

### Challenged constraint and information gained

The original successor assumed one sequence might supply both positive and
low-reference disjoint windows. The formal result falsifies that data-shape
assumption for the frozen W3–W11 grid. It preserves the stronger positive
geometry episodes as source characterization and future regression fixtures,
but it does not spend RGB outcome authority on an unbalanced comparison.

### Reuse and next falsifiable hypothesis

- Reuse W3/W4/W6/W7/W8/W10/W11 only as burned same-sequence positive
  geometry characterization or regression fixtures.
- Preserve W5/W9 as identity-ineligible under R0; do not retrofit them.
- A future, separately versioned test needs an independently frozen source or
  sequence that demonstrably contains both roles before any RGB outcome
  access. That would no longer be this `floor3_1` disjoint-window R0 and would
  need its own authority and claim.

## Authority limits

- The candidate geometry grid is disjoint from the inspected 20 s interval
  and its 10 s guard, but `floor3_1` had prior Discovery-level geometry
  characterization. The honest firewall is RGB-outcome-blind selection, not a
  claim that the entire sequence geometry was historically unseen.
- No RGB outcome was produced, so window-level direction reproduction,
  positive trigger coverage, low-reference false trigger, RGB/geometry onset
  delta and RGB abstention remain `NOT_EVALUABLE`.
- Pairwise R0 remains an outcome-aware posthoc mechanism alignment result.
- This result cannot answer cross-sequence generalization, performance,
  Android behavior, human efficacy, product readiness or safety.

## Execution receipt

The first guarded launcher process exited before claim creation because its
default bare Python lacked `cv2`; it created no formal output or progress and
did not access candidate data. The same locked script was then launched
through the project’s verified venv. The canonical run created one exclusive
claim and completed in `13.968 s` with eight workers.

- Contract SHA-256:
  `7518527e80df589863c9d317fa8581e6d8926ade8c85e00bc0dd77a4495c2011`
- Implementation lock SHA-256:
  `6b6e95a4d1cc765e22d09b5c2423f2d6a5aa02d66938a2c9927a44de5ba8228b`
- Claim SHA-256:
  `a708e6b4df7f7de17de2a7438f8b2e447a159808af71c4af0ab174c2104ace13`
- Geometry ledger SHA-256:
  `78530a0cee19dce35ee1ed248eea487683215918903293a697e16baea25205c1`
- Selection SHA-256:
  `e8db1776b5c00f194d98008f753ade5b06073f31c2675c98388a157ad60b4245`
- Result SHA-256:
  `6d66d6686d6624c39d6ef1de7cdd509b63daf1e3d832258dd41d8051d62144d0`
- Validation SHA-256:
  `e04b78d48ab6c411a1c6badd2ed8c7100d8c37d60330ec2dd963908c05fcab4e`
- Success SHA-256:
  `f954be69bc7ab66b3d8bdadc201ec0280e80b467bcf474f1662a6edc510dcbc1`
