# DTR-C31 source-disjoint fresh confirmation

## Decision

The frozen C31 signed-transport mechanism did not confirm on source-disjoint
JRDB data:

`DTR_C31_SOURCE_DISJOINT_CONFIRMATION_GATE_NOT_MET`.

Do not tune C31's component radius, confidence, decay, transport evidence,
occlusion duration, route, or lifecycle on this cohort. Do not open C32
probabilistic body-route occupancy on the current C31 support: the confirmation
did not establish component-level precision or timing headroom for uncertainty
propagation.

## Frozen confirmation

The metadata admission initially retained all seven remaining JRDB train
sequences that had not been exposed to the earlier algorithm line. A
truth-blind raw-source preflight then found one sequence structurally
unreachable: `gates-to-clark-2019-02-28_1` has no current-or-past native pose at
frame 0. It is `NOT_EVALUABLE`, not an algorithm negative.

The final frozen cohort contains six sequences, 4,811 frames, six bounded
CONTACT events, 18 induced dropout trials, and 278.08 seconds of known
non-CONTACT exposure. All raw bags and ledgers were hash checked. C28
visibility and C30 raw-residual traces were materialized without labels. The
M1-PDC, C30, and C31 predictions plus dropout ledgers were sealed before future
OBB truth and evaluator identity were opened once for scoring.

The gate required all six CONTACT events, no lower recall or more false
segments than M1-PDC, no later first alert for every PDC-recalled event, and
either at least two additional dropout recoveries (10% of 18, rounded up) or at
least 0.25 seconds additional median lead.

## Result

| arm | CONTACT recall | false segments | Event F1 | median first lead | dropout recovery |
| --- | ---: | ---: | ---: | ---: | ---: |
| M1-PDC | `4/6` | **25** | **22.86%** | **2.291 s** | `5/18` |
| C30 local consensus | `4/6` | 27 | 21.62% | 2.291 s | `5/18` |
| C31 signed transport | `4/6` | 35 | 17.78% | 2.291 s | `6/18` |

C31 preserved every PDC-recalled event with equal lead, but it did not recover
either PDC-missed event. Its only continuity gain was one additional dropout
recovery on `memorial-court-2019-03-16_0`, below the frozen `+2` requirement.
False segments increased by ten versus PDC and eight versus C30.

## Failure localization

The two missed events are unchanged from PDC:

- `huang-2-2019-01-25_0:contact:001`;
- `huang-lane-2019-02-12_0:contact:001`.

The C31 false-segment changes versus PDC were `+2` on Huang-2, `+5` on
Huang-basement, `+2` on Huang-lane, and `+1` on Memorial. The two zero-CONTACT
sequences added no false segments. This is not a uniform background-noise
failure: local component extension becomes over-authoritative in complex
contact-bearing scenes without adding earlier route entry or new event recall.

Across the cohort, C31 exposed 80,559 motion-vote rows, authorized 15,400 new
component members, recovered 456 raw and 1,488 occluded rows, and synthesized
594 occlusion-support cells. Those large exposure counts yielded only one
additional induced-gap recovery and ten additional false segments. They are
diagnostics, not successes.

## Consequence

The consumed-cohort C31 win was not source-disjointly transferable. A
probabilistic tube that only spreads the same accepted components would add
uncertainty mass to support that is already too permissive; it is therefore not
the next admissible experiment. A successor must first change component
information or authority so the four false-inflating sequences can be
distinguished without losing the four recalled events. Candidate directions
may include independent track/residual agreement, richer scene-flow
uncertainty learned from fresh component truth, or another source-level
representation, but this result authorizes no threshold or model sweep.

## Receipts

- result SHA-256:
  `b894995b499a17ee2a03f95fe2ff74332b9a94ef173a8a54322f68f7efc999ef`;
- sealed predictions SHA-256:
  `5eaf1c255449a40b66262b541d70f12f38b32f5c0931f3d613918c4a5b90b50a`;
- truth-blind raw trace SHA-256:
  `cd6abf2d6a9a5efc288ba924137b53def0b5a4633b57c6ac578e44b9db502969`;
- source preflight SHA-256:
  `1f9560eff0ba19d5a7c97b0b45b82bdb2a699945ac3b710e48a53593d9b09c71`;
- frozen confirmation executable SHA-256:
  `a3cc0255d561cf2ab1a6ea41b7a1b1ea8d3809b78b14be6dd35d47c93a8943b3`;
- frozen roster SHA-256:
  `7de4f88719e3100812c0bbe8968129eb7b6572df2a1e3a276cf8372d35c10134`.

Ray-visibility batching selected real RTX 5060 CUDA with
`GPU_FASTER_OR_EQUAL_MEASURED`. Reciprocal point-lineage and C31 component
matching selected CPU with `CPU_FASTER_MEASURED`; no GPU fallback is described
as GPU execution.
