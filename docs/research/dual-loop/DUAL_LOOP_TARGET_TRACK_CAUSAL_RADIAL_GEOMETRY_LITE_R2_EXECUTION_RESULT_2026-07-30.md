# Dual-loop causal radial geometry LITE R2 execution result

## Terminal

`BOTH_NOT_READY_FOR_CONFIRMATION / IMPLEMENTATION_NOT_READY`

Execution date: 2026-07-30

Evidence version: `DUAL_LOOP_TARGET_TRACK_CAUSAL_RADIAL_GEOMETRY_LITE_R2`

Claim ceiling: `SINGLE_CAPTURE_ORACLE_ROI_CONDITIONED_DEVELOPMENT_ONLY`

Development truth joined: `true`, evaluator only after every pre-truth gate passed

Confirmation execution authorized: `false`

Producer rerun authorized: `false`

Evaluator rerun authorized: `false`

The one-shot R2 producer and the conditional one-shot evaluator both completed
validly. Neither arm met the preregistered readiness floor. Sparse radial flow was
also worse than the bounding-box area-growth baseline on the frozen 469-event
denominator, so the flow-over-bbox gate failed. This is a valid negative Development
result, not an execution failure and not evidence for effectiveness, runtime
deployment, product improvement or safety.

## Bound evidence chain

- Activation SHA-256:
  `af4d02dab470787b7f13cd9940d4b04296b676d3df2a091c962d353666c960fd`.
- Activation review SHA-256:
  `c48d94f7f514800817d258a598399c071d146429f474efc934ba040ea6c830ea`;
  terminal `ACTIVATION_REVIEW_PASS`.
- Implementation lock SHA-256:
  `c2ba9a2733fd4e6c8529421240348e6b0593d65dd1b44a154cfbb15deb60f7fe`.
- Scientific gate contract SHA-256:
  `bfcb3f009dec726055e762857b3e7e0159190d9e21f583d959f477055e82d4b0`.
- Guarded-run summary SHA-256:
  `2e6d9b764140ba3b303972650ab197e6198a8e1ffc1cc1e0c8e1fe09c07b595e`;
  terminal `COMPLETE`, process exit code `0`, progress contract valid.
- Producer receipt SHA-256:
  `8e8647ada30b2bb1ddb4ed5c176eb6bf71dd75111efed4a3d0422b0eea1d8a3c`.
- Producer progress SHA-256:
  `f60d155aaf3821582cd9ade405f38374502dd09826f548199d81e301885427f1`;
  terminal `complete`, 13,014 / 13,014 units.
- Producer output SHA-256:
  `86509eccc62eff4ed2d89a8b6956e6a5d532d32e4aa38490a1f88b29dd144baa`.
- Evaluation SHA-256:
  `bb2d52a478ebd897c51d229344d653c5bd8736317fad6343ca8a27b0589b7f7e`.
- Frozen replay, truth and natural-event SHA-256 values are respectively
  `14f1f7f0f330d8b01146e37c31505240f3f0e8d301846ebcad44a628948e6440`,
  `dfe53a8be2a18c38f9b2bc2d715290296433f15302ea64911e21ee956423003f`
  and `078881620709efe17f74b8b01a5a76f4e861bfb6363143b9a9e0a589a87a030a`.

The producer reports 13,014 inputs and 26,028 outputs: one
`BBOX_LOG_AREA_GROWTH` row and one `ROI_SPARSE_RADIAL_FLOW` row for every input.
It records 32 source shape-change opportunities and 64 common arm rows, exactly
matching the frozen source-shape audit. The failure receipt is absent and
`truth_joined=false` in the producer receipt. Development truth and natural events
were opened only by the evaluator after those facts and hashes were independently
revalidated.

## Frozen runtime-facing output

Every producer row retains the frozen:

- `target_id`, `region`, `signed_approach_rate_per_s`;
- structured `quality`;
- `ttl_ns` and `valid_until_ns`;
- `abstention_reason`;
- capture, availability, source-frame, track-epoch, arm and implementation
  identities.

The area-growth arm emitted 12,844 finite and 170 abstained frame rows. The sparse
radial-flow arm emitted 12,735 finite and 279 abstained frame rows. Event evaluation
used the frozen coverage rule; both arms remained evaluable on all 469 primary
natural events. Each estimate consumed only the current and immediate previous
frame; no future frame was consumed by either arm.

## Primary natural-event result

| arm | correct | correct fraction | wrong-signed | wrong-signed fraction | evaluable | readiness |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| box log-area growth | 204 / 469 | 0.4350 | 153 / 469 | 0.3262 | 469 / 469 | fail |
| ROI sparse radial flow | 188 / 469 | 0.4009 | 161 / 469 | 0.3433 | 469 / 469 | fail |

The preregistered readiness floor required correct fraction at least 0.60,
wrong-signed fraction at most 0.20, evaluable fraction at least 0.80 and correct
fraction at least 0.50 in every truth state. Both arms failed the first two overall
requirements. Area growth also failed the quasi-static truth-state floor; radial
flow failed all three truth-state floors.

## By target

| arm | target | correct | correct fraction | wrong-signed | wrong-signed fraction |
| --- | --- | ---: | ---: | ---: | ---: |
| box | `track-000` | 103 / 234 | 0.4402 | 72 / 234 | 0.3077 |
| flow | `track-000` | 99 / 234 | 0.4231 | 73 / 234 | 0.3120 |
| box | `track-001` | 101 / 235 | 0.4298 | 81 / 235 | 0.3447 |
| flow | `track-001` | 89 / 235 | 0.3787 | 88 / 235 | 0.3745 |

Flow correct-event gains relative to box were `-4` on `track-000` and `-12` on
`track-001`; the contract required a positive gain on both.

## By anchor region

| arm | region | correct | correct fraction | wrong-signed | wrong-signed fraction |
| --- | --- | ---: | ---: | ---: | ---: |
| box | CENTER | 97 / 205 | 0.4732 | 58 / 205 | 0.2829 |
| flow | CENTER | 92 / 205 | 0.4488 | 63 / 205 | 0.3073 |
| box | LEFT | 63 / 145 | 0.4345 | 54 / 145 | 0.3724 |
| flow | LEFT | 59 / 145 | 0.4069 | 51 / 145 | 0.3517 |
| box | RIGHT | 44 / 119 | 0.3697 | 41 / 119 | 0.3445 |
| flow | RIGHT | 37 / 119 | 0.3109 | 47 / 119 | 0.3950 |

Flow correct-event gains were CENTER `-5`, LEFT `-4` and RIGHT `-7`; the contract
required positive gain in at least two regions.

## By Development truth state

| arm | truth state | correct | correct fraction | wrong-signed | wrong-signed fraction |
| --- | --- | ---: | ---: | ---: | ---: |
| box | approaching | 105 / 200 | 0.5250 | 82 / 200 | 0.4100 |
| flow | approaching | 95 / 200 | 0.4750 | 88 / 200 | 0.4400 |
| box | quasi-static | 4 / 92 | 0.0435 | 0 / 92 | 0.0000 |
| flow | quasi-static | 6 / 92 | 0.0652 | 0 / 92 | 0.0000 |
| box | receding | 95 / 177 | 0.5367 | 71 / 177 | 0.4011 |
| flow | receding | 87 / 177 | 0.4915 | 73 / 177 | 0.4124 |

## Flow-over-box gate

The frozen comparison gate failed:

- correct-event gain was `-16`, versus required minimum `+2`;
- target gains were negative on both targets;
- region gains were negative in all three regions;
- wrong-signed events increased from 153 to 161, while the contract forbade an
  increase;
- evaluable-event loss was zero and the distinct-event dominance guard passed,
  but these non-primary safeguards cannot rescue the failed directional gates.

No post-result threshold adjustment, subgroup rescue, arm mutation or denominator
change is allowed.

## Stop and authority

- Accept `BOTH_NOT_READY_FOR_CONFIRMATION / IMPLEMENTATION_NOT_READY` unchanged.
- Do not rerun R2 producer or evaluator.
- Do not promote either arm to Confirmation or production.
- Do not inspect the old F-1B decision set; it remains sealed.
- Do not infer Android feasibility, device latency, field effectiveness, user
  benefit, product improvement or safety.
- Any later successor must be a new, prospectively frozen Development question
  with independent inputs or a materially different preregistered mechanism; R2
  output cannot be tuned against and relabelled as fresh evidence.
