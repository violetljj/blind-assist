# DTR-C30 confidence + temporal-consistency motion authority

## Decision

C30 is a real algorithmic improvement over M1-PDC on the sealed C25
development cohort, but it misses the preregistered dropout gate by one sample.
Accept
`DTR_C30_CONFIDENCE_TEMPORAL_CONSENSUS_DEVELOPMENT_GATE_NOT_MET` and retain the
candidate as the next representation baseline.  Do not claim confirmation and
do not tune its scalar constants on C25.

## What changed

C28 showed that ray visibility supplies a causal absence state but cannot prove
motion identity.  C30 therefore exposed all current detector-independent M1-PD
residual cells to a truth-blind authority rule.  Each row carries reciprocal
direct velocity, point support, confidence, displacement/velocity consistency,
visibility state, local position, and observed-lineage context.  The route-risk
threshold and event lifecycle stayed frozen.

The retained rule grants authority only when:

- reciprocal direct motion has nonzero flow support, at least two source points,
  and sufficient confidence;
- current motion is consistent with remembered displacement and velocity; and
- a nearby point has compatible velocity, or an observed lineage supports the
  same local motion; weak isolated flow cannot alert.

This is point-wise confidence + temporal consistency + local rigidity before
target/occupancy aggregation.  `HIT` remains occupancy rather than mover
identity, and `OCCLUDED` remains an explanation rather than a velocity grant.

## Frozen search

The search used local `E:\SkyDiscover` at
`c475ed4009071159b4d5b777715f1af9202cebba`, AdaEvolve Pareto mode, 16 frozen
iterations, and `E:\codex-tools\bin\codex.exe` (`codex-cli 0.149.1`, ChatGPT
authenticated).  Candidate code could see only current/past causal row fields;
sequence id, frame id, timestamps, labels, future truth, and answer tables were
not exposed.  The evaluator reproduced M1-PDC exactly before launch.

| candidate | CONTACT recall | false segments | event F1 | median first lead | induced dropout recovery | every event no later than PDC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| M1-PDC baseline | `12/12` | 21 | 53.33% | 1.624 s | `25/36` | yes |
| raw residual reachability | `11/12` | 29 | 42.31% | 3.067 s | **`33/36`** | no |
| C30 retained consensus authority | **`12/12`** | **20** | **54.55%** | **1.734 s** | `29/36` | **yes** |
| best recovery Pareto arm | `12/12` | 29 | 45.28% | 2.869 s | **`33/36`** | yes |
| frozen gate | `12/12` | `<=21` | -- | no later | `>=30/36` | yes |

The retained candidate reduces false segments by `1` (`-4.8%`), raises event
F1 by `1.21` points, improves median lead by `0.109 s`, and recovers four more
dropouts than M1-PDC while preserving all events.  It is one dropout short of
the frozen gate.  In contrast, granting broad raw residual authority reaches
`33/36` but recreates the eight added false segments.  The information source
is therefore sufficient; selective motion authority is the remaining problem.

## Next falsifier

Do not resume this checkpoint or sweep confidence/locality constants.  The next
candidate should add a causal short-window component/occupancy state over the
retained point decisions: accumulate compatible velocity evidence, preserve
component identity across a brief occlusion, and revoke it on known-free or
velocity disagreement.  Its smallest falsifier is whether it recovers the one
missing induced-dropout sample without adding a false segment or delaying any
event on the same development trace; confirmation still requires fresh data.

## Evidence boundary

These are consumed-cohort Development results on five fresh JRDB sequences,
3,358 frames, 12 annotated contact events, and 130.98 seconds of known
non-contact exposure.  AdaEvolve explored causal program structure, but the
reported candidate remains selected on this cohort.  It is not a fresh-test,
universal, production, or safety claim.

Raw residual authority trace SHA-256 is
`ed7dff668ed679e759a12eb59c8df8566f54c6c01c54daeb75755093fb9cf2ee`;
16-iteration stats SHA-256 is
`5f2f0dcde4cd1f62687505dc43b78c47b6237ddbf4d6a9da42c99af1417f3bce`;
retained executable candidate SHA-256 is
`7c6847ca1c7e49b3dcb7f0a45be4d70fc436e038c4547dec1e0010ce8baa05b5`.
