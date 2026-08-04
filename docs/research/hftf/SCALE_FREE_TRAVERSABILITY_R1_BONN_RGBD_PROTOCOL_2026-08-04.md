# Scale-Free Traversability R1 Bonn RGB-D Protocol

Date: 2026-08-04

Status: `FROZEN_BEFORE_CANDIDATE_OUTPUT_EXECUTION`

This round evaluates the already frozen R0 scale-free operator against registered
sensor depth in two public Bonn RGB-D Dynamic Dataset sequences. The sequences
were consumed by older, different project experiments, so their strongest role
is `PROJECT_CONSUMED_DEVELOPMENT`. Because R0 was frozen from the phone sessions
without reading these sequences, this is also an
`OPERATOR_UNSEEN_EXTERNAL_REPLICATION`; it is not globally fresh, sealed,
Confirmation, or safety evidence.

## Frozen inputs and sampling

- `rgbd_bonn_person_tracking.clean.zip`, SHA-256
  `A4810FD91EF2EA1D630B53FE0DF5D76144C1B18D86CA91FB3A035DEBD0C9C5F5`;
- `rgbd_bonn_person_tracking2.clean.zip`, SHA-256
  `D3EF7898529C60DC39919EA699D00490D98A2C6AE4B165610F2955B235B939B5`;
- the known `concurrent-write.corrupt.zip` is forbidden;
- associate every RGB row with its unique nearest depth timestamp within 0.02 s;
- select the first associated pair and then the first pair at least 0.2 s after
  the preceding selected pair, continuing to the end of each sequence;
- the source sequence, not a sampled frame, is the independent unit.

The candidate remains the exact R0 mechanics and checkpoint. RGB is the only
model input. Registered depth is decoded after candidate inference, divided by
5000 to metres, and passed through the same scale-free scoring and causal
decision mechanics to create reference labels. No threshold, ROI, percentile,
window, checkpoint, sampling rule, sequence, or gate may change after candidate
outputs are read.

## Frozen metrics and gates

Report per sequence and sequence-macro summaries. Truth score coverage must be
at least 0.50 and candidate execution coverage at least 0.95 in each sequence.
After each truth stream's four-frame warm-up, each sequence must contain at least
10 non-ambiguous truth directions.

For those truth-directional frames:

- recommendation coverage is the fraction on which the candidate also emits a
  direction and must be at least 0.50 in each sequence;
- directional accuracy is exact left/center/right agreement among candidate
  recommendations; its sequence macro must be at least 0.75 and the worst
  sequence at least 0.60;
- opposite-direction error is specifically left-versus-right disagreement; its
  sequence macro must not exceed 0.05.

Also report exact decision-label agreement, including `AMBIGUOUS`, as a
non-gating diagnostic. A separate validator must recompute all aggregates and
the terminal from the immutable frame ledger without importing the evaluator.

The positive terminal is
`SCALE_FREE_TRAVERSABILITY_R1_EXTERNAL_RGBD_REPLICATION_SUPPORTED_DEVELOPMENT_ONLY`.
Insufficient source support yields
`SCALE_FREE_TRAVERSABILITY_R1_NOT_EVALUABLE_SOURCE_SUPPORT`; otherwise any failed
accuracy gate yields
`SCALE_FREE_TRAVERSABILITY_R1_EXTERNAL_RGBD_REPLICATION_NOT_SUPPORTED`.
No terminal authorizes metric distance, obstacle clearance, alerts, safety, or
production behavior.
