# DTR CARLA C33 X65 source terminal

Date: 2026-09-01

Decision: `DTR_CARLA_C33_SOURCE_NOT_EVALUABLE_IN_DOUBT_PARTIAL_DEPTH`

## Frozen identity

- Cohort: `DTR_CARLA_C33_X65_RENDER_TRANSFER_CONFIRMATION_V1`
- Frozen protocol commit: `aff0405f`
- Frozen protocol SHA-256:
  `421CA0C5B5518B87E2AA55679561E99F03BCA45CE5CE6AD7255DCB1B94446EF5`
- Source root:
  `artifacts.local/CARLA/experiments/dtr-carla-c33-x65-render-transfer/evidence/c33-x65-source-20260901-053000`

## What happened

The instance and wearable shards each completed with 728 durable PNG payloads.
The capture supervision session was then interrupted after 21 depth PNGs had
become durable. The depth shard had no completion result, and the witness shard
had not started.

The frozen protocol states
`ANY_NONZERO_PARTIAL_SHARD_IS_SOURCE_NOT_EVALUABLE_AND_MUST_NOT_RETRY`.
Consequently, C33 is terminal and cannot be resumed or recaptured under the
same cohort identity. Its completed and partial payloads remain audit evidence.

## Terminal evidence

- Instance: complete, 728 payloads
- Wearable: complete, 728 payloads
- Depth: in-doubt partial, 21 durable payloads
- Witness: unstarted
- Model predictions created: no
- Evaluator opened: no
- C33 result SHA-256:
  `A91477D068C16CC3E5747C2565909DB8A4206F813FB9C713045B45BBC0B04087`
- Terminal partial manifest SHA-256:
  `8F43114628D26AEA2D8F0189873CBBDEA25D944E16F803CECEA7062A2D257260`
- Partial depth inventory SHA-256:
  `4E07D09AB8E38E9E202D8EB2A0FD60B31AC46F4377555D7432BC790465C33154`

The storage lease was released. No CARLA server remained, and ports
2000-2002 were free when the terminal result was sealed and rechecked.

## Claim boundary and next action

C33 says nothing about X65 precision, recall, F1, generalization, or promotion.
No C33 algorithm result exists. The admissible successor is a new cohort with a
new source identity and frozen protocol. C34 may retain the unchanged X65
predictor and validated geometry, but must use fresh pixels and explicitly name
C33 as a terminal, unscored parent.
