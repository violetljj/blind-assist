# BlindAssist CMP Facade native-door 89 result

Status: `SEALED / NATIVE_GT / REFERENT_SELECTION_AND_COMMITMENT_DOMINANT / NO_RERUN / NO_P1`

## Frozen cohort

- Source: official CMP Facade Database, CC BY-SA.
- Universe: 606 real rectified facade RGB images with matching XML and PNG annotations.
- Eligible before provider execution: 211 images with exactly one XML `door` object and non-empty native door pixels.
- Roster: first 89 by the frozen SHA-256 ranking rule.
- Provider input: RGB, literal goal `the door`, and frozen Grounding DINO proposal boxes only.
- Private truth: native PNG door pixels plus XML referent identity.
- Teachers: 0. Provider batches: 12. Provider `in_doubt`: 0. Retry/rerun: 0.

The roster file SHA-256 is `961913c69048014b3b86433fd17097f6acb4742838e7974e702d3de2e456e32e`.
The sealed provider report content SHA-256 is
`82d3b2a6731913298dfdd991b6672a5cbdffd828b83ffae4508769bd27fe2bee`.
The authoritative post-run audit content SHA-256 is
`b21ad6a4e2cab80eb1c4756496a3683622903c48b79efc95e5c37f89fc1a487f`.

## Native-truth result

At fixed IoU 0.50:

- proposal availability: `82/89`;
- proposal Recall@1/3/5/10: `44/67/73/82` of 89;
- correct grounding: `45/89`;
- proposal miss: `7/89`;
- abstain or ambiguous despite a usable proposal: `30/89`;
- wrong confident guidance with a usable proposal: `7/89`.

The Brain emitted `55 SELECT / 31 AMBIGUOUS / 3 ABSTAIN`. Of its 55 commitments, 45 were correct and 10 were
wrong, so commitment accuracy is `45/55 = 81.8%`; wrong confident guidance is `10/89 = 11.2%` over all observations.
Among the 82 observations with a usable proposal, selection accuracy including abstention is `45/82 = 54.9%`.

The first-failure accounting is therefore:

```text
89 native-truth observations
├─ 45 CORRECT_GROUNDING
├─ 7  PROPOSAL_MISS
└─ 37 REFERENT_SELECTION / COMMITMENT
   ├─ 30 abstain or ambiguous with a usable proposal
   └─ 7  wrong confident guidance with a usable proposal
```

Proposal is not the dominant bottleneck on this cohort. The supported next research direction is selective
commitment: distinguish safe abstention from missed usable proposals, reduce wrong confident guidance, and represent
contested candidates explicitly. Any algorithm successor must use a separately frozen fresh cohort; this consumed
89-image cohort is evaluation-only and cannot be tuned or rerun.

## Evaluator repair

The first post-run calculation treated CMP XML `<x>` values as horizontal coordinates. The frozen PNG labels showed
that CMP stores vertical coordinates in `<x>` and horizontal coordinates in `<y>`. The authoritative audit uses the
official PNG door pixels directly as region truth and retains XML only for referent identity. Mean normalized XML/PNG
edge difference is `0.001307` after the axis correction versus `0.434569` without it. The repair added zero provider
calls, teacher calls, retries, or reruns and preserved the original model outputs byte-for-byte.

## Claim ceiling

`CMP_FACADE_CURRENT_FRAME_DOOR_ONLY_NO_TRAJECTORY_NO_RANGE_NO_LOST`

The cohort contains rectified facade still images and a generic door goal. It does not establish named-store identity,
approach episodes, closed-loop control, metric range or bearing, arrival, temporal loss, P1 persistence, blind-user
effectiveness, safety, or default-App readiness.
