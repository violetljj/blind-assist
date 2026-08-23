# BlindAssist ABotN `HANDOFF_V1` fresh-task result — 2026-08-23

## Outcome

The lexicographically first remaining task in the already rendered ABotN scene was frozen before provider use:
`abotn-20260212121852-traj-9`, goal `BRANEW布瑞琳`, 135 source poses and 675 official-renderer views.
The action graph reaches native metric arrival in 9 actions under the fixed 12-action budget.

`HANDOFF_V1` changes only termination semantics. The existing centered, height-at-least-0.55 current-frame cue now
stops automatic motion and emits `HANDOFF_READY`; it cannot emit `ARRIVED`, `COMPLETE`, or `COMPLETED_BY_USER`.
Completion still requires an explicit user receipt or a contracted independent interaction receipt. The frozen
handoff engineering window is native distance-to-goal at most 3 m; `HANDOFF_READY` is not completion.

The official renderer completed `675/675` calls. All 675 frame hashes are unique and the public graph/pixel/private
truth firewall had zero private-literal hits. Four of five sampled poses had strong bidirectional ORB direction
evidence; the final sparse-texture pose was recorded as `INSUFFICIENT_FEATURES`, with no wrong-direction evidence.

The provider run is terminal and not rerunnable:

```text
provider observations dispatched: 2
provider observations completed:  1
provider observations in doubt:   0

o001: RUN_SUCCESS
      Grounding DINO proposals = 13
      P0 = ABSTAIN_NO_RELIABLE_EVIDENCE

o002: both frozen schema attempts exited 1
      ChatGPT OAuth refresh token revoked / HTTP 401
```

Terminal classification:

```text
NOT_EVALUABLE_PROVIDER_AUTH_SESSION_ENDED
```

The run provides no evaluable claim about metric progress, `HANDOFF_READY`, arrival, completion, proposal miss,
referent selection, range/bearing, or persistence. In particular, o001's abstention is not promoted into an
algorithm failure denominator. `traj_9` is consumed and must not be rerun after re-authentication.

## Evidence

- prospective freeze SHA-256: `110891ee3e76834d18d9608190a89ae41eec5d9923ab50108f321bc1683eb607`
- action-graph freeze SHA-256: `69759a4798056193ce78f80e4ba063f5cf134e31f61205ee5645660e10579382`
- official-pixel receipt SHA-256: `f5d207b27bcfbfdc219090fba2a93f3fdea7e1afca168e2cdedd7bd6fe720f0d`
- pixel qualification SHA-256: `4b541109da4e6e228f3d469b54b85e5e72864865316d02bf869b73be8b4f473d`
- closed-loop manifest SHA-256: `fb1d0a526b635e44a7a3305941530461894b4c83d27dac4a2fa3ca1a6db98ba5`
- terminal receipt SHA-256: `bf45b0877022df4e75a59d5d2e13e3e866eaebd544c6ee7738fafda9156d628d`
- ignored evidence root: `artifacts.local/evidence/abotn-handoff-v1-traj9/`

The task-owned remote renderer, scene payload, transfer archive, and processes were removed after the local evidence
copy and hashes were verified.

## Follow-up fresh cached-scene task

After CLI authentication was repaired, `traj_9` remained sealed and was not rerun. The first unused task in the
already cached second official scene was frozen independently:

```text
episode: abotn-20260227163550-traj-1
goal: 康乐大药房
source poses / official views: 96 / 480
native shortest arrival path: 6 actions
provider observations: 7 completed, 0 in_doubt
teacher calls / reruns: 0 / 0
```

All `480/480` official frames rendered and had unique hashes. Pixel qualification had `5/5` strong direction
samples and zero private-truth literal hits. The seven closed-loop observations were all `GROUNDED`; native distance
decreased from `13.520 m` to `9.638 m`, for `3.882 m` positive progress.

The frozen action sequence was:

```text
p000 yaw  0: TURN_LEFT
p000 yaw +1: FORWARD
p014 yaw +1: TURN_RIGHT
p014 yaw  0: TURN_RIGHT
p014 yaw -1: TURN_RIGHT
p014 yaw -2: FORWARD
p028 yaw -2: TURN_RIGHT -> edge unavailable
```

The terminal was therefore `CONTROL_POLICY_BOTTLENECK_ACTION_EXHAUSTED`, before handoff or native arrival. More
specifically, this is evidence that the current synthetic action contract is insufficient: viewport yaw is bounded to
`[-2, +2]` while a real person can continue turning, and its `FORWARD` edge follows the released source trajectory
rather than a body-heading-dependent physical action. Expanding a yaw threshold or substituting `FORWARD` at the
boundary would be a substrate rescue, not evidence about the goal-driven algorithm.

Supported conclusion:

```text
HANDOFF_V1_NOT_EVALUABLE_BEYOND_ACTION_CONTRACT_YAW_BOUNDARY
```

This task does establish that current-frame reliability was not the first failure (`7/7` reliable) and that the
closed loop made native metric progress. It does not establish proposal accuracy, referent selection, range/bearing,
persistence, handoff readiness, arrival, or completion. The next legitimate work is a physically executable control
substrate or real interaction path, not prompt/threshold/provider/teacher tuning.

Follow-up evidence SHA-256:

- prospective freeze: `aa1711c09169a327acd1184c7e0786c4687a0a3646810f65e76d1304530726a6`
- action-graph freeze: `2b96aa355860eb6dff8afd4d312913374438a438d54604b7b307b364259dbbfd`
- official-pixel receipt: `39a2a234b1f44534dfd476df869a8862b89c06e4ab6149e966155dfd91d1a725`
- pixel qualification: `647fc036dce7623ad3d90a046baa7031a62ff00c77963eab1c27a9e004677b8d`
- closed-loop manifest: `9bcd6c17ecd26ebd2998cce54cbd894fbe01da0ce279d7633e1bfce4170d4268`
- terminal receipt: `e7e1f81aa686541ddc370888194747f456df6556090c9237a751d45425ea036a`
- ignored evidence root: `artifacts.local/evidence/abotn-handoff-v1-scene2-traj1/`
