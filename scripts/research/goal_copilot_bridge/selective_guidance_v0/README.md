# Real-episode selective guidance V0 contract

状态：`EXPERIMENTAL_CONTRACT_IMPLEMENTED / CURRENT_FRAME_ONLY / DEFAULT_APP_UNCHANGED`

This package owns the pure V0 decision vocabulary, completion authority and lightweight episode event log. It has no
mutable target state and cannot carry a candidate, feature, identity, gallery or direction across frames.

`contract.py` separates `UNIQUE / SET_VALUED / AMBIGUOUS` cardinality from decision state; emits current-frame
`FOUND / CONTESTED / NOT_VISIBLE / STALE / ABSTAIN`, directional commands, range buckets, `STOP_FOR_SAFETY`,
`HANDOFF_READY`, and `COMPLETED_BY_USER`; and rejects perception/provider/controller completion receipts.
`LOST` is an episode interaction event derived only on `VISIBLE -> NOT_VISIBLE_AFTER_VISIBLE`. `NEVER_SEEN ->
NOT_VISIBLE` is not LOST. This FSM retains no embedding, crop, instance match, gallery or identity state.
`event_log.py` appends speech/action-time JSONL sufficient to reconstruct the goal, current candidate set, referent,
decision, output, range/uncertainty, user confirmation/denial, handoff, latency and provider provenance.

This is a research integration seam, not P1 persistence and not default-App behavior. It must not import trackers,
re-ID, gallery growth, world anchors, VIO/SLAM or scene graphs.

```powershell
E:\codex-tools\bin\blindassist-python.cmd -m unittest `
  scripts.research.goal_copilot_bridge.selective_guidance_v0.test_contract
```
