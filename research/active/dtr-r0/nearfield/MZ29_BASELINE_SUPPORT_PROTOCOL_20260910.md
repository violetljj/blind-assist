# MZ29: locate the baseline errors outside the add-only repair surface

2026-09-10 EXPLORE, consumed Development, no training or new inference. MZ28
retains81 placement false bits because its composition preserves all MZ5 alarms.
MZ22's residual arbiter changed only geometrically supported queries and removed
zero placement baseline false bits. Determine whether the errors lie outside its
repair surface, and what true-positive coverage occupies the same surface.

Use frozen MZ20, MZ26 step4800 and MZ28 saved predictions on oldDEV1000,
clean/stress200, relationDEV2000 and distanceDEV1000. Include all baseline-positive
query bits, split by task truth. Report per-query/cohort counts and margins under
original MZ20 geometric support, learned availability support and packet support.
Report frozen local-score agreement/disagreement with baseline independently of
whether the add-only candidate changes it. No new threshold, combination arm,
candidate selection or model is adopted.

Reconstruct eligible geometry from saved observed packets and frozen rays. For
each baseline-positive bit, count actual native query contributors, eligible
query contributors, and measured packet validity. Align moved stress echo labels.
Unavailable local support is not CLEAR; native query truth/known/source are
diagnostic labels only. Preserve actual-positive geometry gaps separately from
source-no-query cases. Include per-frame raw records, family/site identities
where explicitly recorded, and source/site split counts; no guessed identities.

If baseline false bits concentrate outside geometric support while correct bits
also occur there, an unsupported-query correction needs independent observable
evidence; existing supported-only MZ22 arbitration did not test that responsibility.
If they remain supported, revisit the existing observable local-score separation
before adding another fallback model. No claim that missing support justifies
vetoing alarms or that an oracle label is an inference signal.

One complete saved-array/geometry pass, scalar CPU reductions (no neural work).
Register before execution. Bind input/code/protocol hashes, verify baseline/task
identities against frozen MZ20, and independently recount exported rows. Stop
after the finite diagnostic, record the resulting decision and release resources.
No protected EVAL, new source, device or safety claim. Fresh output:
artifacts.local/work/mz29-baseline-support-20260910/run-v1/.
