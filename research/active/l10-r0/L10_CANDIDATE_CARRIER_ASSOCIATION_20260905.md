# Candidate and physical-carrier evidence association

Implementation and synthetic mechanics are complete. The consumed SEVN replay
establishes adapter parity only. No new real-camera identity, ownership, arrival,
handoff, or accuracy improvement is claimed. No existing SEVN runner was changed.

## Implemented behavior

`l10_candidate_carrier_association.py` associates observations by source, frame,
ROI and candidate ID. It retains the distinction between local support,
identity-bearing support, unresolved association and explicit contradiction.
Multiple compatible carriers remain a set; appearance score cannot resolve an
ambiguous text-carrier join by itself. A missing observation is never absence.

The optional Named-POI integration now derives confirmation and guidance position
from the same carrier. Current source/track receipts can continue a semantic lock
for at most four track-only observations by default. Missing/replayed observations
do not refresh that lifetime; a new source, a sibling track, or explicit current
contradiction cannot inherit the lock. Its preexisting bounded missing-frame hold
remains separate. These are observation-count bounds, not measured time or pose
guarantees.

An upstream explicitly `COMPLETE` identity-name observation that does not match a
target alias is a contradiction for that candidate. It cannot fall through to
the preexisting partial-text-plus-appearance rule. No generic-word list or text
similarity threshold changed. When completeness is unknown, the WIP's existing
partial-text heuristic still has a shared-suffix limitation; this work does not
establish its lexical specificity.

## Public observation contract

- Every observation declares `source_id`, `frame_id`, and `roi_id`. Candidate and
  OCR rows may inherit these identifiers; stale source/frame rows are unresolved.
- Candidate rows provide `candidate_id` and `bbox_xyxy_norm` in their declared
  ROI. Named-POI also consumes the existing `poi_id` and `score`. Temporal reuse
  requires a producer's `track_id` and a current `track_evidence_id`.
- Text rows provide `text_id`, `text`, and their own ROI. Unique normalized-box
  containment supplies local spatial support. Geometry supplies identity-bearing
  support only with `role=IDENTITY_SIGN`, `role_evidence_id`, and an explicitly
  complete candidate roster. These are producer claims, not evaluator labels.
- A supported `IDENTITY_TEXT_OF` edge can supply an explicit identity relation.
  `LOCAL_OBSERVATION_OF` and `LOCAL_TOPOLOGY_SUPPORTS` retain only local producer
  support. Edges must name the text and candidate, source/frame, ROI endpoints,
  status, and `evidence_id`. Cross-ROI edges use `text_roi_id` and
  `candidate_roi_id`; no coordinate transform is guessed.
- Private-crop pixel boxes may be retained as `bbox_xyxy_px` when their transform
  was not exported. They cannot supply full-frame containment. A recorded crop
  relation can still retain its existing local witness.
- Advertisement, direction-sign, sibling-sign, and explicit unknown identity
  relations do not name the carrier. `name_completeness=COMPLETE` is optional and
  must be supplied by the identity-observation producer, not inferred from an OCR
  substring. No geometric overlap grants entrance ownership or endpoint extent.
- Legacy Named-POI inputs without the required spatial/semantic relation stay
  unresolved. Existing SEVN surrogate outputs use the separate typed adapter;
  they are not forced through a new exact-name identity veto.

## Evidence

The synthetic counterexample keeps every OCR observation and appearance score
unchanged. Text on the right carrier with a higher-scoring left distractor caused
the old WIP to confirm the target while pointing at `x=0.25`. The integration
selects the right carrier at `x=0.75`. If only the nonoverlapping left candidate
remains, it returns unresolved. Both matched-carrier preservation cases (complete
identity sign and partial identity sign plus appearance) retain confirmation.
These stipulated producer-role fixtures demonstrate mechanics only.

The first SEVN diagnostic incorrectly imposed full-frame containment. Its
triggered `12/5/7 -> 2/3/19` result is preserved as a failed diagnostic. Inspection
identified two actual adapter defects:

1. `portal_private_rows` in `l10_sevn_ppocrv6_medium_portal_witness.py` keeps OCR
   coordinates in the resized private crop; `concise_output` in
   `l10_sevn_progressive_episode.py` drops the crop transform and coordinate tag.
2. `pair_topology` in `l10_sevn_pixel_topology_replay.py` permits a credential
   halo outside the door mask. The old producer's emitted local association must
   not be replaced with door-box containment.

V2 corrects these adapter contracts, preserves the selected producer witness at
its original scope, and opens no new pixels or models. Its paired results are:

| Arm | Before correct/wrong/UNKNOWN | V2 local surrogate | Correct retention | Extra observations |
| --- | --- | --- | --- | --- |
| PASSIVE | 0/3/21 | 0/3/21 | Not evaluable | 0 |
| FIXED_SWEEP | 11/4/9 | 11/4/9 | 100% | 72 |
| TRIGGERED_ACTIVE | 12/5/7 | 12/5/7 | 100% | 49 |

No candidates, OCR values, selected views or action choices changed. Wrong
surrogate bindings are also preserved: parity is compatibility, not an accuracy
gain. Exact named-identity support remains unavailable (`0/0/24` in each arm),
because the legacy export has no independent identity-sign or exact-name
relation. That does not revoke the original local address-door surrogate.

The evidence directory is
`artifacts.local/evidence/l10-candidate-carrier-association-20260905/`:
`synthetic_counterexample.json`, preserved `consumed_sevn_replay.json`,
`v1_diagnostic_disposition.json`, corrected `consumed_sevn_replay_v2.json`, and
`delivery_receipt.json`. The receipt records input, source, patch and snapshot
hashes. The consumed input hash was checked before and after both replays.

## Validation and delivery

The following focused checks passed with `PYTHONDONTWRITEBYTECODE=1`:

```powershell
python -m unittest discover -s research/active/l10-r0 -p test_l10_candidate_carrier_association.py -q
python -m unittest discover -s research/active/named-poi-v1 -p test_named_poi_v1.py -q
python research/active/l10-r0/l10_candidate_carrier_replay.py --input research/active/l10-r0/l10_sevn_progressive_episode_result_v1.json --output artifacts.local/evidence/l10-candidate-carrier-association-20260905/consumed_sevn_replay_v2.json
```

The new suite passed 40/40 checks, including the cross-carrier counterexample,
adjacent doors, advertisements, missing/ambiguous provenance, complete-name
contradiction, private-crop coordinate domains, and bounded continuity. Existing
Named-POI checks passed 6/6 after their synthetic fixture explicitly declared its
carrier and identity-sign evidence. A separate copy without the untracked WIP
passed all 14 standalone checks and explicitly skipped the 26 optional WIP
integration checks. Its task-owned temporary directory was removed and absence
verified.

Clean-checkout source dependencies are only the Python standard library and:

- `l10_candidate_carrier_association.py`
- `l10_candidate_carrier_replay.py`
- `test_l10_candidate_carrier_association.py`

The optional integration is already applied in this working tree to exactly
`named-poi-v1/named_poi_v1.py` and `named-poi-v1/test_named_poi_v1.py`. Both files
preexisted as untracked WIP and were snapshotted before edits. Do not absorb the
whole WIP directories. Deliver this module, replay, tests, report, disposition and
`named_poi_carrier_integration_v1.patch` as task-owned files; the patch carries
only the two-file integration delta and can be applied to the hash-matching WIP
baseline. It is not automatically applied when running standalone tests.

Patch SHA-256:
`5a6dfb613f60c72c4484d3ba92d363852c72eff8fe1e78ad14ca8e4c4fb5172b`.
Baseline `named_poi_v1.py` SHA-256:
`4c6d4be631418dc4c34bfdf51c2fda7e840f6f1792f0218023d66f2871159aeb`.
Baseline `test_named_poi_v1.py` SHA-256:
`7a2a8d2c96d14552fab239be8e3c109507c1777431ab63b566c331fc13c3d6c5`.

Remaining empirical gap: a real observation producer must supply the carrier,
identity-role and temporal-association fields, with an evaluable matched-carrier
panel and sibling/advertisement controls. The existing SEVN export cannot
establish that exact-name effect. No detector/OCR model, trained head,
open-world-POI WIP or route CURRENT was changed by this component implementation.
The [structured disposition](l10_candidate_carrier_disposition_v1.json) retains
the typed adapter as a component; central registration is recorded separately.
The shared delivery registration attempt failed before mutation on the existing
`experiments/index.jsonl:252` input-fingerprint mismatch. The exact error and
unchanged shared-file hashes are in
`artifacts.local/analysis/refactor-20260905/registration-result.json`. This leaves
a central metadata gap; the local disposition and scoped implementation remain
complete, and no concurrent registry or knowledge file was overwritten.
