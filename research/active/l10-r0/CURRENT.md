# L10-R0 current

Updated: 2026-09-05

Status: `L10_R0_ACTIVE`

## Core capability and present gap

Keep the user's exact destination bound through active observation, identify its
functional entrance or endpoint, and hand off only with the required evidence.
`referent != affordance != waypoint != arrival != handoff` is the route contract.
The main question is now **how to commit useful local evidence without confusing
missing reference support, contradictory identity, and incomplete door extent**.
The controller and observation components exist; end-to-end L10 is not established.

## Current baseline: latest SEVN comparison

Eight address/panorama-disjoint PAN episodes, frozen before pixels/models, used
24 distinct frames and 16 explicitly supplied different-frame reference crops.
All were excluded from seven prior panels (245 addresses, 260 frames).

| Frozen arm | Correct / wrong / UNKNOWN | Extra online views | Commit precision |
| --- | --- | ---: | --- |
| PASSIVE | 0 / 0 / 8 | 0 | NOT_EVALUABLE |
| FIXED_SWEEP | 5 / 1 / 2 | 24 | 83.3% |
| TRIGGERED_ACTIVE | 5 / 0 / 3 | 20 | 100% |
| TRIGGERED_VERIFIED | 0 / 0 / 8 | 20 | NOT_EVALUABLE |

Triggered observation retained all five recoveries, avoided one wrong binding,
and used 16.7% fewer online observations on this small same-source panel.
Reference setup costs another 16 supplied views; it is separate from online cost.
This result does not erase the [consumed 24-episode failure](l10_sevn_progressive_episode_result_v1.json):
fixed `11/4/9` versus triggered `12/5/7`, with extra views `72 -> 49`.
Those earlier runtime versions differ; only within-panel arm comparisons apply.

The SIFT/RANSAC hard commit gate retained `0/5` correct bindings, below its frozen
80% retention requirement. Wrong-commit reduction is `NOT_EVALUABLE` because the
triggered baseline made no errors. Zero commits is not perfect precision.
Decision: `L10_SEVN_REFERENCE_COMMITMENT_FRESH_DEVELOPMENT_GATE_NOT_MET`.
The [structured disposition](l10_sevn_reference_commitment_disposition_v1.json)
retains this geometric hard-commit role as `NEGATIVE_CONTROL`; central assignment
was blocked by the recorded pre-existing registry fingerprint error.

SEVN provides address-associated door labels, not independent cross-frame physical
door IDs or entrance ownership. References are privileged inputs. Overlapping
address aliases are not sibling negatives. This is same-source Development.
[Report and reproduction](L10_SEVN_REFERENCE_COMMITMENT_20260905.md) /
[sealed episode results](l10_sevn_reference_commitment_result_v1.json).

## Bottleneck and next decision

Only `2/8` truth-proposed target boxes had geometric support; both corresponding
correct runtime masks still failed the extent agreement. All eight truth-proposed
siblings and 61 detector candidates in target-region-excluded viewports were
rejected. These clustered controls do not establish open-world specificity, and
rejection without correct coverage does not establish useful identification.

**Next hypothesis:** separate local instance support, reference-evidence
availability, and complete endpoint extent instead of imposing one unconditional
whole-door veto. Keep the action policy and candidate selection as the baseline.
The [paired analysis](../../WORKFLOW_UPGRADE_20260905.md) is complete: two lost correct
bindings had target-box support but failed runtime commitment; three lacked target-box
support. This consumed diagnostic changes no threshold, model or confirmation claim.

A subsequent bounded exploration should change one evidence representation and
ask whether it retains useful correct commitments while reducing wrong ones at
an explicit observation/reference cost. Report correct/wrong/UNKNOWN together,
correct retention, target support opportunity, and extra views; freeze a meaningful
gain criterion after checking baseline headroom. Abstention must remain visible.
Freeze the current cohort, references, extraction and thresholds; do not rescue
this terminal by changing ratios, inliers, crops, or overlap thresholds.

## Reusable components and their scope

| Component kept available | Evidence and limit |
| --- | --- |
| Seek/guide/reacquire, causal handoff guard, deficit-specific actions | Implemented mechanics; PanoLab entrance-ray recovery `4/4` is geometric, not pixel-portal or arrival evidence. |
| SEVN episode harness and triggered policy | Paired action/cost comparison above; the earlier failure remains frozen. |
| Portal-private PP-OCRv6 medium witness | Fresh same-source 40 PAN episodes: exact OCR `22/37 -> 29/37`, binding `16/0/24 -> 21/0/19`; five new correct bindings, no wrong ones. |
| Full-frame RoMa with target-visible support; separate support and extent surfaces | SceneNN same-provider confirmation `34/0/0`, F1 `1.0`; 3RScan coherent-cycle target-disjoint support `3/3`. Privileged binding/source conditions apply; repeated-door aliasing defeats universal local hard gates. |
| Target-conditioned appearance/memory and bounded candidate sets | Eligible within recorded Development roles. Proposal recall is not Top-1 selection; fresh ranking failures and zero incremental propagation attribution remain in the ledger. |
| Closed-roster assignment and registered planar-extent support | Retained only for externally justified rosters/registered geometry. Scan-family-disjoint `34/2/0 -> 34/0/0`; partial observed-surface veto failed on provider-disjoint SceneNN. |

Detailed component evidence and frozen failures stay in [the route ledger](README.md).
The [evidence reducer](l10_evidence_authority_lattice.py) and
[three-arm contract](action_conditioned_progressive_evidence_commitment_protocol_v0.json)
keep `UNKNOWN -> DIRECTIONAL -> BINDING -> TERMINAL` revocable on conflict/loss.
Direction allows orient/approach; exact referent/facade continuity allows track;
owned visible endpoint plus terminal-pose and handoff support allows handoff.
Distance, pose and scale are analysis axes, not substitute authority.

## Work modes and source boundary

- **Exploration:** one capability question, credible baseline and explanatory
  hypothesis; necessary coupled edits and disclosed Development reuse are allowed.
- **Confirmation:** freeze the mechanism and an admissible untouched source before
  selected pixels/models; report provider/family/target distinctions exactly. Source
  failure is `NOT_EVALUABLE`, not an algorithm negative. No new source is admitted here.
- **Engineering:** controller/device/demo integration is a separate deliverable;
  algorithm-only gains do not establish readiness or reopen integration priority.

The next authority-changing street cohort needs pose/action labels plus independent
exact facade/entrance ownership, sibling/absence and terminal/handoff truth.
ABotN remains a candidate substrate with all `3/3` addendum scenes unadmitted;
KartaView geometry alone did not establish target visibility. Static HierText,
FSNS sign views, controlled CATALIST actions and registered scans supply distinct
components, not those missing authorities. Generic Panoramax pixel-portal mining
and the consumed SceneFun3D ordinal source remain closed.
`UNKNOWN` and `NOT_EVALUABLE` are neither negative evidence nor known-safe.
No result here establishes natural-camera reliability, user benefit or safety.

## Historical authority

The full pre-compaction current is preserved at
[`daf5720064d98a93b75336469d18e9a2fe0023e5:research/active/l10-r0/CURRENT.md`](https://github.com/violetljj/blind-assist/blob/daf5720064d98a93b75336469d18e9a2fe0023e5/research/active/l10-r0/CURRENT.md).
This navigation rewrite changes no historical verdict, frozen input or evidence role.
