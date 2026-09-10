# MZ28: full-task fixed packet-coordinate availability comparator

2026-09-10 EXPLORE, consumed Development, zero parameter updates. MZ27's fixed
packet retrieval correctly labels27/29 scored hard negative anchors, versus
10/29 raw visual and4/29 learned descriptors, but rejects12/61 true witnesses.
Test whether this conditional signal improves actual alert errors/coverage when
applied to every candidate, without selecting evaluation locations by truth.

Build one immutable bank from the7362 MZ20 unique TRAIN IDs excluding the200 MZ6
sequence frames. For each physical zone store float32 packet4: cleaned two ranges
divided by4 and two good bits (valid, finite,0<range<=4), plus native known49 labels
for that zone. Known0 is missing in-domain native evidence, never CLEAR/occupancy.
Record site/group identities at bank construction for source audit only. The375
bank sites are disjoint from oldDEV, relationDEV and distanceDEV; verify that
before execution. No evaluation site identity is passed to prediction.

At each query frame/zone, compute Euclidean distances in float64 over the float32
packet4 vectors to every bank frame at that same zone. Select exactly five nearest
references, tie by ascending bank globalID. Sum each cell's five known labels;
availability score = votes-2.5, eligible when votes>=3. Bank labels are training
targets; current native labels, object family, site/group, task truth and query
identity are not inputs. Invalid packets stay observation-unavailable; their
geometric candidate mask is empty regardless of retrieval votes.

This fixed full-task comparator has no visual features in availability, no fitted
metric/normalizer, reference class requirement, distance cutoff or query-site
exclusion. MZ27 required both classes only to compare descriptor separation; MZ28
uses every five-neighbor vote, including unanimous neighborhoods. Report vote
histograms, distances and coverage explicitly; unanimity does not prove calibrated
confidence. The previous18 unscored anchors do not become confirmed correct cases.

Replace only MZ26 availability with this packet predictor. Preserve frozen MZ20
candidate geometry/scores/cutoffs and MZ5 add-only fallback. One run evaluates
oldDEV1000, clean/stress200, relationDEV2000 and distanceDEV1000 plus existing
wrong-visual-zone controls on all but oldDEV. Availability is unchanged under
wrong visual correspondence, while the frozen MZ20 scorer receives the control.
No union/intersection arms, k/distance/threshold sweep, new fit or capture.

Report MZ5/MZ20/MZ26 and candidate task bits, addedTP/FP, every baseline-positive
retention, complete frames, group/site/family slices, pole49 clean/stress,
native availability precision/recall and actual-contributor witness retention.
Useful component effect keeps the existing zero-addedFP,>=68/75 placement far
addition, gain on both placement sets, pole>=48/49 both conditions and all MZ5
positive gates. Report raw results regardless of the gate. Add-only cannot remove
the81 original placement false bits; this comparator is not whole-baseline repair.

Record offline bank size and synchronized packet retrieval runtime separately
from cached-feature MZ20 inference; no phone latency claim. Save bank IDs/packets/
known labels and all neighbor IDs/distances/votes, predictions, input/code hashes,
source overlap checks and receipts. Independently recount task metrics and verify
full bank provenance and selected58 frame neighbor votes/ranks. Preserve mechanical
failures in fresh outputs. Stop after this one fixed evaluation, finish scoped
report/terminal/delivery and release processes. Retain task-beneficial component
only within demonstrated scope; otherwise retain the fixed comparator as a
negative control and use its failure pattern to choose a different mechanism.
No protected EVAL, App/default-baseline promotion, temporal or safety claim.
Output: artifacts.local/work/mz28-packet-availability-20260910/run-v1.
