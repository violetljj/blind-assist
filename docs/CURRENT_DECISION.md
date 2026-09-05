# Current research decisions

Updated: 2026-08-31

Status: `L10_R0_ACTIVE / DTR_R2_DYNAMIC_RETAINED`

This file owns only decisions that affect what may run or be claimed now. Full
result chains remain in the two route ledgers and result files; Git preserves
the superseded long-form decision history.

Recorded gates, closures, and next actions apply to their named experiments and
claims. They do not exhaust future research choices. A justified new hypothesis
or evaluation criterion may change the next action while preserving the earlier
result; explain the changed premise and the smallest useful check.

## L10-R0

### Controller and active observation

- Keep seek, guide, reacquire, deficit-specific observation actions, and the
  causal handoff guard.
- PanoLab's `4/4` result establishes entrance-ray recovery mechanics only. It
  does not establish pixel-portal identity, arrival, or handoff readiness.
- Actions remain tied to the information gap: `APPROACH`, `SIDESTEP/PAN`,
  `SWEEP`, or `HOLD`.

### Registered partial-observation identity support

The privileged complete-surface 3RScan line remains the retained geometric
core: unchanged positive registered planar-extent support improved rank-only
`TP=34, FP=2, FN=0`, F1 `0.971429`, and `13/15` exact scenarios to
`TP=34, FP=0, FN=0`, F1 `1.0`, and `15/15` on a scan-family-disjoint cohort.

Provider-disjoint SceneNN scene 096 then supplied four different doors and all
`8/8` frozen synchronized RGB-D observations. The single-frame carrier was
`NOT_EVALUABLE`, not negative: two queries retained only `159` and `120` points
against the fixed 256-point minimum, before any score or overlap was computed.
A frozen three-frame registered carrier repaired source support for every role
(`931` to `12,288` points before the fused cap), but falsified convex-hull
intersection as a general observed-surface veto. Rank-only reached
`TP=34, FP=3, FN=0`, F1 `0.957746`, `12/15` exact; support fell to
`TP=25, FP=2, FN=9`, F1 `0.819672`, `6/15` exact because one true pair had zero
partial-view overlap.

Two Exa-motivated appearance successors were also rejected without tuning.
Threshold-free Hue/geometry consensus reached `TP=2, FP=0, FN=32`, and frozen
DINOv2-S foreground-feature/geometry consensus reached
`TP=7, FP=1, FN=27`. Both are `DEAD_FOR_THIS_ROLE` as hard identity vetoes.

A separately frozen four-scene SceneNN challenger then tested EfficientLoFTR
local correspondence with explicit NONE on eight fresh observations. Against
same-crop DINOv2 (`TP=6, FP=24, FN=28`, F1 `0.1875`), it improved to
`TP=11, FP=18, FN=23`, F1 `0.349206`, but supported only two of four true
diagonals and reached `0/15` exact scenarios. The fixed homography-inlier hard
gate is `DEAD_FOR_THIS_ROLE`; the positive delta remains diagnostic evidence.

Retain SceneNN rank-only registered-surface matching as a Development component,
not absence authority. Exa evidence now admits calibrated dense certainty and
spatially balanced correspondence (RoMa/DKM class) as the next challenger on
fresh targets. Do not retune any consumed SceneNN frame, mask, descriptor,
crop, matcher threshold, RANSAC tolerance, or score weight.

### PanoLab referent-candidate router

The fixed appearance bank first ranked `4/4` consumed positives, rejected `4/4`
fresh target-absent images, but covered only `1/2` fresh cross-collection
positives. Exact temporal lexical evidence repaired that pair only posthoc. A
harder three-producer panel was therefore frozen before selected pixels or OCR:
appearance returned `1/0/2` correct/wrong/`UNKNOWN`, and the unchanged exact
token bank returned `0/0/3`.

One structural successor was frozen from those observed failures. It accepts
only high-confidence, roster-unique normalized tokens at edit distance at most
one; length-six/seven tokens contribute one evidence unit, length-eight tokens
two, and two units are required. On the consumed three rows it reached `3/0/0`,
gaining two correct routes over appearance with `0/30` wrong-target matches.
This result is mechanism evidence only.

The decisive confirmation then froze Monoprix and Maison de Quartier Chemin
Vert from a metadata ledger before their ten selected pixels or any OCR call.
Both target ways and all items were router-unseen, used disjoint reference/query
collections, and spanned nlehuby/Nzau capture producers. The unchanged lexical
branch recovered Maison de Quartier from `maison + chemin` and safely returned
`NO_MATCH` for Monoprix; the unchanged conditional appearance fallback recovered
Monoprix. Decision:
`L10_PANOLAB_FRESH_DISTINCTIVE_TOKEN_CONDITIONAL_APPEARANCE_ROUTER_DEVELOPMENT_GATE_MET`
with combined `2/0/0`, `0/24` wrong-target lexical matches, `0/12` appearance
wrong-goal candidates, and zero ownership bindings in the 13-target roster.

A separate attack ran the fixed fuzzy-token branch on four exact-roster-absent
controls across three cities. It emitted no match in all 52 target trials and
kept all four `UNKNOWN`. Decision:
`L10_PANOLAB_DISTINCTIVE_EDIT_TOKEN_OPEN_SET_NEGATIVE_DEVELOPMENT_GATE_MET`.
Those pixels had already been used by the appearance branch, so this is a first
governed OCR negative replay, not a fresh-pixel or formal open-world guarantee.

A consumed progressive early-exit replay retained the fresh `2/2` result using
only the first frame of each sequence (`6 -> 2`, `-66.67%`), with one lexical
and one appearance exit. Per-frame evidence was inspected before freezing the
rule, so this has no fresh online-latency, energy, compute, motion, or dynamic-
length authority.

The first provider/city-disjoint text transfer used four Mapillary panoramas in
Rotterdam and Den Haag that PB19 had human-reviewed but never sent to governed
OCR. Human facade intervals conditioned the crops. The unchanged branch returned
`0/0/4/0` correct/wrong/`UNKNOWN`/ambiguous with `0/64` wrong-target matches:
Markthal exposed only `HAL@0.937`, Ontmoetingskerk was read as
`ONTMOETINOBKERK@0.972`, Oude Kerk exposed only numbers, and Kievitkerk no OCR
row. Decision:
`L10_MAPILLARY_DISTINCTIVE_EDIT_TOKEN_PROVIDER_TRANSFER_DEVELOPMENT_GATE_NOT_MET`.
This freezes a provider-disjoint observation-reachability gap, not a full
combined-router miss and not permission to relax edit distance.

Next admissible action: freeze PS/DF/MR and seek a new provider/city-disjoint
sequence panel with independent exact-target truth, visual references, and
target-absent controls. Develop provider-normalized or character-sequence text
observation only on a separate cohort; do not tune MR crop scale, suffix rules,
OCR preprocessing, or the appearance gate. Exa research supports a separately
calibrated evidence-conditioned next step—sequence-matching receptiveness,
dynamic observation length, or patch verification—but current results have no
conformal/FDR authority. A referent search candidate is still not facade/portal
ownership, access, waypoint, arrival, or handoff evidence.

A metadata-only KartaView/Jakarta successor is now frozen before pixels in
`research/active/l10-r0/l10_kartaview_jakarta_provider_disjoint_candidate_v1.json`.
It admits one character-sequence Development row and one query-independent brand-
appearance row from distinct contributors/sequences, plus two contributor-
disjoint geometric negative-control candidates. The exact successor manifest
`research/active/l10-r0/l10_kartaview_jakarta_materialization_v1.json` then
materialized `8/8` identity/hash-verified images (`6` query-window plus `2`
negative candidates) across four sequences and four contributors. Human truth,
OCR, router, and appearance calls remained `0`. A direct full-frame Codex source
audit then found either selected target in `0/6` query frames while both negative
candidates provisionally lacked their targets. Decision:
`L10_KARTAVIEW_JAKARTA_CODEX_VISUAL_SOURCE_AUDIT_NOT_EVALUABLE`. This is not an
algorithm miss and not independent human truth. Freeze both opened Jakarta
neighbourhoods; the next new provider/city source must add source-side facade or
sign visibility authority before selection because point-bearing/FOV geometry did
not establish target pixel reachability.

The official public `mcp-karta-view` implementation was then pinned at commit
`37ac5c062c9ace66bcc540086236127a3fb86bf0` and reviewed without installing its
dependencies or opening another cohort. Public `nearby_photos` only calls the
same geometric photo endpoint; `object_search` returns an empty result plus
`KartaView token is required` before dispatch when `x-karta-token` is absent.
Decision:
`L10_KARTAVIEW_PUBLIC_MCP_SOURCE_SIDE_VISIBILITY_AUTHORITY_NOT_ADMITTED`.

A separate source-disjoint lexical-development line used the official
RoadTextVQA annotations and sampled-frame OCR. The first two fresh 30-video
panels proved that information-weighted text raised correct rows by six and five
respectively, but also exposed a contract error: many apparent wrong/ambiguous
roster candidates were genuinely co-visible businesses. The successor therefore
conditions verification on the already selected user goal, uses video-document
IDF from 2,635 training videos, requires at least eight information bits, permits
exact evidence for four/five-letter tokens and at most edit distance one for
longer tokens, and can emit only the conditioned goal. On a third disjoint,
OCR-unseen 30-video panel it changed correct/`UNKNOWN` `6/24 -> 19/11`
(`20.0% -> 63.3%`, `+43.3 pp`), with `0/30` generated-negative accepts, zero
wrong-goal candidates, and zero identity/portal bindings. Decision:
`L10_ROADTEXTVQA_GOAL_CONDITIONED_BACKGROUND_IDF_VERIFIER_DEVELOPMENT_GATE_MET`.
One cyclic cross-video challenge matched a word actually visible in that video;
without independent absence truth it is a collision diagnostic, not specificity.

One layout-only successor retained the IDF contract and concatenated two to four
same-line OCR boxes at fixed overlap/gap thresholds. On the consumed panel it
recovered `U + HAUL` and `HIGH + END`, changing `19/30 -> 21/30` with `0/30`
generated negatives. The separately frozen fourth 30-video panel recovered only
`VAPOR + IN`, changing `20/30 -> 21/30` (`70.0%`) with `0/30` generated
two-token accepts. Decision:
`L10_ROADTEXTVQA_FRESH_GOAL_CONDITIONED_LAYOUT_PHRASE_VERIFIER_DEVELOPMENT_GATE_NOT_MET`
because the preregistered minimum gain was two. This is fresh evidence of a
small real effect, not permission to tune box gap, overlap, edit distance, or
information threshold on the consumed pixels.

The independent specificity gap was then tested on official HierText test
pixels with complete word transcripts and polygons. A first fresh 30-image
multiscale panel gained four correct carriers but added two wrong carriers. On a
second disjoint panel, projecting a unique exact query substring onto its span
inside a merged OCR quadrilateral raised correct carrier `7 -> 22`, while three
errors exposed two invalid query contracts: text-identical instances and truth
fragments inside longer words. The fixed third-panel successor uses complete
truth only for evaluator selection: one occurrence across every word node, no
target-as-longer-token substring, and complete-truth absent controls. Runtime
keeps query-span projection and abstains when best-rank OCR evidence has multiple
spatial carrier components. On 30 further pixel-unseen test images it changed
correct/wrong/`UNKNOWN` `11/0/19 -> 22/0/8` (`36.7% -> 73.3%`, `+11` correct),
with `0/30` complete-truth-absent accepts. Decision:
`L10_HIERTEXT_TEST_UNIQUE_REFERENT_SPAN_CARRIER_DEVELOPMENT_GATE_MET`.

The static carrier mechanism then transferred to official French Street Name
Signs multi-view samples. Each selected sample contains four different-
position/time Street View crops of one physical street sign and a canonical
street-name label. On 30 official testdata examples, adding later views changed
single-first-view exact recognition `26/30 -> 30/30`; this missed only the
development panel's preregistered `+5` gain requirement because its maximum
possible gain was four. The unchanged algorithm was frozen on 40 validation-
shard samples whose image hashes, normalized labels, and distinctive goal tokens
were disjoint from all 50 testdata examples. It changed correct/`UNKNOWN`
`34/6 -> 40/0` (`85% -> 100%`, `+6` correct). All six recoveries were exact;
three first appeared at view 2 and three at view 3. Canonical-label-disjoint
challenges and synthetic negatives both stayed `0/40`. Decision:
`L10_FSNS_FRESH_MULTIVIEW_GOAL_EVIDENCE_DEVELOPMENT_GATE_MET`.

The next Exa-admitted source tested actual controlled camera transformations.
Official CATALIST videos identify translation, pan, tilt, zoom, or roll and have
manually verified video-level target text. Before OCR, 30 validation videos were
frozen at six per action; the 12 complete labels and 12 conditioned target
tokens were absent from every training label/token. Transferring the unchanged
FSNS exact-plus-two-frame-edit verifier changed first-frame correct/`UNKNOWN`
`27/3 -> 29/1` (`90.0% -> 96.7%`, `+2`). Both new recoveries were pan episodes;
label-disjoint challenges and synthetic negatives remained `0/30`. This missed
the preregistered gain of five, so record
`L10_CATALIST_CONTROLLED_ACTION_GOAL_RECOVERY_DEVELOPMENT_GATE_NOT_MET`.
A consumed-cache two-token diagnostic changed `27 -> 27`, so do not open the
remaining CATALIST validation labels for the same representation.

Next admissible action: freeze all four RoadTextVQA fresh panels. Keep goal
conditioning, background rarity, HierText query-span projection, and spatial
ambiguity abstention as the text-observation successor; freeze all three
HierText panels, both FSNS panels, and the layout branch. The next decision-
changing source is a direction/pose-labelled street sequence that tests whether
an L10-selected observation causes recovery and links the sign to exact target-
instance/facade/entrance truth. FSNS proves multi-view canonical sign reading;
CATALIST adds executed transformation classes but not direction, metric pose,
stationary counterfactuals, or sign-to-facade association. A text carrier
remains neither exact real-world instance/facade/portal ownership nor access,
waypoint, arrival, or handoff evidence.

### 3D Street View center-target lock

Exa discovery admitted the official 3D Street View benchmark because its test
pairs carry provider-verified same-physical-point truth and place the target at
the optical center. The official 1 GB archive was rate-limited, but its frozen
181,018,624-byte prefix contained 6,220 fully decodable real images. Intersecting
those images with the official pair ledger and applying a deterministic endpoint-
prefix split produced 2,180 train pairs and 426 prefix-disjoint test pairs; 957
cross-partition pairs were discarded.

One frozen DINOv2-small mechanism keeps provider target patches unchanged and
scale-gates only full-image center crops. Against global/global cosine, held-out
AUROC improved `0.949859 -> 0.984860`, balanced accuracy
`0.895773 -> 0.926341`, average precision `0.954445 -> 0.982243`, and retrieval
Top-1 across 32 eligible anchors `0.843750 -> 1.000000`. Image/patch AUROC moved
`0.874644 -> 0.966274`. Decision:
`L10_3DSTREETVIEW_CENTER_TARGET_LOCK_DEVELOPMENT_GATE_MET`.

Freeze this cohort, split, model, crop scales, and score. The result strengthens
real-image exact-point persistence under scale/view change but remains same-
provider archive-prefix Development evidence. The provider truth does not label
doors, entrances, venue ownership, public access, traversability, approach,
waypoints, arrival, handoff, user benefit, or safety. The next authority-changing
step must pair the unchanged point lock with independent portal-class/ownership
truth or test it unchanged on a provider-disjoint exact-point source.

### Truth-proposed door roster transfer

The first provider transfer reused three consumed SceneNN door episodes. Its six
truth envelopes were severely edge-clipped, and the unchanged center-scale score
missed the gate: AUROC `0.500000 -> 0.444444`, Top-1 stayed `2/6`, and minimum
margin was `-0.221899`. Freeze those six crops as a partial-visibility falsifier;
this is an information-contract mismatch, not a general cross-provider verdict.

The next 3RScan cohort selected three stable doors from pose/depth/instance
geometry before opening RGB, with at least `32 px` source-image margin. Global
cosine reached `5/6`; the center-scale score reached only `4/6` because two doors
from one scan family collided. A parameter-free one-to-one Hungarian assignment
repaired that consumed matrix to an equivalent `6/6`, with a `0.246219` complete-
assignment margin. That repair was then frozen before selecting three physically
target-disjoint 3RScan doorframes. On the new targets, pair AUROC/AP, independent
Top-1, and assigned identity were all perfect (`1.0`, `1.0`, `6/6`, `3/3`); the
best complete center-scale assignment exceeded second-best by `0.419960`.
Decision:
`L10_3RSCAN_ROSTER_ASSIGNMENT_TARGET_DISJOINT_DEVELOPMENT_GATE_MET`.

This admits closed-roster exact stable-door matching only when an external
authority supplies a complete one-to-one reference/query roster and privileged
geometry supplies the proposals. The single 3RScan provider and scan families
overlap prior Development; labels do not establish a doorway aperture, named-
venue ownership, public access, traversability, waypoint, arrival, handoff, user
benefit, or safety. Freeze all consumed SceneNN/3RScan targets, the center scales,
score, visibility rule, and assignment. Do not reinterpret the closed assignment
as open-set evidence.

Inheritance disposition: the 3D Street View center lock is `RETAINED_CORE` as
the current real-image point-pair score below portal authority; the SceneNN edge-
sliver result is a `NEGATIVE_CONTROL` for partial-visibility source admission;
the 3RScan center-scale pair score is
`COMPONENT_OR_CHALLENGER / COMPONENT` inside the roster composition; and the
parameter-free closed-roster assignment is `RETAINED_CORE` for this exact
closed one-to-one responsibility. Neither failed pre-RGB freeze is classified as
dead. The enforced records live in
`research/knowledge/decision/inheritance.json`.

The open-roster successor then froze four new physical targets and four scenarios
(closed, query-extra, reference-extra, and balanced missing-plus-extra) before
RGB. Multiview appearance plus strict reciprocal zero assignment retained only
`4/12` true matches, emitted `5` false matches, missed `8`, and reached F1
`0.380952`; the balanced swap produced `0` true and `2` false matches. Record
`L10_3RSCAN_OPEN_ROSTER_ZERO_ASSIGNMENT_TARGET_DISJOINT_DEVELOPMENT_GATE_NOT_MET`
as `NEGATIVE_CONTROL`: relative appearance rank cannot establish that a match
exists.

Registered target-surface distance recovered all `12/12` true correspondences on
four further physical targets, but rank-only reciprocity still forced one
adjacent doorframe-to-door match in the balanced swap (`TP=12, FP=1, FN=0`, F1
`0.96`, exact unmatched sets `3/4`). Record the surface scorer as
`COMPONENT_OR_CHALLENGER / COMPONENT`, not standalone absence authority. A third
fresh cohort predeclared two registration witnesses; their maximum residual set
an unchanged `0.038756 m` rejector ceiling. Witness-calibrated zero assignment
reached `TP=12, FP=0, FN=0`, F1 `1.0`, and exact unmatched sets `4/4`, versus
complete-assignment F1 `0.96`. Record
`L10_3RSCAN_WITNESS_CALIBRATED_ZERO_ASSIGNMENT_TARGET_DISJOINT_DEVELOPMENT_GATE_MET`
as `COMPONENT_OR_CHALLENGER / CHALLENGER` because rank-only also reached F1
`1.0` on this fresh cohort: witness incremental value is not yet established.

A scan-family-disjoint test then froze six same-class doors and all twelve
ordered balanced swaps. Rank-only reached `TP=64, FP=2, FN=0`, F1 `0.984615`;
the unchanged raw two-witness ceiling removed both false matches but reduced true
matches to `48`, introduced `16` misses, and lowered F1 to `0.857143`.
Record the raw witness maximum as `DEAD_FOR_THIS_ROLE`: it is not a family-
general absolute rejector, and the consumed `0.376774 m` ceiling must not be
rescued by slack, multipliers, or another selected statistic.

The information-changing successor froze six unconsumed same-family
`doorframe` targets and all thirty ordered missing-plus-extra swaps before
opening surface scores or overlaps. Strict reciprocal surface rank reached
`TP=136, FP=4, FN=0`, F1 `0.985507`, with exact unmatched sets in `29/33`
scenarios. Requiring only strictly positive registered horizontal-by-vertical
convex-hull overlap retained all `136` true matches, removed all `4` false
matches, reached F1 `1.0`, and made all `33/33` unmatched sets exact. Record
`L10_3RSCAN_REGISTERED_EXTENT_SUPPORT_ZERO_ASSIGNMENT_TARGET_DISJOINT_DEVELOPMENT_GATE_MET`
as `RETAINED_CORE` for privileged registered-geometry partial rosters.

The rule was then kept unchanged while only `2.85 MiB` of semantic and instance
geometry was materialized for the first eligible unconsumed scan family. On four
same-class doors and all twelve ordered swaps, rank-only reached
`TP=34, FP=2, FN=0`, F1 `0.971429`, with exact unmatched sets in `13/15`
scenarios. Positive registered extent support reached `TP=34, FP=0, FN=0`, F1
`1.0`, and `15/15` exact sets. Record
`L10_3RSCAN_EXTENT_SUPPORT_SCAN_FAMILY_DISJOINT_DEVELOPMENT_GATE_MET` as an
independent-family confirmation of the retained core.

The next authority-changing check is provider-disjoint confirmation or an
independently justified phone-side target-conditioned planar-support carrier.
Current overlap remains privileged geometry and spatial support; it does not
establish named-entrance ownership, public access, traversability, waypoint,
arrival, handoff, user benefit, or safety.

### SEVN address-door backend

Decision:
`L10_SEVN_PPOCRV6_MEDIUM_PORTAL_WITNESS_FRESH_PAN_DEVELOPMENT_GATE_MET`.

On 40 further fresh PAN episodes, portal-private PP-OCRv6 medium observations
improved visible-number exact OCR `22/37 -> 29/37` and binding
`16/0/24 -> 21/0/19` correct/wrong/`UNKNOWN`. All five new proposals were
correct, both ambiguous witness sets abstained, every baseline-correct binding
was retained, and binding precision remained `100%`. The panel has zero overlap
against all 205 prior addresses and 220 prior panorama frames.

Next admissible action: keep the V5 representation frozen and seek one genuinely
source-disjoint provider/city with independent door-instance and
address-credential truth plus negative/no-portal controls. Do not tune private
crops, OCR models, witness uniqueness, or abstention on the opened V5 panel.
Same-source SEVN Development success is not portal ownership, access,
traversability, waypoint, arrival, handoff, user-benefit, or safety evidence.

### Metric portal and endpoint

3RScan registered extent established a Development ceiling. The latest
source-distinct spatial mask reached `0.5403` complete IoU and `0.422 m`
centroid error, below the `60%` ceiling-retention gate, and confused an
overlapping doorframe.

Next admissible action: add exact-instance or portal-set authority before
another endpoint-mask successor. A geometrically plausible nearby frame cannot
be counted as the target entrance.

Generic Panoramax pixel-portal mining and the consumed SceneFun3D ordinal source
remain closed.

## DTR-R2

### Public/JRDB line

Decision: `DTR_X21_TRACK_CARRIED_COMPONENT_ANCESTRY_GATE_MET` for Development
only.

X21 reached `5/6` CONTACT, 11 false segments, 45.45% Event F1, `3.061 s`
median lead, and `8/18` dropout recovery on six consumed sequences. It may
transport only an already authorized component row while its anchor remains in
the same live track; it cannot absorb a new current cell.

Next admissible action: one frozen, genuinely source-disjoint confirmation of
unchanged X21. Do not tune or resample the six opened sequences.

### CARLA algorithm and occlusion-source line

- X24 remains the same-source C2 Development reference.
- X26 and X30 missed their frozen gates; their consumed cohorts are closed.
- C8-C11 did not admit an evaluable X31 occlusion source. C11 reached only
  `1/8` valid complete-occlusion episodes and ran no X31 prediction or metric.
- X31 remains a candidate representation, not a result.

Next admissible action: admit a new raster-observable occlusion source before
inference. Do not tune C8-C11 source thresholds, select favorable episodes, or
convert `NOT_EVALUABLE` into an algorithm failure/success.

### CARLA native-dynamics line

N3 materialized Town01, Town04, and Town05 with `12/12` authored long-tail
effects. The sole frozen N4 invocation completed Town01, then stopped before
Town04 pixels when free memory was below the frozen floor; Town05 never ran.

Decision: `DTR_CARLA_N4_REPLAY_ATTEMPT_CONSUMED_INCOMPLETE`. A complete replay
requires a new versioned authority. N4 v1 cannot be resumed, retried, or
reported as a three-town result.

## Cross-route boundaries

- L10 and DTR do not wait for, modify, or validate one another.
- Proposal, selection, referent, affordance, waypoint, arrival, and handoff are
  distinct authorities.
- `UNKNOWN` is not `CLEAR`; `NOT_EVALUABLE` is not a negative result.
- Curated, synthetic, replay, registered-source, and device evidence keep their
  actual claim ceilings.
- Local uncommitted candidates are not current authority until a scoped result
  delivery updates the owning route current.

## Stop here

- Do not rescue consumed evidence with threshold, seed, cohort, backbone,
  aggregation, or narrative changes.
- Do not add governance or tests unless they protect interpretation, prevent a
  material irreversible failure, or change the next decision.
- Do not infer Android readiness, natural-distribution performance, user
  benefit, navigation reliability, or safety from the current research results.

Route authority:
[L10-R0 current](../research/active/l10-r0/CURRENT.md) and
[DTR-R2 current](../research/active/dtr-r0/CURRENT.md).
