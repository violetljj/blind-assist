# L10-R0 current

Updated: 2026-09-01

Status: `L10_R0_ACTIVE`

## Question

Can BlindAssist keep the user's exact destination bound through active
observation, identify its functional entrance or endpoint, and hand off only
when the required evidence is actually present?

`referent != affordance != waypoint != arrival != handoff` remains the route
contract.

## Current decisions

- **Controller:** seek, guide, reacquire, and the causal action-belief handoff
  guard remain implemented. Controlled controller results are mechanics
  evidence, not real-camera or product evidence.
- **Provider-disjoint partial observed surfaces:** unchanged complete-surface
  extent support remains retained after scan-family-disjoint 3RScan
  `TP/FP/FN 34/2/0 -> 34/0/0`, F1 `0.971429 -> 1.0`, and `13/15 -> 15/15`
  exact unmatched sets. SceneNN scene 096 then supplied four provider-disjoint
  doors and `8/8` synchronized RGB-D frames. The single-frame carrier stopped
  pre-score because two queries had `159` and `120` points below the fixed 256
  minimum (`NOT_EVALUABLE`). Exact `[-5,0,+5]` registered fusion made all eight
  roles evaluable, but partial convex-hull support regressed rank-only
  `34/3/0`, F1 `0.957746`, `12/15` exact to `25/2/9`, F1 `0.819672`, `6/15`
  exact. One true pair had zero observed overlap. Two independently motivated
  hard-veto successors also failed: Hue/geometry consensus was `2/0/32`, F1
  `0.111111`; frozen DINOv2-S FFA/geometry consensus was `7/1/27`, F1
  `0.333333`. A new four-scene fresh challenger then compared EfficientLoFTR
  local correspondence plus explicit NONE with same-crop DINOv2. It improved
  F1 `0.1875 -> 0.349206` and TP/FP/FN `6/24/28 -> 11/18/23`, but supported
  only `2/4` true diagonals and reached `0/15` exact scenarios. Freeze all four
  veto roles. A following four-scene fresh RoMa indoor challenger used frozen
  `0.30 m` minimum-baseline active frames and bilateral dense-cycle support. It
  removed all `23` DINOv2 false positives and improved F1
  `0.268657 -> 0.418605` at precision `1.0`, but only `1/4` true diagonals had
  absolute support. Failure attribution showed that minimum-baseline-first could
  select temporally distant revisits (`90-525` frames). One consumed posthoc
  action-only repair changed pair ranking to minimum frame gap while retaining
  `0.30-0.37 m` displacement and every matcher/support value. Gaps fell to
  `20-35` frames; DINOv2 reached `34/2/0`, F1 `0.971429`, `13/15` exact, and
  unchanged RoMa reached `27/0/7`, F1 `0.885246`, precision `1.0`, with `3/4`
  supported true diagonals. The full gate still failed on TL04. The unchanged
  confirmation then admitted four new source-disjoint scenes (`043`, `082`,
  `213`, `207`) before geometry or RGB. The temporal-local rule itself
  generalized, selecting `20, 25, 50, 35` frame gaps at `0.30-0.34 m`.
  Same-crop DINOv2 reached `25/4/9`, F1 `0.793651`, `5/15` exact; RoMa kept
  precision `1.0` and removed all four false positives, but supported only
  TF01/TF04 and fell to `16/0/18`, recall `0.470588`, F1 `0.64`, `2/15`
  exact. Record
  `L10_SCENENN_ROMA_TEMPORAL_LOCAL_FRESH_DEVELOPMENT_GATE_NOT_MET`. Retain the
  action policy and precision branch as `COMPONENT_OR_CHALLENGER / COMPONENT`.
  Two consumed-cohort repairs then failed without changing thresholds: exact
  target-plane rectification and one fixed midpoint bridge both remained
  `16/0/18`, F1 `0.64`, `2/15` exact, and `2/4` supported. An Exa-motivated
  spatial-domain successor instead kept each complete `640x480` frame as RoMa
  input and counted cycles only from the provider target-visible mask into the
  paired target-visible mask. That single change reached `34/0/0`, precision
  and recall `1.0`, F1 `1.0`, `15/15` exact, and `4/4` true support without
  changing weights, resolutions, certainty, cycle error, support thresholds,
  scenarios, or assignment. The identical rule then admitted the final four
  unconsumed exactly-one-door SceneNN scenes (`074`, `109`, `005`, `076`) before
  selected geometry, RGB-D, or model access and independently retained
  `34/0/0`, precision/recall/F1 `1.0`, `15/15` exact, and `4/4` true support.
  Record
  `L10_SCENENN_ROMA_FULL_CONTEXT_MASK_FRESH_CONFIRMATION_GATE_MET` and upgrade
  it to `COMPONENT_OR_CHALLENGER / COMPONENT` for same-provider Development.
  A first proposal-carrier bridge then replaced all eight provider masks with
  truth-blind GroundingDINO top-one `door` boxes and native SAM2.1 masks while
  keeping full-frame RoMa and every support/assignment threshold unchanged.
  Despite proposal IoU spanning `0.60-0.95`, it retained `34/0/0`, F1 `1.0`,
  `15/15` exact, and `4/4` true support. Record
  `L10_SCENENN_ROMA_GROUNDED_SAM_PROPOSAL_POSTHOC_DEVELOPMENT_GATE_MET` and
  retain it as `COMPONENT_OR_CHALLENGER / CHALLENGER`: it is consumed posthoc,
  exactly-one-door, and desktop-GPU evidence. The next structural check must
  preserve prompt and thresholds on a multi-door or provider-disjoint source.
  That frozen 3RScan check stopped on its first required reference image:
  GroundingDINO retained `0` boxes at the unchanged `door` prompt and `0.4/0.3`
  thresholds despite an evaluation-only provider target box. It made one
  GroundingDINO call and zero SAM2/RoMa calls, so record
  `L10_3RSCAN_ROMA_GROUNDED_SAM_PROPOSAL_POSTHOC_SOURCE_NOT_EVALUABLE`, not a
  matcher or segmentation negative. Freeze the consumed six-frame cohort and
  category-prompt carrier. A target-conditioned successor then transported an
  already-bound reference rectangle through the largest bidirectionally
  cycle-consistent RoMa component and one affine envelope. It localized all
  three target doors without category text or query truth: prompt IoU
  `0.638-0.838` and query SAM2-mask bbox IoU `0.584-0.916`, but unchanged
  bilateral support was `2/3` because DR01 forward purity was `0.494961`.
  Replacing the rectangle entirely with a native reference SAM2 mask restored
  support to `3/3` but collapsed DR01 extent. The final dual-surface split used
  that SAM mask only for identity support and the already-bound rectangle only
  for complete extent. It passed the frozen gate with prompt IoU `0.637-0.837`,
  query-mask bbox IoU `0.582-0.909`, and absolute support `3/3`; record
  `L10_3RSCAN_ROMA_CYCLE_PROMPT_DUAL_SURFACE_POSTHOC_DEVELOPMENT_GATE_MET` and
  retain it as `COMPONENT_OR_CHALLENGER / CHALLENGER`. This is consumed posthoc
  evidence with privileged initial binding. The first pre-RGB physical-target-
  disjoint cohort retained transported-extent IoU `0.767-0.920` and bilateral
  support `3/3`, but native query SAM undersegmentation made the overloaded
  mask-as-extent gate fail at IoU `0.371`. A four-surface split retained SAM
  masks only for identity support and bound/transported boxes only for extent.
  On a second new-target cohort, extent IoU was `0.738-0.793` and support-mask
  bbox IoU `0.724-0.870`, but whole-mask global purity supported only `2/3`.
  The spatially coherent successor kept the unchanged `0.01` cycle opportunity
  and required the largest 8-connected cycle component to contain at least half
  of all valid cycles. Its posthoc minimum dominance was `0.653`. The unchanged
  rule then confirmed on a third pre-RGB physical-target-disjoint cohort:
  extent IoU `0.532-0.851`, support-mask bbox IoU `0.542-0.915`, cycle
  opportunity at least `0.508`, dominant-component fraction at least `0.825`,
  and coherent support `3/3`; legacy global purity again reached only `2/3`.
  Record
  `L10_3RSCAN_CYCLE_COMPONENT_SUPPORT_PHYSICAL_TARGET_DISJOINT_CONFIRMATION_GATE_MET`
  and retain the four-surface/coherent-cycle mechanism as
  `COMPONENT_OR_CHALLENGER / COMPONENT` for same-provider Development. The
  frozen cross-scene open-set successor kept those three positives and added
  four exact-target-absent pairings whose query came from another physical
  3RScan scene family. It retained `3/3` positives and rejected `4/4` negatives
  with `0` false commits. Positive cycle opportunity was `0.508-0.645`; all
  negatives stayed below the unchanged `0.01` minimum (`0-0.002821`) even when
  the largest residual component held a majority of the few surviving cycles.
  Record
  `L10_3RSCAN_CYCLE_COMPONENT_OPEN_SET_POSTHOC_DEVELOPMENT_GATE_MET` and retain
  exact-target-absent cross-scene rejection as consumed posthoc Development
  evidence. The first execution aborted because zero cycles lacked a natural
  non-commit branch; the recorded implementation-only repair mapped that case
  to rejection without changing any model, pair, threshold, or gate. Stop
  spending 3RScan positive targets on tuning. The harder same-scene panel then
  exposed one repeated-door false binding: the unchanged local mechanism kept
  `3/3` positives but committed `1/2` negatives. Bilateral reference/query
  masks, a target-excluded global fundamental matrix, the official ICCV 2023
  Doppelgangers classifier with a frozen RoMa adapter, two bound references,
  and a `0.594-1.035 m` second query were each tested once. Global geometry
  rejected both negatives but lost DR03; the learned classifier, dual reference,
  and active query all retained the DR01->DR02 false binding. The active-query
  arm also lost edge-touching DR01, ending at `2/3` positives and `1/2` false
  commits. This freezes repeated-door perceptual aliasing as a real ceiling for
  any single local hard gate.

  A zero-model posthoc recomposition now requires the retained local bilateral
  binding plus either the unchanged `0.5` target-excluded global-epipolar
  majority or `0.5` paired-cycle coverage in both directions between primary
  and active query masks. It uses complementary failure modes: global geometry
  preserves DR01/DR02, active-query majority preserves DR02/DR03, and neither
  corroborates the false sibling. On the consumed five-pair panel it reached
  `3/3` positive commits, rejected `2/2` target-absent siblings, made `0` false
  commits, and had committed precision `1.0`. Record
  `L10_3RSCAN_COMPLEMENTARY_CORROBORATION_POSTHOC_DEVELOPMENT_GATE_MET`; it is
  mechanism evidence, not fresh confirmation. The first frozen physical-target-
  disjoint confirmation source stopped before RGB/model access because FC30's
  best second query covered only `0.591093` of the target versus the frozen
  `0.98` requirement. A pre-RGB successor retained FC30 only as the absent-
  target bound and admitted new positive targets FC31/FC08. The unchanged
  universal corroboration then rejected both positives despite successful local
  binding and extent localization: primary cycle opportunity was `0.613/0.324`,
  while global epipolar support was only `0.415/0.363` and active-query paired
  coverage only `0.298/0.435` and `0.287/0.184`. The exact-target-absent FC30
  negative had zero local cycles and was correctly rejected. Record
  `L10_3RSCAN_COMPLEMENTARY_CORROBORATION_PARTIAL_PHYSICAL_TARGET_CONFIRMATION_GATE_NOT_MET`;
  universal specialist corroboration is not a general hard gate.

  An Exa-motivated selective cascade now lets retained local bindings with at
  least one-quarter primary coherent-cycle coverage exit directly and requests
  complementary global/active corroboration only below that point. On the two
  consumed panels it restored `5/5` positives, retained `0/3` false commits and
  precision `1.0`, while only `1/8` rows requested the extra branch (a
  counterfactual `87.5%` branch-avoidance rate). Record
  `L10_3RSCAN_SELECTIVE_CORROBORATION_CASCADE_POSTHOC_DEVELOPMENT_GATE_MET`.
  The `0.25` point was selected after seeing the failed confirmation, without a
  sweep, so this is Development only; freeze it and require another pre-model
  source before any confirmation or efficiency claim.

  That next source froze never-consumed target SC34 from a scan family not used
  by the coherent-cycle mechanism plus two new cross-scene pairings. SC34 had
  strong local cycle evidence (`0.442978` opportunity, `0.981006` component
  dominance) and both negatives had zero cycles, but affine extent IoU was only
  `0.065238`; the existing extent gate correctly prevented a false target
  commit. One Exa-motivated USAC-MAGSAC homography used the same component,
  inherited `6 px` scale, and no sweep. Despite `0.942384` inliers and `1.452 px`
  mean inlier residual, extent IoU reached only `0.081832`. Record both
  `L10_3RSCAN_SELECTIVE_CORROBORATION_CASCADE_SCAN_FAMILY_CHALLENGE_GATE_NOT_MET`
  and `L10_3RSCAN_PROJECTIVE_EXTENT_POSTHOC_DEVELOPMENT_GATE_NOT_MET`. The stable
  cycles bind the wrong repeated structure. A target-excluded `2x` context ring
  then produced `0.929730` homography inliers and `1.543 px` mean residual but
  only `0.094112` extent IoU, so surrounding same-RoMa context is not an
  independent instance cue either. Stop changing geometric fit family.

  A geometry/depth-only observation-adequacy audit then found the actual source
  defect. SC34 query truth is a bottom-edge sliver: bbox short side `44.693 px`
  (`0.082765` of the image short side) and aspect ratio `12.5635`. Within `1.1 m`
  of the primary pose, `143` poses were checked, only the original view passed
  the inherited selector, and `0` views passed the frozen `0.1` short-side plus
  aspect-at-most-`8` adequacy rule. No RGB or model was opened. Record
  `L10_3RSCAN_SC34_SOURCE_NOT_EVALUABLE_WITHIN_BOUNDED_ACTIVE_OBSERVATION`:
  SC34 is not positive transfer evidence and is not an algorithm-negative row.
  Apply the adequacy rule before RGB/model access to the successor source.
  Two subsequent metadata-frozen, entirely new scan families opened `359` and
  `636` pose/depth pairs but no RGB/models; neither had a single view with the
  old `0.98` inside-target requirement. A three-view, no-threshold observation-
  portfolio diagnostic instead admitted anti-sliver partial views. On the second
  family, the best reference/query views exposed `0.639163/0.799754` of target
  vertices, and a second query view raised cumulative coverage to `0.898649`.
  Freezing the best single views and keeping the coherent-cycle carrier entirely
  unchanged yielded `1/1` positive, `0/2` false commits, cycle opportunity
  `0.402330`, dominance `0.822290`, and extent IoU `0.646993`. Record
  `L10_3RSCAN_CYCLE_COMPONENT_OPEN_SET_POSTHOC_DEVELOPMENT_GATE_MET`. This is
  consumed-source Development, but it establishes that observation admission,
  not another model threshold, was the decision-changing lever. Freeze the
  anti-sliver/max-new-visible-surface rule for a third pre-RGB/model family.

  The third pre-download-frozen family passed that anti-sliver source gate but
  exposed a sharper ceiling. Its independently maximum-visible frames had
  strong per-view depth visibility (`0.918/0.860`) yet the unchanged RoMa
  carrier produced zero reference cycles. Geometry showed that the selected
  pair shared only `0.092593` of mutually registered target surface and had
  viewing-direction cosine `0.722007`. Neither a whole-reference-box support
  fallback nor a structurally different frozen EfficientLoFTR carrier repaired
  it: EfficientLoFTR missed the positive (`5` inliers against `6`) and falsely
  supported one negative (`6` inliers). A geometry-only rerank selected a
  nearly aligned alternative (`0.988995` cosine), but RoMa still had zero
  cycles. Freeze this family as RGB-transport-unreachable for the two tested
  carriers; do not tune either matcher on it.

  The successor therefore moved matchability into source admission. Before
  RGB/model access it required at least `0.10` mutually visible registered
  target surface and `0.90` viewing-direction cosine, then maximized their
  product. The next metadata-frozen family (`1dd7209f -> 1dd720a1`, target `3`)
  passed strongly on frames `84 -> 0`: mutual-surface fraction `0.402247`,
  direction cosine `0.978666`, score `0.393666`, with `0` RGB/model calls. The
  unchanged carrier then committed the new-source positive and rejected both
  fixed cross-scene negatives with zero false commits. Positive cycle fraction
  was `0.132233`, dominant-component fraction `0.500136`, and transported
  extent IoU `0.648127`; both negatives had zero cycles. Record this as narrow
  same-provider action-plus-carrier confirmation: the innovation is a
  pre-model transport-opportunity observation gate, not a relaxed matcher.
  It still proves no raw-camera referent acquisition, named entrance, ownership,
  access, waypoint, arrival, handoff, user benefit, reliability, or safety.
  The unchanged source gate and carrier then replicated on a second entirely
  new pre-download-frozen family (`2ea047d1 -> 2ea047cd`, target `23`). Frames
  `170 -> 115` were selected before RGB/model access with mutual-surface
  fraction `0.672619`, direction cosine `0.999998`, and score `0.672618`.
  The one frozen model run again reached `1/1` positive commit and `0/2` false
  commits; positive cycle fraction was `0.871316`, component dominance
  `0.995959`, and transported extent IoU `0.725189`, while both negatives had
  zero cycles. This is second-family Development replication of the same narrow
  component claim, not provider independence or end-to-end ten-metre evidence.
  A literature-motivated bounded multi-view memory then tested whether the
  frozen PV28 zero-cycle family merely suffered from one bad reference view.
  Geometry/depth-only greedy portfolios raised visible-surface coverage from
  `0.2301 -> 0.3864` on the reference side and `0.2087 -> 0.3932` on the query
  side. Yet all `3 x 3 = 9` positive view combinations had zero reference
  cycles; the memory produced `0/9` positive commits while all six cross-family
  controls remained non-commits. Freeze multi-view expansion of the same pixel
  carrier as insufficient for PV28. The next mechanism must add an instance-
  level appearance/geometry representation, not more RoMa views or thresholds.
  A frozen DINOv2 object-plus-context successor then exposed that signal. A
  three-view mean prototype reached only `4/6` cross-family rankings and margin
  `-0.038055`; preserving the three reference descriptors as a set and scoring
  the nearest member reached `6/6`, with minimum-positive `0.437761`, maximum-
  negative `0.433960`, and margin `+0.003801`. A geometry-selected same-scene
  sibling door was then added before descriptor access; the unchanged set
  memory reached `9/9` positive-versus-negative rankings, with sibling score
  `0.389877`. This is a promising consumed Development mechanism under
  privileged boxes, not calibrated open-set or fresh-family evidence. Freeze
  the representation and seek a wholly new family rather than tune the margin.
  That frozen mechanism then passed its first wholly new-family test. Candidate
  `3b7b33af -> 80b8588d`, target door `16`, was fixed before download; the
  geometry/depth source gate obtained three reference views with `0.927771`
  cumulative target coverage, three query views with `0.994361`, and same-scene
  sibling `32`, all before RGB/model access. The unchanged object-plus-context
  set memory reached `9/9` strict positive-versus-negative rankings. Minimum
  positive similarity was `0.578135`, maximum negative `0.445939`, separation
  margin `+0.132196`, and sibling score `0.279386`. Record narrow same-provider
  new-family confirmation under privileged boxes. This improves instance-
  memory generalization, but it is not raw-camera binding or end-to-end L10.
  An Exa/NIDS-Net-motivated successor then removed query truth boxes. The fixed
  `objects` prompt at the literature-reported `0.10` box threshold supplied an
  IoU-`0.5` target opportunity in `3/3` full query frames; unchanged set-memory
  ranking selected the target in `2/3`, with the failed frame placing a wrong
  right-side proposal at `0.577516` just above a target proposal at `0.567368`.
  A frozen target-minus-sibling memory changed that selection back to the target
  body but reached only IoU `0.495605`, so its strict gate remained `2/3`.
  Completing the unchanged proposal pipeline with native SAM2 mask refinement
  crossed the frozen gate at `3/3`: minimum refined-box IoU `0.500517`, mean
  `0.773318`. This is consumed same-scene Development evidence that the bound
  memory can drive class-agnostic query acquisition; the minimum margin is thin
  and no fresh-family or raw-phone confirmation exists yet.
  The sealed full chain then received a pre-download, pre-RGB, pre-model new-
  family test on `422885ce -> 1c211554`, target door `13`, sibling door `59`.
  The source gate supplied three views per side (`0.544934` reference and
  `0.936099` query cumulative coverage) and a sibling from `9` eligible views.
  Class-agnostic proposal opportunity stayed strong at `3/3`, but unchanged
  target-minus-sibling CLS ranking plus SAM2 localized only `1/3`; minimum
  refined IoU was `0.018747`, mean `0.325531`. Freeze this as a new-family
  representation failure. A NIDS-Net-style foreground feature aggregation
  successor then replaced global CLS crops with SAM-masked DINOv2 patch means
  on the exact frozen proposals. Keeping the scene-specific sibling subtraction
  improved localization to `2/3` (minimum refined IoU `0.402587`, mean
  `0.712768`), but on the remaining failure it ranked a partial target above the
  full target because the sibling term reversed their target-only order. The
  smallest structural ablation removed only that subtraction: direct max-over-
  target-memory FFA ranking plus unchanged SAM2 reached `3/3`, with minimum
  refined IoU `0.926509` and mean `0.934408`. Freeze target-only FFA as the next
  candidate representation. These two posthoc runs use the already-consumed
  D13 roster, so they are Development attribution—not fresh-family confirmation,
  calibrated rejection, or end-to-end L10 evidence.
  The frozen target-only FFA chain then moved to a pre-download/pre-RGB/pre-
  model family, `422885e9 -> 422885d4`, target door `3`. Geometry/depth admitted
  three reference views with `1.000000` cumulative coverage and three query
  views with `0.971037`. Full-frame GroundingDINO supplied IoU-`0.5` opportunity
  in only `2/3` frames; target-only FFA selected and SAM localized both reachable
  targets (`2/2`), but the third roster's best possible proposal IoU was only
  `0.265858`. The sealed full-frame confirmation therefore failed at `2/3`,
  minimum refined IoU `0.265405`, mean `0.566257`, specifically at proposal
  reachability rather than memory ranking. An Exa/literature-motivated fixed
  four-corner `65% x 65%` overlapping-tile successor changed only the proposal
  set. On consumed D03 it restored opportunity and full localization to `3/3`;
  the repaired top-left-tile row refined to IoU `0.561599`, while overall minimum
  was `0.558732` and mean `0.664988`. Freeze tiling plus target-only FFA as the
  next candidate chain and seek another wholly new family. The tiled result is
  posthoc Development and cannot inherit the earlier new-family authority.
  That frozen tiled chain then moved unchanged to pre-download/pre-RGB/pre-model
  family `43b8cae9 -> 43b8cae5`, target door `15`. Three reference views covered
  `0.666952` of target vertices and three thin/edge query views covered only
  `0.206753`. The fixed proposal union supplied IoU-`0.5` opportunity in `2/3`,
  but target-only max-memory FFA selected oversized context-bearing masks in all
  three rows: refined `0/3`, minimum IoU `0.085538`, mean `0.232151`. In D15M1
  an IoU-`0.771087` proposal was rank `2` by only `0.002388`; in D15M2 an IoU-
  `0.8616` proposal fell to rank `26`; D15M3 still had no reachable proposal.
  Freeze this as a fresh failure of single-reference maximum aggregation under
  thin/partial target views, plus a remaining scale-reachability gap. The next
  Development change must make memory robust to one context-heavy reference
  before adding further proposal scales.
  On this now-consumed D15 roster, replacing max-over-reference with a top-two
  mean repaired D15M1 and changed refined success `0/3 -> 1/3` (IoU
  `0.760687`). Multiplying that consensus by agreement with the median
  reference-mask aspect ratio repaired both originally reachable rows `2/2`
  (D15M1 `0.760687`, D15M2 `0.876069`). A separate fixed full-plus-four-corner
  `40% x 40%` proposal audit then restored IoU-`0.5` opportunity to `3/3`,
  including D15M3 at raw IoU `0.507007`. But composing those two changes did
  not generalize: the enlarged roster introduced geometry-mimicking distractors
  and the integrated chain passed only `1/3` (minimum `0`, mean `0.251676`;
  one empty SAM mask skipped). Freeze median aspect as an insufficient ranking
  proxy across views. The next structural successor must improve the foreground
  representation or learn target-independent channel discrimination, not add a
  vertical-position prior or another D15-specific geometry term. All four runs
  are consumed-cohort Development attribution only.
  Exa then exposed the exact training-free NIDS-Net/SAM-6D local appearance
  score: choose the maximum-FFA template, average each query foreground patch's
  maximum cosine to that template's foreground patches, and average this with
  the instance score. Applied unchanged to the `40%` roster, it retained
  opportunity `3/3` but localized only `1/3` (minimum `0`, mean `0.382016`).
  Crucially it recovered the previously missing D15M3 thin target—raw IoU
  `0.507007`, refined IoU `0.723311`—while D15M1's best target ranked `3` and
  D15M2's ranked `25`. Freeze local patch coverage as a complementary partial-
  view mechanism, not a standalone repair. The remaining gap is reusable
  channel discrimination or another instance-bearing cue, not more score
  blending on D15.
  The official NIDS Weight Adapter was then trained on `22` FFA descriptors
  from consumed C16, D13, D03, NC08, and NC31 families while excluding every
  D15 input. Its frozen InfoNCE loss fell `0.698618 -> 0.095535`, but the held-
  out D15 chain was unchanged at `1/3`, minimum `0`, mean `0.382016`; D15M2's
  best target fell to rank `32`. Freeze this as no cross-family transfer from
  the small adapter bank. Do not tune epochs, seed, or optimizer on D15. Exa's
  FoundPose evidence instead points to intermediate-layer DINOv2 patches for
  positional discrimination when final-layer semantics are ambiguous.
  Replacing only the local score's final-layer patches with FoundPose's frozen
  block `18/24` representation improved D15 `1/3 -> 2/3`, mean refined IoU
  `0.382016 -> 0.456744`: D15M1 reached `0.646921` and D15M3 retained
  `0.723311`. D15M2's true proposal became the highest layer-18 local match but
  final-layer FFA fusion demoted it to rank `7`. Removing only that harmful
  semantic fusion made the fixed local-only chain pass `3/3`, minimum refined
  IoU `0.594841`, mean `0.705944` (D15M2 `0.876069`). Freeze layer-18 query-
  directed local appearance plus SAM as the next candidate. This is strong
  consumed D15 Development attribution; it now requires an untouched family.
  Two official `422885` sequence archives were then added to local assets, but
  the unchanged strict source selector admitted `0/7` candidate targets before
  RGB/model access. The next metadata-only selection therefore froze wholly
  unmaterialized family `47319774 -> 47319776`, target door `8`, before download,
  RGB, or model access. Its geometry/depth gate admitted three reference views
  at `0.968627` cumulative vertex coverage and three query views at `0.905336`,
  plus a same-scene sibling door. On this genuinely fresh family, unchanged
  layer-18 local-only ranking supplied proposal opportunity `3/3` but localized
  `2/3`: refined IoU `0.953660`, `0.145739`, `0.834245`, minimum `0.145739`,
  mean `0.644548`. Record
  `L10_3RSCAN_FOUNDPOSE_LAYER18_LOCAL_ONLY_FRESH_CONFIRMATION_GATE_NOT_MET`:
  the representation transferred, but all-view extent reliability did not.
  The failed row's tile anchor and correct containing full-frame proposal kept
  the same target reference and differed by only `0.001192` local score. A
  consumed posthoc part-to-whole completion repaired it to IoU `0.955422`,
  producing `3/3`, minimum `0.834245`, mean `0.914442`. A first center-only
  association then failed a D15 preservation audit. The tightened successor
  requires at least `0.8` tile-anchor containment and `0.99` local-score
  retention. It exactly preserved all six sealed selections across consumed
  D15 and fresh-consumed door-8: D15 remains `3/3`, minimum `0.594841`, mean
  `0.705944`; door-8 remains `3/3`, minimum `0.834245`, mean `0.914442`.
  Retain geometry-and-appearance-consistent cross-scale extent completion as
  the next candidate, but only as two-family consumed Development. It still
  requires another untouched family for confirmation.
  That untouched check froze wholly new `48699c02 -> a7616234`, door `12`,
  before RGB/model access. The geometry gate admitted three reference views at
  `0.791815` cumulative coverage and three queries at `0.652356`, plus sibling
  door `17`. The fixed layer-18 plus `0.8/0.99` extent chain then failed `1/3`:
  refined IoU `0.449737`, `0`, `0.976225`, minimum `0`, mean `0.475321`, and
  extent completion triggered `0/3`. Record
  `L10_3RSCAN_CROSS_SCALE_EXTENT_FRESH_CONFIRMATION_GATE_NOT_MET`; proposal
  opportunity remained `3/3`, so this is ranking/mask failure, not source
  unreachability. A consumed failure-driven successor adds one reference-derived
  scale cue: subtract `0.05 * abs(log(candidate area / reference median target
  area))` from local appearance. Across consumed doors `12/8/15` it reached
  `8/9`, mean `0.795400`, minimum `0.449737`; door 8 and door 15 remained `3/3`,
  and door 12 improved to `2/3`. Freeze this as a promising scale-aware
  Development candidate, not confirmation. The remaining miss is a correct
  large proposal whose SAM mask truncates the target; the next untouched test
  must freeze a mask-extent/stability mechanism without tuning on door 12. A
  provenance-aware successor reuses the existing `0.8` threshold: only for a
  full-frame proposal, retain the proposal box when the SAM mask bbox keeps
  less than `0.8` of its area; tiles retain SAM refinement. It fired once and
  passed all consumed doors `12/8/15`: `9/9`, minimum IoU `0.594841`, mean
  `0.827756`; door 12 became `3/3` with minimum `0.740937`. Freeze this complete
  scale-prior plus full-proposal extent-guard chain for an untouched family.
  This is Development, not fresh confirmation or pixel-mask evidence.
  Exa's official CVPR source independently identifies the same SAM failure mode:
  high-scoring local parts can miss whole-object extent, and its Box Mining
  Strategy expands overlapping larger proposals. This supports the mechanism
  class, not the present numeric threshold or confirmation claim.
  The frozen chain then moved to wholly untouched
  `4a9a43e4 -> 4a9a43e6`, target door `8`, with three reference views at
  `0.901163` cumulative coverage, three query views at `0.657407`, and sibling
  door `27`. The unchanged scale-prior plus extent-guard chain failed all
  `3/3` fresh queries: minimum IoU `0.059670`, mean `0.275421`, and the guard
  fired `0/3`. Record
  `L10_3RSCAN_SCALE_EXTENT_GUARD_FRESH_CONFIRMATION_GATE_NOT_MET`: one global
  median reference scale does not represent this target's multimodal view
  scale, so the consumed `9/9` result does not generalize as top-one ranking.
  A single failure-driven structural successor keeps the disagreement instead
  of hiding it: a truth-blind set of at most three proposals, contributed by
  layer-18 local appearance, equal semantic/local fusion, and nearest
  reference-scale mixture. Across the four consumed families it covers
  `12/12` queries with mean set size `2.0`, minimum best-proposal IoU
  `0.507007`, and mean best-proposal IoU `0.830661`; the individual mechanisms
  cover only `7/12`, `8/12`, and `10/12`. Record
  `L10_3RSCAN_TRI_EVIDENCE_HYPOTHESIS_SET_FOUR_FAMILY_DEVELOPMENT_GATE_MET`.
  This is bounded proposal-recall Development, not top-one localization,
  calibrated uncertainty, or fresh confirmation. Freeze the set unchanged on
  a fifth untouched family; if it transfers, use cross-view belief or active
  observation to collapse the set rather than another global scalar ranker.
  UncOS and Latent-MaskRCNN independently motivate retaining multiple structured
  interpretations under ambiguity; they do not validate this set or its size.
  That fifth family was frozen before pixels as
  `4acaebc0-6c10-2a2a-852e-0226d6539299 ->
  185d741b-3698-223c-8ba0-48db6ecbe220`, target door `4`. The source gate
  admitted three reference views with `0.853837` cumulative target coverage,
  three query views with `0.765778`, and sibling door `35`. The unchanged
  tri-evidence set failed fresh confirmation at `2/3`: minimum best IoU
  `0.072349`, mean best IoU `0.534691`, and mean/max set size `1.667/2`. In the
  missed view all three contributors collapsed onto the same wrong tile even
  though a full proposal at IoU `0.735693` remained reachable. Record
  `L10_3RSCAN_TRI_EVIDENCE_HYPOTHESIS_SET_FRESH_CONFIRMATION_GATE_NOT_MET`.
  This is correlated ranking failure, not source unreachability.
  A consumed structural successor drops the weakest standalone local top one
  and adds truth-blind cross-view reference consensus: the three query fusion
  winners vote for one target-reference view, and each query may retain its
  best fusion candidate constrained to that reference. Across five consumed
  families it covers `15/15`, with mean/max set size `1.733/3`, minimum best IoU
  `0.507007`, and mean best IoU `0.803731`. Record
  `L10_3RSCAN_CROSS_VIEW_REFERENCE_CONSENSUS_FIVE_FAMILY_DEVELOPMENT_GATE_MET`.
  Nassar et al. and CoMatcher independently motivate joint multi-view evidence
  and consistency; they do not validate these metrics. Freeze this rule for a
  sixth untouched family. A future pass would establish bounded proposal recall
  only; choosing one referent still requires actual cross-view belief or active
  disambiguation.
  The sixth family was frozen before source download and model execution as
  `4d3d829e-8cf4-2e04-8318-b76f02d91c93 ->
  4d3d82a0-8cf4-2e04-800f-97deb20e860b`, target door `10`. The source gate was
  positive: three reference views covered `0.983495` cumulative target vertices,
  three query views covered `0.999151`, and sibling door `24` was independently
  visible. The unchanged consensus set nevertheless failed fresh confirmation
  at `2/3`, with proposal opportunity also only `2/3`: CVR1's best reachable
  detector proposal was IoU `0.433665`. Minimum/mean best set IoU were
  `0/0.499487`. Record
  `L10_3RSCAN_CROSS_VIEW_REFERENCE_CONSENSUS_FRESH_CONFIRMATION_GATE_NOT_MET`.
  This localizes the failure to the proposal carrier, not source observability
  or evidence against cross-view ranking.
  A first consumed elongated-target successor replaced one contributor with a
  reference-scale axis-completion candidate. It repaired the sixth family to
  `3/3` and CVR1 to IoU `0.728335`, but regressed the six-family replay to
  `16/18`; record its role as not met. The next structural successor instead
  preserves all original fusion, scale-mixture, and reference-consensus
  candidates, deduplicates them, and fills only an otherwise unused slot with
  one horizontal/vertical axis completion. It reached `18/18` across all six
  consumed families, maximum set size `3`, mean set size `2.5`, minimum best IoU
  `0.507007`, and mean best IoU `0.800938`; CVR1 remained repaired at
  `0.728335`. Record
  `L10_3RSCAN_AXIS_COMPLETION_VACANCY_SIX_FAMILY_DEVELOPMENT_GATE_MET`.
  Dynamic Tiling and ARC-RCNN motivate adaptive spatial/aspect completion only;
  they do not validate this rule. Freeze the bounded vacancy rule unchanged on
  a seventh untouched family. A pass would still establish proposal recall only,
  not selection of one referent or any downstream portal/navigation authority.
  Correspondence still
  proves no named entrance, ownership, access, waypoint, arrival, or handoff.
- **PanoLab active observation:** entrance-ray recovery passed `4/4`. This
  authorizes an entrance ray geometrically, not a pixel portal or arrival.
- **PanoLab referent-candidate router:** the earlier exact-token/appearance
  router reached `2/2` only posthoc on consumed Rumillat/Halle Rebatet. A harder
  three-producer, fully selected-pixel-unseen Caen panel then exposed the real
  gap: fixed appearance was `1/0/2` and exact lexical evidence added nothing.
  One structural successor, frozen after those failures, accepts only
  high-confidence roster-unique tokens: exact or edit distance one, minimum
  length six, two evidence units, with a length-eight token worth two. It changed
  the consumed panel to `3/0/0`, with `0/30` wrong-target matches. The decisive
  fresh check selected two further target-way/item-disjoint targets by metadata
  before pixels or OCR. The fixed lexical branch recovered Maison de Quartier
  (`maison + chemin`) and returned `NO_MATCH` for Monoprix; the unchanged
  conditional appearance fallback recovered Monoprix. Combined result: `2/0/0`
  in a 13-target closed roster across two capture producers, with `0/24`
  wrong-target lexical matches, `0/12` appearance wrong-goal candidates, and
  zero ownership bindings. A separate first governed OCR replay on four
  exact-roster-absent controls across three cities returned `UNKNOWN` `4/4`,
  with `0/52` fuzzy-token matches. This is fresh same-provider Development
  routing evidence, not open-world confirmation. A consumed progressive replay
  retained `2/2` while using the first frame of each sequence (`6 -> 2` frames,
  `-66.67%`), but this is posthoc efficiency mechanism evidence. The first
  provider/city-disjoint OCR transfer then failed honestly: four human-window-
  conditioned Mapillary panoramas across Rotterdam/Den Haag yielded `0/0/4/0`
  correct/wrong/`UNKNOWN`/ambiguous and `0/64` wrong-target matches. Visible
  Markthal/Ontmoetingskerk credentials were only partially or incorrectly read;
  freeze this as observation-reachability failure, not permission to loosen the
  edit rule and not a full combined-router result.
- **RoadTextVQA goal-conditioned text verifier:** the official KartaView MCP
  source preflight rejected another geometric-nearby cohort: public
  `nearby_photos` has no sign/object visibility authority, while
  `object_search` exits without a user-owned `x-karta-token`. A separate public
  driving-video source then isolated the lexical design problem. A first fresh
  30-video roster-information replay improved correct `9 -> 15` and UNKNOWN
  `15 -> 3`, but produced one wrong and 11 ambiguous episodes; a second fresh
  panel using video-document IDF from 2,635 disjoint training videos improved
  correct `10 -> 15` and UNKNOWN `19 -> 10`, but still produced one wrong and
  four ambiguous episodes. Audit showed that these were often genuinely
  co-visible businesses, so global mutually-exclusive scene classification was
  the wrong contract. The fixed successor verifies only the user-selected goal.
  On a third, 30-video OCR-unseen panel it improved correct/UNKNOWN
  `6/24 -> 19/11` (`20.0% -> 63.3%`, `+43.3 pp`, `+13` correct), accepted
  `0/30` hash-derived negative queries, emitted zero wrong-goal candidates and
  zero identity/portal bindings, and met
  `L10_ROADTEXTVQA_GOAL_CONDITIONED_BACKGROUND_IDF_VERIFIER_DEVELOPMENT_GATE_MET`.
  One of 30 cyclic cross-video challenges matched (`canary` was actually visible
  in the challenged video); because target absence is not independently
  annotated, this remains a collision diagnostic, not a false-positive rate. A
  same-line compact-phrase successor then used the retained OCR boxes without
  changing the IDF branch. On the consumed panel it added `U + HAUL` and
  `HIGH + END` (`19/30 -> 21/30`) with `0/30` negatives. The fixed fresh
  multi-token panel added only `VAPOR + IN`, moving `20/30 -> 21/30` (`70.0%`)
  with `0/30` generated two-token accepts. This is a reproducible fresh gain but
  missed the preregistered minimum gain of two, so
  `L10_ROADTEXTVQA_FRESH_GOAL_CONDITIONED_LAYOUT_PHRASE_VERIFIER_DEVELOPMENT_GATE_NOT_MET`;
  freeze its box/layout thresholds.
- **HierText exhaustive-truth carrier verifier:** official HierText test pixels
  and complete word polygons supplied the independent target-present/target-
  absent authority missing from RoadTextVQA. The first multiscale panel improved
  correct carrier `9 -> 13` with `0/30` truth-absent accepts but exposed two
  wrong carriers. A second fresh panel showed that exact query spans inside
  merged OCR lines are the decisive representation: correct carrier
  `7 -> 22`, but two new errors came from text-identical instances and one from
  a truncated truth fragment. The frozen successor therefore requires a target
  to occur exactly once across all annotated word nodes, rejects targets that
  are substrings of longer truth tokens, projects a unique query span onto the
  OCR quadrilateral, and abstains when best-rank observations form multiple
  spatial carrier components. On a third disjoint 30-image test cohort it
  changed correct/wrong/`UNKNOWN` `11/0/19 -> 22/0/8` (`36.7% -> 73.3%`,
  `+36.7 pp`, `+11` correct) and accepted `0/30` complete-truth-absent queries.
  Decision:
  `L10_HIERTEXT_TEST_UNIQUE_REFERENT_SPAN_CARRIER_DEVELOPMENT_GATE_MET`.
  This is static OCR carrier Development evidence, not open-world instance
  identity, facade/portal ownership, arrival, or handoff.
- **FSNS multi-view street-sign generalization:** official French Street Name
  Signs provides up to four different-position/time Street View observations of
  one physical sign and its canonical street-name truth. A 30-example official
  testdata development panel changed single-first-view exact recognition
  `26/30 -> 30/30`, but its preregistered `+5` gain gate was impossible once the
  baseline reached 26 and therefore did not pass. The unchanged mechanism was
  then frozen on 40 pixel-unseen validation-shard signs whose image hashes,
  canonical labels, and distinctive goal tokens were all disjoint from all 50
  testdata examples. Accumulating goal-conditioned evidence across four views
  changed correct/`UNKNOWN` `34/6 -> 40/0` (`85% -> 100%`, `+15 pp`, `+6`
  correct); all six recoveries were exact, with first acceptance at view 2 for
  three signs and view 3 for three. Both 40 canonical-label-disjoint challenges
  and 40 synthetic negatives had zero accepts. Decision:
  `L10_FSNS_FRESH_MULTIVIEW_GOAL_EVIDENCE_DEVELOPMENT_GATE_MET`. This proves a
  cross-source multi-view canonical sign-reading mechanism, not that an L10
  action caused the view, nor facade/portal association, arrival, or handoff.
- **CATALIST controlled camera actions:** Exa source discovery admitted the
  official CATALIST validation split because each video records an executed
  tripod-camera transformation and a manually verified video-level text label.
  A metadata-only selector froze 30 validation videos balanced across
  translation/pan/tilt/zoom/roll. Their 12 complete labels and 12 conditioned
  target tokens were absent from every training label/token. The unchanged FSNS
  exact-plus-two-frame-edit rule changed first-frame correct/`UNKNOWN`
  `27/3 -> 29/1` (`90.0% -> 96.7%`, `+6.7 pp`); both recoveries occurred under
  pan, with `0/30` label-disjoint challenge accepts and `0/30` synthetic
  accepts. It missed the frozen minimum gain of five, so
  `L10_CATALIST_CONTROLLED_ACTION_GOAL_RECOVERY_DEVELOPMENT_GATE_NOT_MET`.
  A consumed-cache two-token diagnostic added `0`; do not open the remaining
  validation labels for the same representation. CATALIST supplies executed
  action classes but not direction, metric pose, stationary counterfactuals,
  exhaustive pixels, facade/sign ownership, or entrance truth.
- **3D Street View center-target lock:** Exa source discovery found the official
  3D Street View same-physical-point benchmark. A rate-limited 181,018,624-byte
  archive prefix still yielded 6,220 fully decodable real images and 2,606
  provider-labelled pairs after a deterministic endpoint-prefix split: 2,180
  train pairs and 426 test pairs, with 957 cross-partition pairs discarded. A
  frozen DINOv2-small successor uses the provider contract that the verified
  target point lies at the optical center: keep target patches unchanged and
  scale-gate only full-image center crops. Against one global embedding, held-
  out AUROC improved `0.949859 -> 0.984860` (`+3.5001 pp`), balanced accuracy
  `0.895773 -> 0.926341` (`+3.0568 pp`), average precision
  `0.954445 -> 0.982243`, and 32-anchor retrieval Top-1
  `0.843750 -> 1.000000` (`+15.625 pp`). The difficult image/patch subgroup
  improved AUROC `0.874644 -> 0.966274`. Decision:
  `L10_3DSTREETVIEW_CENTER_TARGET_LOCK_DEVELOPMENT_GATE_MET`. This is real-
  image, same-provider, archive-prefix Development evidence for locking a
  provider-verified physical point. It does not identify that point as a door,
  entrance, venue-owned portal, access route, waypoint, arrival, or handoff.
- **3RScan closed-roster stable-door assignment:** the unchanged center-scale
  score first failed on three consumed SceneNN doors whose truth proposals were
  edge-clipped (`2/6` Top-1; AUROC `0.444444`). A fully visible 3RScan transfer
  then reached global `5/6` and center-scale `4/6`, exposing a sibling-door
  collision. Parameter-free one-to-one assignment repaired that consumed matrix
  to an equivalent `6/6` with `0.246219` assignment margin. Before further RGB,
  a confirmation cohort excluded every prior physical
  `(reference_scan_id,target_instance_id)` and froze three stable doorframes
  under the same `32 px` visibility rule. Pair AUROC/AP, independent Top-1, and
  assigned identity reached `1.0/1.0/6-of-6/3-of-3`; the complete-assignment
  margin was `0.419960`. Decision:
  `L10_3RSCAN_ROSTER_ASSIGNMENT_TARGET_DISJOINT_DEVELOPMENT_GATE_MET`. This is
  truth-proposed, single-provider, closed-roster Development only.
  Inheritance is explicit: 3D Street View scoring is `RETAINED_CORE`; the
  SceneNN edge-clipping signature is `NEGATIVE_CONTROL`; the 3RScan pair score
  is `COMPONENT_OR_CHALLENGER / COMPONENT`; and one-to-one roster assignment is
  `RETAINED_CORE` only for externally justified closed rosters. The two zero-RGB
  pre-freeze attempts remain working material rather than `DEAD_FOR_THIS_ROLE`.
- **3RScan partial-roster zero assignment:** appearance-only multiview reciprocal
  matching failed on four new physical targets across closed, extra, missing, and
  balanced-swap scenarios (`TP=4, FP=5, FN=8`, F1 `0.380952`; balanced swap
  `0` true/`2` false) and is frozen as `NEGATIVE_CONTROL`. Registered target-
  surface ranking then recovered `12/12` true matches on four further targets but
  forced one adjacent doorframe-to-door match (F1 `0.96`, exact unmatched sets
  `3/4`); it remains `COMPONENT_OR_CHALLENGER / COMPONENT`. On a third fresh
  cohort, two predeclared registration witnesses set a `0.038756 m` maximum-
  residual ceiling. The unchanged surface score plus that rejector reached
  `TP=12, FP=0, FN=0`, F1 `1.0`, and exact unmatched sets `4/4`, versus complete-
  assignment F1 `0.96`. The gate met, but rank-only also reached F1 `1.0` on
  this cohort, so witness calibration is `COMPONENT_OR_CHALLENGER / CHALLENGER`,
  not proven incremental value. All three results are privileged registered-
  geometry, single-provider Development only.
- **3RScan absolute-support upgrade:** a scan-family-disjoint six-door test with
  all twelve ordered balanced swaps gave rank-only `TP=64, FP=2, FN=0`, F1
  `0.984615`. The unchanged raw two-witness ceiling removed both false matches
  but reduced true matches to `48`, added `16` misses, and lowered F1 to
  `0.857143`; that ceiling is now `DEAD_FOR_THIS_ROLE` as a family-general
  rejector. A new parameter-free successor froze six unconsumed same-family
  `doorframe` targets and all thirty ordered swaps. Positive registered portal-
  plane extent overlap upgraded rank-only `136/4/0`, F1 `0.985507`, exact
  unmatched sets `29/33`, to `136/0/0`, F1 `1.0`, and `33/33`, with no
  true-match loss. It is `RETAINED_CORE` for privileged registered-geometry
  partial rosters. Kept unchanged on a newly materialized scan family, it again
  improved rank-only `34/2/0`, F1 `0.971429`, exact unmatched `13/15`, to
  `34/0/0`, F1 `1.0`, and `15/15`. This is scan-family-disjoint but still
  same-provider confirmation.
- **SEVN address-door backend:** the portal-private PP-OCRv6 medium witness met
  every frozen gate on 40 further fresh PAN episodes with zero address or frame
  overlap against all 205 addresses and 220 frames used by earlier panels.
  Visible-number exact OCR improved `22/37 -> 29/37` (`+18.92 pp`) and
  correct/wrong/`UNKNOWN` binding improved `16/0/24 -> 21/0/19`. All five new
  bindings were correct, both ambiguous witness sets abstained, and binding
  precision remained `100%`. This is same-source SEVN Development evidence.
- **Metric portal extent:** 3RScan registered extent established a strong
  synthetic/registered Development ceiling. The latest source-distinct spatial
  mask reached `0.5403` complete IoU and `0.422 m` centroid error, but stayed
  below the `60%` ceiling-retention gate and confused an overlapping doorframe.
  Exact-instance and portal-set binding remain the information gap.

## Next admissible work

1. Freeze the SEVN portal-private medium mechanism. The next decision-changing
   check is a genuinely source-disjoint provider/city with independent
   door-instance and address-credential truth, including negative or no-portal
   controls. Do not tune the V5 panel, private crops, OCR models, witness
   uniqueness, or abstention after observing its result.
2. For metric portals, add exact-instance or portal-set authority before
   another endpoint-mask successor; do not reinterpret overlap with a nearby
   frame as a correct entrance.
   The 3D Street View center-target result establishes a stronger real-image
   appearance lock but supplies no portal class or ownership truth. Freeze its
   split, crop scales, score weights, model, and test groups. The 3RScan result
   now adds physically target-disjoint stable-door identity, closed-roster
   assignment, and parameter-free registered planar-extent support for missing/
   extra candidates. Freeze all SceneNN/3RScan targets, score matrices, and the
   exact positive-overlap predicate; the raw two-witness ceiling is terminal for
   this role. Scan-family-disjoint confirmation is now positive. The next
   admissible test is provider-disjoint confirmation or an independently
   justified phone-side target-conditioned planar-support carrier. Do not infer
   open-world identity, ownership, access, aperture, arrival, or handoff.
3. Freeze the distinctive-token/conditional-appearance router and the failed
   Mapillary MR rows. The next decision-changing source is a new provider/city-
   disjoint sequence panel with independent exact-target truth, visual reference
   views, and target-absent controls. Develop provider-normalized/character-
   sequence text observation only on a separate cohort; do not tune edit
   distance, suffix rules, crop scale, or OCR preprocessing on MR. Pre-register
   progressive observation length on the new panel; current fixed thresholds
   have no conformal/FDR authority. A metadata-only KartaView/Jakarta candidate
   is now frozen in
   [l10_kartaview_jakarta_provider_disjoint_candidate_v1.json](l10_kartaview_jakarta_provider_disjoint_candidate_v1.json):
   two targets, two query contributors/sequences, two contributor-disjoint
   geometric negative-control candidates, and zero selected pixels at freeze.
   Its exact materialization successor is now frozen in
   [l10_kartaview_jakarta_materialization_v1.json](l10_kartaview_jakarta_materialization_v1.json):
   `8/8` provider-identity/hash-verified images (`6` query-window plus `2`
   geometric-negative candidates) across four sequences and four contributors,
   with zero OCR/router/appearance calls. The subsequent full-frame Codex visual
   audit found target visibility in `0/6` query frames and provisionally verified
   target absence in both negative candidates, so the source is
   `L10_KARTAVIEW_JAKARTA_CODEX_VISUAL_SOURCE_AUDIT_NOT_EVALUABLE`. This is an
   observation-reachability failure: freeze both neighbourhoods and require
   source-side facade/sign visibility authority on the next provider/city panel.
   The official `mcp-karta-view` source was pinned and reviewed under
   `E:/codex-tools`: public `nearby_photos` still supplies geometry only and
   `object_search` requires a user-owned token, so no new KartaView cohort was
   admitted. Freeze that source decision. The RoadTextVQA successor now supplies
   a stronger goal-conditioned text-observation mechanism, and HierText test
   supplies a source-disjoint static carrier result with independent exhaustive
   target-present/target-absent word truth. Freeze all three HierText panels.
   FSNS now supplies a source-distinct multi-view canonical street-sign result.
   CATALIST adds controlled action labels and two honest pan recoveries, but its
   already-readable starts leave the frozen gain gate unmet and it still lacks
   direction, metric pose, facade association, and entrance truth. Freeze its
   selected panel and the no-gain phrase diagnostic.
   The next authority-changing check is a pose/action-labelled street sequence
   that can test whether an L10-chosen observation action causes recovery and
   links the recovered sign to the exact target facade/entrance. Static HierText
   and pre-cropped FSNS do not supply portal ownership, action utility, arrival,
   or handoff.
4. Freeze the six-family axis-completion vacancy rule on a seventh untouched
   3RScan family before source download, RGB inspection, or model execution.
   Keep the set at no more than three and do not tune its `0.05` scale penalty,
   axis modes, vacancy condition, or IoU gate. If it transfers, the next layer
   must resolve the retained set with actual cross-view belief or active
   observation; do not report proposal recall as top-one referent localization.
5. Keep active actions tied to the actual deficit: `APPROACH`, `SIDESTEP/PAN`,
   `SWEEP`, or `HOLD`. An action proposal is not an arrival or handoff.

## Stop and claim boundary

- Generic Panoramax pixel-portal mining and the consumed SceneFun3D ordinal
  source are closed.
- `UNKNOWN` and `NOT_EVALUABLE` are neither failure nor known-safe.
- Synthetic, registered, replay, and curated Development results do not prove
  natural-camera performance, user benefit, navigation, reliability, or
  safety.
- Device/demo integration is not reopened by an algorithm-only result.

## Detail and evidence

- Detailed route ledger and reproduction commands: [README.md](README.md)
- SEVN portal-private medium witness result:
  [l10_sevn_ppocrv6_medium_portal_witness_result_v1.json](l10_sevn_ppocrv6_medium_portal_witness_result_v1.json)
- PanoLab active-ray result:
  [l10_panolab_active_ray_recovery_result_v2.json](l10_panolab_active_ray_recovery_result_v2.json)
- Fresh target-absent abstention result:
  [l10_panolab_open_set_router_result_v1.json](l10_panolab_open_set_router_result_v1.json)
- Fresh cross-collection positive result:
  [l10_panolab_cross_collection_positive_router_result_v1.json](l10_panolab_cross_collection_positive_router_result_v1.json)
- Posthoc temporal lexical-appearance router result:
  [l10_panolab_temporal_lexical_appearance_router_result_v1.json](l10_panolab_temporal_lexical_appearance_router_result_v1.json)
- Fresh distinctive-token/conditional-appearance result:
  [l10_panolab_distinctive_token_fresh_combined_router_result_v1.json](l10_panolab_distinctive_token_fresh_combined_router_result_v1.json)
- Distinctive-token open-set negative result:
  [l10_panolab_distinctive_token_open_set_negative_result_v1.json](l10_panolab_distinctive_token_open_set_negative_result_v1.json)
- Mapillary provider-transfer result:
  [l10_mapillary_distinctive_token_provider_transfer_result_v1.json](l10_mapillary_distinctive_token_provider_transfer_result_v1.json)
- KartaView public-MCP capability preflight:
  [l10_kartaview_public_mcp_capability_preflight_v1.json](l10_kartaview_public_mcp_capability_preflight_v1.json)
- RoadTextVQA goal-conditioned IDF result:
  [l10_roadtextvqa_goal_conditioned_idf_verifier_result_v1.json](l10_roadtextvqa_goal_conditioned_idf_verifier_result_v1.json)
- RoadTextVQA fresh layout-phrase result:
  [l10_roadtextvqa_layout_phrase_fresh_result_v1.json](l10_roadtextvqa_layout_phrase_fresh_result_v1.json)
- HierText unique-referent query-span carrier result:
  [l10_hiertext_test_unique_referent_span_verifier_result_v1.json](l10_hiertext_test_unique_referent_span_verifier_result_v1.json)
- FSNS fresh multi-view goal result:
  [l10_fsns_multiview_goal_verifier_result_v1.json](l10_fsns_multiview_goal_verifier_result_v1.json)
- Consumed progressive early-exit result:
  [l10_panolab_progressive_evidence_early_exit_posthoc_result_v1.json](l10_panolab_progressive_evidence_early_exit_posthoc_result_v1.json)
- Latest 3RScan spatial-mask result:
  [l10_3rscan_spatial_reference_mask_result_v1.json](l10_3rscan_spatial_reference_mask_result_v1.json)
- 3D Street View center-target lock result:
  [l10_3dstreetview_center_target_lock_result_v1.json](l10_3dstreetview_center_target_lock_result_v1.json)
- 3RScan pairwise door-transfer result:
  [l10_3rscan_center_target_door_retrieval_result_v2.json](l10_3rscan_center_target_door_retrieval_result_v2.json)
- 3RScan roster-assignment posthoc result:
  [l10_3rscan_roster_assignment_posthoc_result_v1.json](l10_3rscan_roster_assignment_posthoc_result_v1.json)
- 3RScan physically target-disjoint assignment result:
  [l10_3rscan_roster_assignment_confirmation_result_v2.json](l10_3rscan_roster_assignment_confirmation_result_v2.json)
- Fresh full-context mask-gated RoMa confirmation:
  [l10_scenenn_roma_full_context_mask_fresh_result_v1.json](l10_scenenn_roma_full_context_mask_fresh_result_v1.json)
- Grounded-SAM proposal-mask bridge result:
  [l10_scenenn_roma_grounded_sam_proposal_posthoc_result_v1.json](l10_scenenn_roma_grounded_sam_proposal_posthoc_result_v1.json)
- 3RScan Grounded-SAM source-reachability result:
  [l10_3rscan_roma_grounded_sam_proposal_posthoc_result_v1.json](l10_3rscan_roma_grounded_sam_proposal_posthoc_result_v1.json)
- Sixth-family fresh cross-view consensus result:
  [l10_3rscan_cross_view_reference_consensus_confirmation_result_v1.json](l10_3rscan_cross_view_reference_consensus_confirmation_result_v1.json)
- Six-family axis-completion vacancy Development result:
  [l10_3rscan_axis_completion_vacancy_posthoc_result_v1.json](l10_3rscan_axis_completion_vacancy_posthoc_result_v1.json)
- 3RScan reference-conditioned cycle-prompt result:
  [l10_3rscan_roma_cycle_prompt_sam_posthoc_result_v1.json](l10_3rscan_roma_cycle_prompt_sam_posthoc_result_v1.json)
- 3RScan native-reference-mask ablation result:
  [l10_3rscan_roma_cycle_prompt_sam_reference_mask_posthoc_result_v1.json](l10_3rscan_roma_cycle_prompt_sam_reference_mask_posthoc_result_v1.json)
- 3RScan dual-surface proposal result:
  [l10_3rscan_roma_cycle_prompt_dual_surface_posthoc_result_v1.json](l10_3rscan_roma_cycle_prompt_dual_surface_posthoc_result_v1.json)
- 3RScan first physical-target-disjoint dual-surface result:
  [l10_3rscan_roma_cycle_prompt_dual_surface_confirmation_result_v1.json](l10_3rscan_roma_cycle_prompt_dual_surface_confirmation_result_v1.json)
- 3RScan four-surface physical-target-disjoint result:
  [l10_3rscan_roma_cycle_prompt_four_surface_confirmation_result_v1.json](l10_3rscan_roma_cycle_prompt_four_surface_confirmation_result_v1.json)
- 3RScan coherent-cycle physical-target-disjoint confirmation:
  [l10_3rscan_cycle_component_support_confirmation_result_v1.json](l10_3rscan_cycle_component_support_confirmation_result_v1.json)
- 3RScan coherent-cycle cross-scene open-set result:
  [l10_3rscan_cycle_component_open_set_posthoc_result_v1.json](l10_3rscan_cycle_component_open_set_posthoc_result_v1.json)
