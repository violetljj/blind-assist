# L10-R0 current

Updated: 2026-08-31

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
  veto roles. Retain rank-only registered-surface matching only as a Development
  component; next work needs calibrated dense certainty and spatially balanced
  correspondence on fresh targets, not matcher-threshold tuning.
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
4. Keep active actions tied to the actual deficit: `APPROACH`, `SIDESTEP/PAN`,
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
