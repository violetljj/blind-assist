# DTR-R2 current

Updated: 2026-09-01

Status: `DTR_R2_DYNAMIC_RETAINED`

## Question

Can BlindAssist emit stable `ONSET / HOLD / ESCALATE / CLEAR` events when
future obstacle occupancy intersects the wearer's route, while preserving
`UNKNOWN` instead of turning missing evidence into safety?

## Current decisions

- **Public/JRDB line:** X21 transports only a component already authorized by
  raw X13 birth and the same live track. Its six-sequence replay reached `5/6`
  CONTACT, 11 false segments, 45.45% Event F1, `3.061 s` median lead, and
  `8/18` dropout recovery. This is a same-source Development pass only.
- **CARLA algorithm line:** X65 pooled `+15 TP / +0 FP / +1.44 pp F1` over X64
  across consumed C26/C27/C28/C32 Development. C33 then terminated as frozen
  source-not-evaluable before any prediction. The sole C34 scored invocation
  completed on genuinely new pixels at `83.01/73.84/78.15` percent
  precision/recall/F1 with all authority invariants zero and acceptable safe
  segments. It improved X54 by `-9 FP / +2.11 pp F1`, but tied X64 exactly.
  Although C34 contained 17 selected contact-loss ambiguity frames and 16
  pre-conflict joint-credential frames, X65 recorded zero ancestry
  synchronization and zero handback frames. C34 is therefore
  mechanism-not-exercised, not incremental X65 confirmation; its 83.01%
  precision also missed the frozen 85% floor. Consumed diagnosis then produced
  X67, which separates existence from route-risk authority only after a dormant
  track was reactivated and lost again beyond the inherited measurement hold
  horizon with receding direction-only motion. X68 then preserves each surface
  footprint but uses a same-direction, lateral-nonexpanding object-local metric
  velocity to remove lattice-quantized near-miss motion. Across
  C26/C27/C28/C32/C34, X68 improved four cohorts and was classification-neutral
  on one: pooled `636 TP / 63 FP / 227 FN`, or `90.99/73.70/81.43%`, for
  `0 TP / -15 FP / +0.77 pp F1` over X67. C34 reached
  `88.19/73.84/80.38%`. X69 then allows a current X25 object-local rigid
  footprint to falsify only mature, measured cross-route surface ambiguity
  after the inherited 1.0 s history window. It improved every cohort, removing
  another 12 false positives with no true-positive loss. Pooled X69 is
  `636 TP / 51 FP / 227 FN` at `92.58/73.70/82.06%`; C34 is
  `90.71/73.84/81.41%`. X70 then gives an X25 rigid identity a collision
  credential only when current X69 surface, X25 rigid-footprint, and X24
  metric-point risk spatially agree. That identity may hand risk back across a
  current surface dropout, while X69 explicit contradiction release retains
  precedence. X70 recovered four true positives with no false-positive cost:
  pooled `640 TP / 51 FP / 223 FN` at `92.62/74.16/82.37%`; C34 is
  `90.78/74.42/81.79%`. X71 then permits an object-local occupancy birth when a
  current X24 metric point lies inside a same-class X25 rigid footprint, their
  route-forward motion directions agree, and the representations remain
  associated at their later predicted route-entry time. It recovered three
  more true positives with no false-positive cost: pooled
  `643 TP / 51 FP / 220 FN` at `92.65/74.51/82.59%`; C34 is
  `90.97/76.16/82.91%`. X72 then lets a current X25 collision footprint
  complete a still-live credentialed surface parent only when it intersects
  measured fragments at their boundary while its rigid center lies inside none
  of that parent's fragments. X72 recovered another eight true positives with
  no false-positive cost: pooled `651 TP / 51 FP / 212 FN` at
  `92.74/75.43/83.19%`; C34 is `91.22/78.49/84.38%`. This is cross-cohort
  non-regressing Development, not fresh X72 confirmation. X73 then
  reconstructs the convex hull of all current measured fragments belonging to
  one still-live credentialed surface parent and transports it with their
  area-weighted current velocity. Any current measured X25 center contained by
  a parent fragment vetoes reconstruction. X73 recovered 12 more true
  positives with no false-positive cost across four cohorts: pooled
  `663 TP / 51 FP / 200 FN` at `92.86/76.83/84.08%`; C34 remained
  classification-neutral at `91.22/78.49/84.38%`. This is consumed
  cross-cohort Development. The sole C35 scored invocation then tested
  unchanged X73 on genuinely new seed-, render-domain-, and pixel-disjoint
  CARLA evidence. X73 improved X72 from `126/18/46` to `132/18/40`
  TP/FP/FN, or from `87.50/73.26/79.75%` to
  `88.00/76.74/81.99%` precision/recall/F1: `+6 TP / +0 FP / +2.24 pp F1`.
  Parent-hull reconstruction was exercised on six frames, all primary transfer
  checks passed, and all required authority invariants remained zero. Accept
  `DTR_CARLA_C35_X73_GENERALIZATION_GATE_MET` as source-disjoint synthetic
  Development confirmation within the frozen same-map/scripted boundary. C35
  post-confirmation diagnosis then exposed six false-positive frames from one
  X57 metric handback whose stale `truck` identity disagreed with the nearest
  current, non-route X25 `person` footprint. X74 clears only when every
  confirmed carrier is such a metric handback, the nearest current measured
  rigid footprint is inside the inherited association radius, that footprint
  is not itself a route candidate, and its detector class differs. Across
  consumed C26/C27/C28/C32/C34/C35, X74 changed only those six C35 false
  positives: pooled `795 TP / 63 FP / 240 FN` at
  `92.66/76.81/83.99%`, or `0 TP / -6 FP / +0.27 pp F1` over X73. The other
  five cohorts were classification-neutral and all required authority
  invariants remained zero. X74 is the strongest current six-cohort CARLA
  Development arm, but only C35 exercised it. The sole C36 scored invocation
  then tested unchanged X74 under a new seed and four new render assignments.
  X74 and X73 were classification-identical at
  `143 TP / 44 FP / 29 FN`, or `76.47/83.14/79.67%`; X74 recorded zero class
  contradiction releases, so the incremental mechanism was not exercised and
  is not freshly confirmed. All required authority invariants and contact/safe
  constraints passed, but precision missed the frozen 85% floor. C36 is now
  consumed diagnosis material for the line's render-domain false-alert gap.
  X75 then separates object-existence memory from collision-risk authority: an
  occupancy-peak permanence belief may retain route risk only if its parent
  previously earned spatially agreeing surface + X25 + X24 collision
  credentials, or if its transport history contains a contradiction that
  warrants conservative retention. Across consumed
  C26/C27/C28/C32/C34/C35/C36, X75 removed 19 C36 false positives with no TP
  loss and changed no earlier cohort: pooled `938 TP / 88 FP / 269 FN` at
  `91.42/77.71/84.01%`, or `0 TP / -19 FP / +0.71 pp F1` over X74. C36 alone
  rose to `85.12/83.14/84.12%`, restoring its precision floor. All required
  authority invariants and contact/safe constraints passed. X75 is the
  strongest current seven-cohort Development arm. The sole C37 invocation then
  exercised X75 once on a new seed and four new render assignments. X75 changed
  X74 from `133 TP / 35 FP / 39 FN` to `133 / 34 / 39`, a fresh
  `0 TP / -1 FP / +0.23 pp F1` effect. All incremental, authority, contact, and
  safe constraints passed, but full-arm precision was `79.64%`, below the
  frozen 85% floor. The incremental direction therefore has fresh positive
  evidence, while the full X75 generalization gate did not pass. Across all
  eight consumed cohorts, X75 is `1,071 TP / 122 FP / 308 FN` at
  `89.77/77.67/83.28%`, a cumulative `0 TP / -20 FP / +0.64 pp F1` over X74.
  C37 then exposed a parent-hull transport contradiction: ten false-positive
  frames declared zero-shift support while retaining nonzero reconstructed
  velocity. X76 rejects only that all-carrier inconsistency. It changes no
  earlier cohort and moves C37 to `133 TP / 24 FP / 39 FN` at
  `84.71/77.33/80.85%`, or `0 TP / -10 FP / +2.38 pp F1` over X75. Across all
  eight cohorts X76 is `1,071 TP / 112 FP / 308 FN` at
  `90.53/77.67/83.61%`. All required constraints pass, but C37 remains 0.29 pp
  below the 85% precision reference. X76 is Development-only; X73 retains
  confirmation authority. X77 then rejects only a metric temporal handoff
  whose forward velocity is positive, meaning its obstacle is already
  receding. Across the eight consumed cohorts, all seven true-positive metric
  handoffs were approaching while all six false-positive handoffs were
  receding. X77 removes those six false positives with zero TP loss: pooled
  `1,071 TP / 106 FP / 308 FN` at `90.99/77.66/83.80%`. C37 reaches
  `133 TP / 23 FP / 39 FN` at `85.26/77.33/81.10%`, crossing the 85% precision
  reference. All required constraints pass. X77 is Development-only; X73
  retains confirmation authority until unchanged X77 passes a fresh gate. The
  sole C38 invocation then tested unchanged X77 on seed `381077` and four new
  render assignments. X77 and X76 were classification-identical at
  `124 TP / 48 FP / 48 FN`, or `72.09/72.09/72.09%`; X77 recorded zero
  receding temporal-handoff releases. C38 is therefore
  mechanism-not-exercised, not negative incremental evidence. All required
  authority invariants and safe-segment constraints passed, but full precision
  and F1 failed and episode 05 contact recall was only `52.17%`. C38 is now
  consumed diagnosis material for a measurement-backed existence/uncertainty
  successor. X78 separates identity continuity from collision-risk authority:
  an all-carrier, zero-contradiction object-permanence belief with zero-shift
  support and non-closing velocity remains in memory but no longer authorizes
  route risk. Across C26/C27/C28/C32/C34/C35/C36/C37, however, the mechanism
  was never exercised; X78 remains identical to X77 at pooled
  `1,071 TP / 106 FP / 308 FN` and `90.99/77.66/83.80%`. This is a compatible
  structural refinement, not an incremental metric result. X79 then assigns
  lateral-only collision timing to the existing X75 triple credential: an
  uncredentialed, conflict-free surface branch may retain identity and lateral
  motion but cannot independently authorize route risk. Across the eight
  consumed cohorts, the credential protects all three lateral-only true
  positives while X79 removes 15 false positives across five cohorts with zero
  TP loss. Pooled X79 is `1,071 TP / 91 FP / 308 FN` at
  `92.17/77.66/84.30%`, a further `+0.49 pp F1` over X78. Every required check
  passes. X79 is the strongest current eight-cohort Development arm; X73
  continues to retain confirmation authority until unchanged X79 passes a
  fresh source-disjoint gate. The sole C39 invocation then tested unchanged
  X79 at seed `391079` with four new weather/render assignments. X79 and X78
  were classification-identical at `137 TP / 22 FP / 35 FN`, or
  `86.16/79.65/82.78%`; X79 recorded zero lateral-only releases. Every full-arm,
  contact-recall, safe-segment, and authority-invariant constraint passed, but
  the frozen incremental mechanism and false-positive reduction requirements
  did not. C39 is mechanism-not-exercised and now consumed diagnosis material;
  it supplies no fresh promotion evidence. X73 therefore still retains the
  latest positive source-disjoint confirmation authority. X80 then requires an
  otherwise uncredentialed X71 entry-cotransport birth to carry a rigid
  footprint whose lateral span strictly exceeds its route-forward span before
  it can authorize cross-route occupancy. This ordinal shape credential adds no
  fitted numeric threshold. Across C26/C27/C28/C32/C34/C35/C36/C37/C39, X80
  changes only C39: it removes six false positives with zero TP loss, moving
  C39 to `137 TP / 16 FP / 35 FN` at `89.54/79.65/84.31%`. Nine-cohort pooled
  X80 is `1,208 TP / 107 FP / 343 FN` at `91.86/77.89/84.30%`, or
  `0 TP / -6 FP / +0.18 pp F1` over X79. Every required check passes. Because
  X80 was designed after C39 opened, this is Development-only and X73 retains
  confirmation authority. X81 then applies the same ordinal cross-route shape
  credential to an uncredentialed zero-shift surface-support carrier. Across
  the same nine consumed cohorts it changes only C26, removing two false
  positives with zero TP loss. Pooled X81 is
  `1,208 TP / 105 FP / 343 FN` at `92.00/77.89/84.36%`, or
  `0 TP / -2 FP / +0.06 pp F1` over X80. Every required check passes. X81 is
  the strongest current nine-cohort Development arm, but its effect was
  designed and measured on consumed C26; X73 retains fresh confirmation
  authority until unchanged X81 passes a later source-disjoint gate. The sole
  C40 invocation then tested unchanged X81 on seed `401081` with four changed
  render assignments. X81 exercised three zero-shift shape releases and
  improved X80 from `129 TP / 28 FP / 43 FN` to `129 / 25 / 43`, or from
  `82.17/75.00/78.42%` to `83.77/75.00/79.14%` precision/recall/F1:
  `0 TP / -3 FP / +0.72 pp F1`. Every incremental, recall, F1, contact, safe,
  and authority constraint passed, but full-arm precision missed the frozen
  85% floor. X81 therefore has fresh positive incremental evidence without a
  complete generalization gate; X73 retains full fresh confirmation authority.
  Consumed C40 diagnosis then exposed a narrower authority failure: multiple
  X72 completion proxies were all carried (`HOLD`) with no current measured
  risk carrier, yet their multiplicity still owned route risk. X82 clears only
  this held-only proxy consensus while retaining single proxies, any current
  measurement, mixed direct carriers, and all track state. Across
  C26/C27/C28/C32/C34/C35/C36/C37/C39/C40 it changes only C40, removing three
  false positives with zero TP loss. C40 moves to `129 TP / 22 FP / 43 FN` at
  `85.43/75.00/79.88%`, crossing the formerly missed precision floor. Pooled
  ten-cohort X82 is `1,337 TP / 127 FP / 386 FN` at
  `91.33/77.60/83.90%`, or `0 TP / -3 FP / +0.08 pp F1` over X81. Every
  required check passes. Because X82 was designed after C40 opened, this is
  Development-only; C40 cannot retroactively confirm X82 and X73 retains full
  fresh authority until a new source-disjoint gate tests frozen X82. C41 then
  froze unchanged X82 at seed `411082` with four changed render assignments.
  Its final four-sensor source passed after the protocol-authorized single
  recovery of an empty, zero-frame witness shard. X81 and X82 were identical at
  `135 TP / 15 FP / 37 FN`, or `90.00/78.49/83.85%`; X82 recorded zero
  held-proxy consensus releases. Full-arm, contact, and safe-segment metrics
  passed, but the mechanism and incremental-FP requirements did not. One
  inherited non-rigid risk reference and one parent-identity mismatch also
  failed the authority gate. C41 is mechanism-not-exercised, supplies no fresh
  incremental promotion or rejection evidence for X82, and is now consumed
  successor-design material. X73 still retains full fresh authority.
  X83 then corrects the exact C41 ownership defect without changing event
  classification: when confirmed references mix eligible `RIGID_DYNAMIC` and
  non-rigid carriers, it returns only the non-rigid references to candidates
  and rebuilds confirmed parent identity from the rigid owners. Across
  C26/C27/C28/C32/C34/C35/C36/C37/C39/C40/C41, X83 changes one C41 frame,
  demotes one static reference, and removes both the non-rigid-reference and
  parent-mismatch defects. Every TP/FP/FN is identical to X82 and all required
  authority invariants are zero in all eleven cohorts. Pooled X83 remains
  `1,472 TP / 142 FP / 423 FN` at `91.20/77.68/83.90%`. This is
  post-hoc Development-only because X83 was designed after C41 opened; X73
  retains fresh authority until a later source-disjoint gate exercises frozen
  X83 without a classification regression. Consumed C41 diagnosis then exposed
  a narrower continuation-authority gap: its three remaining false-positive
  frames were held, forward-closing, direction-consistent continuations whose
  authorized branch hypotheses outnumbered their direct transport anchors.
  X84 releases only that relational partition, while retaining occupancy-peak
  carriers, non-closing motion, anchor-covered continuations, and all track
  evidence. Across the same eleven consumed cohorts, X84 changes only C41:
  `135 TP / 15 FP / 37 FN` becomes `135 / 12 / 37`, with zero TP loss.
  Pooled X84 is `1,472 TP / 139 FP / 423 FN` at
  `91.37/77.68/83.97%`, or `0 TP / -3 FP / +0.07 pp F1` over X83. All
  contact, safe-segment, full-arm, and authority checks pass. This is post-hoc
  Development-only; X73 still retains fresh authority until unchanged X84 is
  exercised by a new preregistered source-disjoint gate. Cross-cohort diagnosis
  then found two C36 frames where X68 current object-local geometry had already
  released surface risk, but X72 reopened it in the same frame using only
  historical boundary-completion proxies. X85 gives the current X68 geometric
  falsifier precedence over that pure X72 reopening while preserving direct
  carriers and later independent evidence. It changes only C36, from
  `143 TP / 23 FP / 29 FN` to `143 / 21 / 29`; pooled X85 is
  `1,472 TP / 137 FP / 423 FN` at `91.49/77.68/84.02%`, or
  `0 TP / -2 FP / +0.05 pp F1` over X84. All required checks pass. X85 is
  consumed Development-only and X73 retains fresh authority. X86 then binds a
  forward-receding X57 metric handback's predicted entry to the inherited X24
  evidence hold window: a route entry later than `0.60 s` cannot remain
  authorized after its supporting measurement authority expires. It removes
  one C32, two C39, and one C41 false-positive frames with zero TP loss. Pooled
  X86 is `1,472 TP / 133 FP / 423 FN` at `91.71/77.68/84.11%`, or
  `0 TP / -4 FP / +0.10 pp F1` over X85. All 30 receding-handback true
  positives and all five closing-handback true positives remain protected; all
  required checks pass. X86 is consumed Development-only and X73 retains fresh
  authority. X87 then applies the same evidence-horizon principle to an
  isolated X72 boundary-completion decision: completion proxies alone cannot
  forecast route entry beyond the inherited `0.60 s` measurement hold window.
  It removes two C35 and one C40 false-positive frames with zero TP loss. Pooled
  X87 is `1,472 TP / 130 FP / 423 FN` at `91.89/77.68/84.19%`, or
  `0 TP / -3 FP / +0.07 pp F1` over X86. The other nine cohorts are
  classification-identical and every required check passes. X87 is consumed
  Development-only and X73 retains fresh authority.
- **CARLA occlusion-source line:** C8 through C11 did not admit an evaluable X31
  source. C11 improved full disappearance coverage to `1/8`, but failed the
  frozen physical-occlusion source gate; no X31 prediction or metric was run.
- **CARLA native-dynamics line:** N3 materialized three towns and all `12/12`
  authored long-tail effects. The sole N4 replay attempt completed Town01, then
  stopped before Town04 pixels because free memory was below the frozen floor;
  Town05 never started. N4 v1 is a consumed incomplete Development attempt.

## Next admissible work

1. Run one genuinely source-disjoint confirmation of unchanged X21 before any
   promotion claim.
2. For X31, admit a new source using raster-observable occlusion authority
   before model inference; do not tune C8-C11 source thresholds or select
   favorable episodes.
3. A new N4 replay requires a new versioned authority. The consumed incomplete
   invocation cannot be resumed or reported as a three-town result.
4. Do not rerun C35, C36, C37, C38, C39, C40, or C41 as confirmation. X73 remains
   positively confirmed on C35; C36 and C37 are consumed successor-design
   evidence. X77 reduced C37's remaining false positives from 24 to 23 and crossed its
   precision floor with zero TP loss across all eight consumed cohorts. C38
   froze X77 at seed `381077` with four changed render assignments. Its
   single-use X76/X77 runner is bound to protocol SHA-256
   `B11E8C0B138D075FEF9A74295AA8E4A3F730350C42F1237A453130A6838DD31D`;
   C38 admitted and scored its sole invocation, but the X77 mechanism was not
   exercised and the full gate failed. C39 then admitted and scored its sole
   X78/X79 invocation under protocol SHA-256
   `EC62FF07F2E1FBF2A43046083D4792D6A8A6ADF1CFAB65102505BCBE965637F3`.
   Its full-arm constraints passed, but X79 exercised no lateral-only release
   and produced no incremental effect. C40 then exercised X81 on three fresh
   false-positive frames with zero TP loss, but its 83.77% full-arm precision
   missed the 85% floor. Until a later full fresh gate passes, X73 retains
   source-disjoint confirmation authority. Use C40 only for successor
   diagnosis. X82 used that consumed diagnosis to reject held-only completion
   proxy consensus and reaches 85.43% precision on C40 with zero TP loss across
   ten consumed cohorts. Any later promotion still requires a new
   preregistered source testing byte-frozen X82.
   C41 subsequently produced strong inherited full-arm metrics but exercised
   X82 zero times and exposed one non-rigid reference plus one parent mismatch.
   Use C41 only for successor diagnosis; it supplies no incremental X82
   confirmation and cannot be retried. X83 has now corrected C41's mixed
   authority reference with classification identity across eleven consumed
   cohorts. X84 then removes three branch-overloaded held-continuation false
   positives with zero TP loss across those cohorts. X85 further removes two
   C36 false positives caused by X72 reopening risk after same-frame X68
   geometric release, again with zero TP loss. X86 then removes four
   receding-handback forecasts that outlive their inherited evidence horizon,
   again with zero TP loss across three cohorts. X87 applies that same evidence
   horizon to isolated X72 completion proxies, removing three more false
   positives across C35/C40 with zero TP loss. Any promotion requires a new
   preregistered source testing frozen X87.

Local uncommitted candidates and outputs are work in progress, not route
authority. This page changes only in the scoped delivery that accepts or closes
their result.

## Stop and claim boundary

- Do not tune the consumed JRDB sequences, CARLA cohorts, route tube,
  lifecycle, association, or source gates against opened outcomes.
- `UNKNOWN` and `NOT_EVALUABLE` are not `CLEAR`, negative evidence, or safety.
- Component identity is diagnostic; wearer-global route conflict owns event
  correctness.
- Public replay and CARLA evidence are Development/mechanism evidence, not
  Android, natural-distribution, user-benefit, deployment, or safety evidence.

## Detail and evidence

- Detailed route ledger and reproduction commands: [README.md](README.md)
- X21 result:
  [X17_X21_TRACK_CARRIED_COMPONENT_ANCESTRY_2026-08-29.md](X17_X21_TRACK_CARRIED_COMPONENT_ANCESTRY_2026-08-29.md)
- X24 result:
  [DTR_CARLA_X24_PLAN_ADHERENT_DEVELOPMENT_RESULT_2026-08-30.md](carla/DTR_CARLA_X24_PLAN_ADHERENT_DEVELOPMENT_RESULT_2026-08-30.md)
- C11 source terminal:
  [DTR_CARLA_C11_X31_SOURCE_NOT_EVALUABLE_2026-08-30.md](carla/DTR_CARLA_C11_X31_SOURCE_NOT_EVALUABLE_2026-08-30.md)
- N3/N4 result:
  [DTR_CARLA_N3_N4_MULTITOWN_NATIVE_FROZEN_REPLAY_RESULT_2026-08-31.md](carla/DTR_CARLA_N3_N4_MULTITOWN_NATIVE_FROZEN_REPLAY_RESULT_2026-08-31.md)
- X84 consumed Development result:
  [DTR_CARLA_X84_CONSUMED_ELEVEN_COHORT_DEVELOPMENT_20260901.md](carla/DTR_CARLA_X84_CONSUMED_ELEVEN_COHORT_DEVELOPMENT_20260901.md)
- X85 consumed Development result:
  [DTR_CARLA_X85_CONSUMED_ELEVEN_COHORT_DEVELOPMENT_20260901.md](carla/DTR_CARLA_X85_CONSUMED_ELEVEN_COHORT_DEVELOPMENT_20260901.md)
- X86 consumed Development result:
  [DTR_CARLA_X86_CONSUMED_ELEVEN_COHORT_DEVELOPMENT_20260901.md](carla/DTR_CARLA_X86_CONSUMED_ELEVEN_COHORT_DEVELOPMENT_20260901.md)
- X87 consumed Development result:
  [DTR_CARLA_X87_CONSUMED_ELEVEN_COHORT_DEVELOPMENT_20260901.md](carla/DTR_CARLA_X87_CONSUMED_ELEVEN_COHORT_DEVELOPMENT_20260901.md)
- X64 consumed transfer Development:
  [DTR_CARLA_X64_CONSUMED_TRANSFER_DEVELOPMENT_20260831.md](carla/DTR_CARLA_X64_CONSUMED_TRANSFER_DEVELOPMENT_20260831.md)
- X64 C29-C32 fresh confirmation:
  [DTR_CARLA_C29_C32_X64_FRESH_CONFIRMATION_20260901.md](carla/DTR_CARLA_C29_C32_X64_FRESH_CONFIRMATION_20260901.md)
- X65 consumed cross-cohort Development:
  [DTR_CARLA_X65_CONSUMED_CROSS_COHORT_DEVELOPMENT_20260901.md](carla/DTR_CARLA_X65_CONSUMED_CROSS_COHORT_DEVELOPMENT_20260901.md)
- C33 terminal source result:
  [DTR_CARLA_C33_SOURCE_NOT_EVALUABLE_20260901.md](carla/DTR_CARLA_C33_SOURCE_NOT_EVALUABLE_20260901.md)
- Frozen C34 X65 protocol:
  [dtr_carla_c34_x65_fresh_source_protocol.json](carla/dtr_carla_c34_x65_fresh_source_protocol.json)
- C34 X65 fresh confirmation:
  [DTR_CARLA_C34_X65_FRESH_CONFIRMATION_20260901.md](carla/DTR_CARLA_C34_X65_FRESH_CONFIRMATION_20260901.md)
- X67 consumed cross-cohort Development:
  [DTR_CARLA_X67_CONSUMED_CROSS_COHORT_DEVELOPMENT_20260901.md](carla/DTR_CARLA_X67_CONSUMED_CROSS_COHORT_DEVELOPMENT_20260901.md)
- X68 consumed cross-cohort Development:
  [DTR_CARLA_X68_CONSUMED_CROSS_COHORT_DEVELOPMENT_20260901.md](carla/DTR_CARLA_X68_CONSUMED_CROSS_COHORT_DEVELOPMENT_20260901.md)
- X69 consumed cross-cohort Development:
  [DTR_CARLA_X69_CONSUMED_CROSS_COHORT_DEVELOPMENT_20260901.md](carla/DTR_CARLA_X69_CONSUMED_CROSS_COHORT_DEVELOPMENT_20260901.md)
- X70 consumed cross-cohort Development:
  [DTR_CARLA_X70_CONSUMED_CROSS_COHORT_DEVELOPMENT_20260901.md](carla/DTR_CARLA_X70_CONSUMED_CROSS_COHORT_DEVELOPMENT_20260901.md)
- X71 consumed cross-cohort Development:
  [DTR_CARLA_X71_CONSUMED_CROSS_COHORT_DEVELOPMENT_20260901.md](carla/DTR_CARLA_X71_CONSUMED_CROSS_COHORT_DEVELOPMENT_20260901.md)
- X72 consumed cross-cohort Development:
  [DTR_CARLA_X72_CONSUMED_CROSS_COHORT_DEVELOPMENT_20260901.md](carla/DTR_CARLA_X72_CONSUMED_CROSS_COHORT_DEVELOPMENT_20260901.md)
- X73 consumed cross-cohort Development:
  [DTR_CARLA_X73_CONSUMED_CROSS_COHORT_DEVELOPMENT_20260901.md](carla/DTR_CARLA_X73_CONSUMED_CROSS_COHORT_DEVELOPMENT_20260901.md)
- C35 X73 fresh confirmation:
  [DTR_CARLA_C35_X73_FRESH_CONFIRMATION_20260901.md](carla/DTR_CARLA_C35_X73_FRESH_CONFIRMATION_20260901.md)
- X77 consumed cross-cohort Development:
  [DTR_CARLA_X77_CONSUMED_CROSS_COHORT_DEVELOPMENT_20260901.md](carla/DTR_CARLA_X77_CONSUMED_CROSS_COHORT_DEVELOPMENT_20260901.md)
- Frozen C38 X77 protocol:
  [dtr_carla_c38_x77_fresh_confirmation_protocol.json](carla/dtr_carla_c38_x77_fresh_confirmation_protocol.json)
- C38 single-use confirmation runner:
  [run_dtr_carla_c38_x77_fresh_confirmation.py](carla/run_dtr_carla_c38_x77_fresh_confirmation.py)
- C38 X77 fresh outcome:
  [DTR_CARLA_C38_X77_FRESH_CONFIRMATION_20260901.md](carla/DTR_CARLA_C38_X77_FRESH_CONFIRMATION_20260901.md)
- X78 consumed cross-cohort Development:
  [DTR_CARLA_X78_CONSUMED_CROSS_COHORT_DEVELOPMENT_20260901.md](carla/DTR_CARLA_X78_CONSUMED_CROSS_COHORT_DEVELOPMENT_20260901.md)
- X79 consumed cross-cohort Development:
  [DTR_CARLA_X79_CONSUMED_CROSS_COHORT_DEVELOPMENT_20260901.md](carla/DTR_CARLA_X79_CONSUMED_CROSS_COHORT_DEVELOPMENT_20260901.md)
- Frozen C39 X79 protocol:
  [dtr_carla_c39_x79_fresh_confirmation_protocol.json](carla/dtr_carla_c39_x79_fresh_confirmation_protocol.json)
- C39 X79 fresh outcome:
  [DTR_CARLA_C39_X79_FRESH_CONFIRMATION_20260901.md](carla/DTR_CARLA_C39_X79_FRESH_CONFIRMATION_20260901.md)
- X80 consumed nine-cohort Development:
  [DTR_CARLA_X80_CONSUMED_NINE_COHORT_DEVELOPMENT_20260901.md](carla/DTR_CARLA_X80_CONSUMED_NINE_COHORT_DEVELOPMENT_20260901.md)
- X81 consumed nine-cohort Development:
  [DTR_CARLA_X81_CONSUMED_NINE_COHORT_DEVELOPMENT_20260901.md](carla/DTR_CARLA_X81_CONSUMED_NINE_COHORT_DEVELOPMENT_20260901.md)
- C40 X81 fresh outcome:
  [DTR_CARLA_C40_X81_FRESH_CONFIRMATION_20260901.md](carla/DTR_CARLA_C40_X81_FRESH_CONFIRMATION_20260901.md)
- X82 consumed ten-cohort Development:
  [DTR_CARLA_X82_CONSUMED_TEN_COHORT_DEVELOPMENT_20260901.md](carla/DTR_CARLA_X82_CONSUMED_TEN_COHORT_DEVELOPMENT_20260901.md)
- C41 X82 fresh outcome:
  [DTR_CARLA_C41_X82_FRESH_CONFIRMATION_20260901.md](carla/DTR_CARLA_C41_X82_FRESH_CONFIRMATION_20260901.md)
- X83 consumed eleven-cohort Development:
  [DTR_CARLA_X83_CONSUMED_ELEVEN_COHORT_DEVELOPMENT_20260901.md](carla/DTR_CARLA_X83_CONSUMED_ELEVEN_COHORT_DEVELOPMENT_20260901.md)
