# DTR X11-X15 RGB-authorized motion continuation

Date: 2026-08-29

## Decision

X11-X15 tested whether independent RGB motion can improve X7 source authority
without changing X7 velocity, the frozen R7 route geometry, X3 lifecycle, or
the scorer.  Three frame-local alternatives closed on the already opened X6
60-frame diagnostic roster:

- X11 calibrated raw-multiview static veto did not beat the stitched X8 veto
  and retained only one correct route-entry frame;
- X12 CIWT agreement obtained perfect source-error suppression by rejecting all
  positive motion and therefore supplied no usable positive authority;
- X13 stitched RGB dynamic birth recovered two correct-motion frames but zero
  correct route-entry frames.

Accept their terminals without a threshold, camera-fusion, tracker, or
association rescue:

- `DTR_X11_RAW_MULTIVIEW_STATIC_VETO_FALSIFIER_GATE_NOT_MET`;
- `DTR_X12_CIWT_TRACK_MOTION_AGREEMENT_FALSIFIER_GATE_NOT_MET`;
- `DTR_X13_STITCHED_DYNAMIC_BIRTH_AUTHORITY_FALSIFIER_GATE_NOT_MET`.

X14 changed the question from frame-local birth to causal persistence.  Only a
cell already authorized by independent stitched RGB dynamic agreement may
continue, at its unchanged X7 velocity, for the frozen `0.50 s` R1 clear-grace
duration.  Continuation cannot originate authority.  That canary passed all
four frozen checks and authorized exactly one full replay.

The resulting X15 six-sequence replay recovered **`5/6` CONTACT events**,
produced **18 false segments**, reached **34.48% Event F1**, retained
**`3.06 s` median first-alert lead**, and recovered only **`2/18` dropout
trials**.  Its full gate did not pass: false segments exceed 16, Event F1 is
below 35%, and dropout recovery is below `5/18`, although recall, lead, and
false-segments-below-PDC passed.

Accept `DTR_X15_FULL_RGB_AUTHORIZED_CONTINUATION_GATE_NOT_MET`.  X15 is useful
Development evidence that independent RGB birth plus bounded continuation can
approach the frozen selectivity target while preserving useful lead, but it is
not a passing full-route authority and does not authorize parameter rescue or
confirmation.

## Frozen mechanism

The route keeps sealed X7 cells as the only geometric and velocity candidates.
At consecutive frames, sparse forward/backward LK tracks compare observed
stitched-RGB displacement against the ego-pose-conditioned moving hypothesis.
The fixed C22/X13 representation uses five height anchors, `1.5 px`
forward/backward and moving-agreement scales, `0.5 px` motion-strength scale,
the `0.75` cell quantile, and `0.5` decision confidence.

The source semantics are deliberately asymmetric:

1. independent RGB dynamic agreement may create a birth from an X7 cell;
2. a previously authorized cell may be transported at unchanged X7 velocity
   for at most `0.50 s`;
3. continuation never creates a new birth and does not refresh itself without
   new RGB authorization;
4. missing images, invalid projection, or failed tracking create no new
   authority, while an already valid bounded continuation may finish its
   frozen grace interval.

X7 velocity and motion bounds, R7 route-entry geometry, X3 downstream event
lifecycle, and the scorer remain unchanged.  This is a motion-source authority
change, not a route, lifecycle, or evaluator change.

## X11 raw-multiview static veto terminal

X11 asked whether native distorted perspective images improve X8's
static-versus-moving veto.  It used only cameras `(6, 8, 0, 2, 4)`, for which
the repository has validated `cam2ego` calibration.  Missing images, invalid
projections, and failed raw LK retained X7 candidates.

| check | frozen requirement | observed |
|---|---:|---:|
| correct positive frames | `>=2` | **3** |
| correct route-entry frames | `>=2` | **1** |
| source-error suppression | strictly above X8 `27/34` | **`26/34` (76.47%)** |
| source compute p95 | `<=0.06961 s` | **`0.01891 s`** |

Raw multiview was fast and retained some correct motion, but it neither met the
route-positive check nor improved the stitched veto's suppression.  It is
closed without a camera-selection, fusion, distortion, or confidence sweep.

Local SHA-256:

- script: `c17d6072937b2d2c00fe5fff401b85583928361db5a4ab1ebf7653353bee2490`;
- materialization: `66761a9aa83f548404993b6ff79d068e75dc7a96bb4b84af3a68527cd5b946e5`;
- result: `7a4cd7163a7647881d564f65e8bf1cc231ec6279a78b9ae2d36610d9f0aa638b`.

## X12 CIWT track-motion agreement terminal

X12 tested CIWT 2D track displacement as fail-closed positive dynamic
authority.  Inference excluded CIWT 3D placeholder fields, native OBB,
known-height tracks, and evaluator truth.  Missing, ambiguous, malformed, or
no-previous-frame track evidence rejected the X7 candidate.

| check | frozen requirement | observed |
|---|---:|---:|
| correct positive frames | `>=2` | **0** |
| correct route-entry frames | `>=2` | **0** |
| source-error suppression | `>=24/34` | **`34/34` (100%)** |

The 100% suppression is deletion, not selectivity: no positive frame retained
associated motion.  Bundled CIWT generation and training provenance is also
incomplete, so the experiment cannot promote CIWT authority even apart from
the failed positive gate.  X12 is closed without tracker, association, or
height-model tuning.

Local SHA-256:

- script: `cedda053cddbb5d6ba86b9eaf67f0b8327278e1013d5afbd51798fe69d8e10e4`;
- materialization: `2d5aec7868f1144d2d1b176d6c11aff04c6864928589ee4a6ea44614c28b226e`;
- result: `a0dc62d711e6eaa5f18da0aee92888835e368f960c9da83003d1127de66e97f1`.

## X13 stitched dynamic-birth authority terminal

X13 inverted the visual question: instead of vetoing cells that look static,
could stitched RGB positively authorize X7 dynamic births?  It remained
frame-local; missing images, invalid projections, and failed tracks rejected
new authority.

| check | frozen requirement | observed |
|---|---:|---:|
| correct positive frames | `>=2` | **2** |
| correct route-entry frames | `>=2` | **0** |
| source-error suppression | `>=24/34` | **`33/34` (97.06%)** |
| source compute p95 | `<=0.06961 s` | **`0.03136 s`** |

Frame-local RGB birth had real correct-motion evidence but did not survive long
enough to produce a correct route-entry frame.  X13 therefore closes as a
standalone authority.  Its failure specifically left bounded continuation
untested; it did not authorize retuning dynamic confidence or route geometry.

Local SHA-256:

- script: `e5e69dad4fec307488b0580c887292abb990eb26e81ae3ec8f0ce04f67547ce1`;
- materialization: `82135c3e8055ec5b2c8983f85630cb4a9b5a59b277e2448e5f86e446d87da27a`;
- result: `349ece2451ffb5039cd158c205c472827bbb600f2eda9f9de5b347ca2be1d756`.

## X14 bounded-continuation canary

X14 allowed only previously RGB-authorized motion to continue for the frozen
`0.50 s` grace.  Transport used unchanged X7 velocity; continuation could not
originate or recursively refresh authority.

| check | frozen requirement | observed |
|---|---:|---:|
| correct positive frames | `>=2` | **13** |
| correct route-entry frames | `>=2` | **4** |
| source-error suppression | `>=24/34` | **`30/34` (88.24%)** |
| source compute p95 | `<=0.06961 s` | **`0.02814 s`** |

All four checks passed.  Accept
`DTR_X14_RGB_AUTHORIZED_MOTION_CONTINUATION_FALSIFIER_GATE_MET`.  This is
mechanism headroom on an opened 60-frame diagnostic roster and authorized one
frozen full replay only; it is not event-level success or confirmation.

Local SHA-256:

- script: `1bd600373548d9cbd29d8d6daa6736049081b7580065f0d3588095cc401b4062`;
- materialization: `2d6fb8ccf2410fd5f4fa9aa10eb7a9e8845f864f44a0fc1b605696a8d4908269`;
- result: `9a96bf686f6134f7f092dbba711927c2cb868d7e71fae844004426000da0505f`.

## X15 full-replay terminal

X15 materialized all 4,811 timeline frames across the six already opened C31
Development sequences.  All requested stitched images matched, 4,803 frames
had evaluable consecutive visual context, and eight frames created no new
authority.  The replay inspected 301,483 sealed X7 input cells and recorded
36,400 RGB-authorized cell instances plus 283,148 bounded-continuation cell
instances.  These diagnostics are not a disjoint partition: an authorization
may support transported instances over several following frames.

All six ledgers and truth-blind predictions were sealed before the scorer
opened native OBB truth.

| arm | CONTACT recall | false segments | Event F1 | median first lead | dropout recovery |
|---|---:|---:|---:|---:|---:|
| frozen M1-PDC | `4/6` | 25 | 22.86% | `2.291 s` | `5/18` |
| X15 RGB-authorized continuation | **`5/6`** | **18** | **34.48%** | **`3.061 s`** | **`2/18`** |

Frozen gate checks:

- CONTACT recall at least `5/6`: **pass**;
- false segments at most 16: **fail** (`18`);
- Event F1 at least 35%: **fail** (`34.48%`);
- median first-alert lead at least `2.0 s`: **pass** (`3.061 s`);
- dropout recovery at least `5/18`: **fail** (`2/18`);
- false segments strictly below PDC's 25: **pass**.

X15 improves PDC recall by one event, removes seven false segments, raises F1
by 11.62 percentage points, and adds `0.769 s` median lead.  The aggregate is
close to the false/F1 thresholds but fails them exactly as frozen, while the
dropout regression is material.  Do not round `34.48%` to a pass, substitute
PDC dropout behavior, or tune continuation duration, confidence, route,
lifecycle, or scoring on this opened outcome.

Local SHA-256:

- script: `1bf3493b3d740375b55322d4c4504bf4e64f39a163b0e5613733e98acf785a02`;
- freeze: `bdf14094e8a8f5880d81c9d6cfe494ce008e5e880418a6afaee1d0c4742069d8`;
- materialization: `ba52559809cc28c459fdbd95968351c9c0057e770a8b4f8a1759856c71b01990`;
- sealed predictions: `ef2a5682ba466676f9a544c3c9fab7a228d5fec9794c140099e241b6944242cb`;
- result: `93ca95b88ff0d849a715d8ee0dfb3f5de4f2309b8d543f7e54ef53730f132b22`.

## Evidence boundary

X11-X14 are post-outcome mechanism diagnostics on the already opened X6
60-frame roster.  X14's passing canary establishes bounded mechanism headroom,
not confirmation or full-event success.  X15 is a full six-sequence replay on
the already opened C31/X0 Development cohort, not a new source-disjoint cohort.

The source uses public robot RGB, LiDAR-derived X7 candidates, and measured ego
pose.  Native OBB truth is evaluator-only after prediction sealing.  X12's
incomplete CIWT provenance prevents authority promotion.  Canary runtime covers
projection, LK or track agreement, confidence, and bounded transport as stated
in each result; bag scanning, image matching, and image decoding are excluded.
X15 does not establish end-to-end latency.

These results do not establish Android execution, real-device behavior,
user benefit, reliability, or safety.  X11, X12, and X13 remain closed.  X14
authorizes only the completed X15 replay, and X15's failed full gate does not
authorize another replay, parameter sweep, or confirmation claim.

## X16 frozen-composition terminal

One post-outcome structural composition asked whether the already sealed X10
sequence-held-out authority could replace X7 as X15's birth base. X13 RGB
birth, X14 `0.50 s` continuation, route, lifecycle, and scorer remained
unchanged. Every X10 held-out ledger was trained on the other five sequences
and sealed without its own labels.

X16 processed all 4,811 frames, 108,403 X10 input cells, 12,353 doubly
authorized births, and 96,191 continued cell instances. It produced `4/6`
CONTACT recall, 15 false segments, 32.00% Event F1, `3.630 s` median lead, and
`2/18` dropout recovery. The stricter composition therefore crossed the
false-segment bound but lost one X15 CONTACT and did not solve dropout. Accept
`DTR_X16_CROSSFIT_RGB_AUTHORIZED_CONTINUATION_GATE_NOT_MET` and close the
composition without changing the X10 threshold, visual confidence, or
continuation duration.

Local SHA-256:

- script: `def36ad6effb9223b61caf7182023908dbcbcd1efe13a8c47f9317c7c83c962e`;
- freeze: `cb6b57a93253bc1cf83ebc02d540b5da5c9379eaabc6b8f129f97ef392f0a3d0`;
- materialization: `00c4e1ca0b32400edacfb6fd079617b27483836a91e0f9c5a9d748c791bb4636`;
- sealed predictions: `bd11c29521db16e164bf7c539f9da575e8e1b194b3002a15377d019061d1acdb`;
- result: `56045777e0dc0d78596987a2a2e82e81e496e206177adb869bd21c62c8d7b4a3`.

X16 is still Development evidence on the same opened cohort, not confirmation.
It establishes neither deployable learned authority nor real-device or safety
performance. X15 remains the stronger recall/F1 reference.

## Residual failure layer

A read-only replay of X15 birth, transport, raw-risk, and lifecycle frames
shows that the 18 false segments are not 18 independent risky RGB births. All
18 include cells authorized outside the route and later transported into it;
14 segments have no same-frame risky birth. Memorial contributes 11 segments:
seven are transport-only and four mix transport with a small number of risky
births. Across its 684 active false frames, 537 have raw risk and 147 are only
lifecycle HOLD tails.

The missed Huang-2 `contact:001` dies before route scoring: X7 has a raw route
entry at frame 87, while X15 has no raw or active alert anywhere in truth frames
36--94. RGB birth authorization/materialization removes it; lifecycle does not.
The `2/18` dropout score likewise measures whether candidate raw-risk frames
intersect each synthetically dropped window, not whether HOLD spans the gap.

This closes lifecycle shortening, range merging, same-X7 reciprocal closure,
and another learned deletion filter as successors. The next admissible
representation is a sensor-disjoint, ego-compensated RGB movable-instance
mask/track that must persist across two frames before it can originate an X15
birth. X15's transport, route, lifecycle, and scorer should remain frozen for
that one full falsifier.
