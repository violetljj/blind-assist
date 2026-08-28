# L10-SC7 General Goal-Locked Visual Copilot

## Decision question

Can the same reference-bound temporal-belief abstraction used by SC6 lock,
retain, and reacquire a **non-text building-door instance** in real egocentric
video when same-category distractors, occlusion, and out-of-frame gaps exist?

This stage deliberately tests a zero-OCR evidence provider before active
viewpoint selection or arrival logic. It is not an OCR-improvement route, a
door-function classifier, or a navigation-completion test.

## Frozen G0 real-video experiment

- Stage: `L10-SC7-G0-REAL-EGOTRACKS-DOOR`.
- Source: licensed public Ego4D EgoTracks annotations and the corresponding
  videos, obtained through the official Ego4D CLI under the user's approved
  dataset agreement.
- Development split: EgoTracks `train`.
- Fresh confirmation split: EgoTracks `val`, opened only after the development
  implementation, detector, embeddings, thresholds, and selection policy are
  frozen.
- Cohort size: first 12 eligible tracks in source order per split. No item is
  selected by an arm's score or success.
- User secrets, AWS credentials, and private account identifiers are never
  written into repository artifacts.

### Frozen evaluator-only eligibility

`egotracks_sc7_source_audit.py` admits a track only when:

1. the official query set is valid;
2. `object_title` denotes a building door, doorway, entrance, or gate and does
   not denote a cabinet, appliance, drawer, or cupboard door;
3. a valid public `visual_crop` exists;
4. the long-term track contains at least 30 annotated target boxes;
5. at least one target-absent gap contains five or more missing frames; and
6. at least three consecutive target-visible annotated frames follow a gap.

The annotation hash, all denominators, title counts, eligible count, frozen
source-order cohort IDs, and selected video UIDs are emitted before any model
evaluation. `object_title`, target boxes, and target presence are evaluator
authority only after cohort admission; they must not enter arm decisions.

## Input and evidence-provider seam

The controller receives a provider-neutral `GoalRepresentation` and a list of
`CandidateEvidence` values per frame.

- Goal evidence: the official public EgoTracks visual crop.
- Candidate acquisition: one frozen category-level door proposal source,
  initially the locally pinned Grounding DINO model with the prompt `door`.
- Candidate identity: one frozen DINOv2 crop embedding against the reference
  crop, plus normalized candidate geometry, visibility quality, and temporal
  association evidence.
- Temporal motion: frame-to-frame image-plane displacement estimated without
  target annotations and passed through the same provider seam as an optional
  action/camera-motion delta.
- Forbidden evidence: OCR, rendered text, `object_title` after admission,
  EgoTracks target boxes, frame-level target presence, evaluator IDs, or a
  manually chosen frame/candidate.

Both arms receive exactly the same proposals, embeddings, and frames. Official
target boxes are joined only after each decision for scoring.

## Frozen arms

1. `RB0-STATELESS`: independently select the highest reference-similarity
   candidate when DINO cosine is at least `0.55` and its margin over the runner
   up is at least `0.04`; otherwise emit `UNKNOWN/SEARCH`.
2. `SC7-G0-BELIEF`: use the same cold-start gates, retain a bound hypothesis
   through target-blind temporal geometry, emit `LOST/SEARCH` when evidence is
   insufficient, and reacquire only when reference appearance and predicted
   geometry agree.

The reference threshold and margin are inherited from SC6 and are frozen
before materializing this cohort. The confirmation split must not change any
model, prompt, threshold, motion estimator, or selection rule.

## Metrics and decision gate

All metrics are target-box-authority scores over declared denominators:

- exact-instance precision on committed frames;
- exact-instance coverage on target-visible eligible frames;
- correct-direction coverage (`LEFT/CENTER/RIGHT`) on target-visible frames;
- wrong-instance committed frames and wrong-instance switches;
- target-absent false commits;
- gap-reacquisition success and visible frames to reacquire; and
- first reliable target-lock frame.

Development may proceed to fresh confirmation only if:

- `SC7-G0-BELIEF` exact-instance coverage is at least `RB0 + 10pp`;
- exact-instance precision is at least 95%;
- wrong-instance committed frames do not increase and are exactly zero;
- target-absent false commits are exactly zero;
- at least 80% of eligible gaps are reacquired within three target-visible
  frames after re-entry; and
- correct-direction coverage is at least `RB0 + 10pp`.

Confirmation must cross the same gate without changing code, cohort policy, or
thresholds. `UNKNOWN` frames reduce coverage but are not wrong-instance commits.

## Active-observation boundary

EgoTracks is passive recorded video. A positive G0 result establishes whether
non-text exact-instance belief is worth controlling, but cannot establish that
an action caused a better next observation.

Only after G0 confirmation may `L10-SC7-G1-DEFICIT-ACTIVE` add an action graph
whose edges correspond to real adjacent observations. Its controller must
choose a discrete action from a declared evidence deficit:

- target outside the predicted view -> `PAN/SWEEP` toward the bound trajectory;
- insufficient reference discrimination -> seek a viewpoint with greater
  instance-specific appearance separation;
- poor usable scale -> `APPROACH` only when relative geometry says the target
  direction is stable;
- sufficient evidence -> `HOLD/NAVIGATE`.

The active comparison remains `PASSIVE/HOLD` versus fixed sweep versus
deficit-specific action. Its primary outcomes are exact-instance coverage,
correct-direction coverage, frames to reliable lock, gap recovery, and
wrong-instance commits. Generic information-gain scanning is out of scope.

## Claim ceiling and stop conditions

A positive G0 result establishes only that a provider-neutral,
reference-conditioned belief improves exact non-text door-instance lock and
reacquisition on a small source-order EgoTracks cohort under a frozen generic
door proposal source. It does not establish that a door is the requested
functional entrance, that control improves the next frame, that the user can
walk safely, that arrival is known, or that the result generalizes to live
devices, users, products, or safety-critical use.

If development misses the gate, do not tune DINO, thresholds, prompt variants,
source order, or the cohort. Close this exact information route and change the
source or representation. If development passes but fresh confirmation fails,
report a development-only signal and do not promote SC7.

## Current execution state

- The signed agreement has been inspected locally; it proves agreement
  completion, not credential provisioning.
- The official Ego4D CLI is installed in the ignored project runtime and its
  entry point is runnable.
- No AWS credentials or default profile are present on this machine as of the
  current preflight, so dataset listing and download are not yet evaluable.
- ProcTHOR is retained only as a rejected source attempt:
  `NOT_EVALUABLE_RUNTIME_NO_SUPPORTED_LOCAL_BUILD`. No ProcTHOR algorithm
  outcome was observed and no negative model claim follows.

Once the approved credentials are provisioned on this machine, execution
continues with annotation-only source admission, cohort freezing, download of
only the selected video UIDs, development evaluation, and then one fresh
confirmation pass.
