# NF-G7 — Body contact from short visual history

EXPLORE, controlled synthetic Development. G6 found useful forward correspondence
but recovered none of four MDE-missed 2–3 m bars; ideal pose does not resolve the
aperture problem. Test whether continuous, motion-conditioned edge evidence helps
a small learned contact predictor beyond both a current-frame and ordinary video
predictor. No old G0–G6 observations or thresholds are changed.

## Frozen first experiment

Generate 24 appearance/trajectory groups, two clips per group, 16 frames per clip
at simulated 10 Hz: 768 RGB/depth captures maximum. Groups 0–11 train, 12–15
validation, 16–23 test. Each pair shares appearance, camera trajectory, speed,
height and lighting. One clip contains a head/body horizontal bar intersecting
the straight wearer corridor; the other a farther scaled bar, distant wall mark,
or lateral near miss. Split whole groups, never adjacent frames. The frozen
Willow map is not saved or visually revised. Rendering uses settled posed samples
of a continuous prescribed trajectory, not measured wall-clock sensor cadence.

Eight past/current observations yield nine windows per clip. Inputs are RGB,
calibration, speed and **ideal metric ego poses**, in metres/degrees; no native
depth, obstacle coordinates, masks, group IDs or future frames are model features.
These privileged poses do not represent an IMU estimator. Camera bob/yaw are
separate from the constant straight body motion. Labels are analytic first
contact of the specified BODY/HEAD boxes with task object boxes while continuing
that body velocity, at horizons 1/2/3 seconds. This is not human-mesh contact or
closed-loop avoidance. Native depth independently checks the rendered boxes.
No-risk windows and unobserved contact beyond a horizon are explicit zeros for
that horizon; only task-object contact is the controlled endpoint.

Compare frozen Hypersim Small518 with supported raw/ground geometry adapted to
the same body corridor and contact horizon; a current-frame learner; ordinary
eight-frame learner; and a geometry-guided temporal learner. Three learners use
the same small backbone/head capacity and motion metadata. The structural arm
adds rotation-compensated plane-sweep/edge-normal evidence, with its extra cost
reported. Use seeds 17/29/43, at most 100 epochs each, one optimizer/configuration,
validation checkpoint selection, no test-driven tuning. Prediction-time repeated
current RGB/pose histories test temporal reliance for both video learners; the
trained current-frame arm distinguishes this intervention from a fair baseline.

Threshold selection uses validation negatives only, targeting zero false-positive
windows and reporting the resulting test burden. Report each seed plus ensemble,
BODY/HEAD and each horizon, TP/FP/FN, contact-episode recovery, simulated warning
lead, Brier score, and actual GPU preprocessing/inference cost. Keep the fixed
binary geometry operating point distinct; unavailable geometry is UNKNOWN, not
an observed negative. Equal-parameter learners are not equal-total-compute arms.

Retain the structural mechanism only if it recovers additional test contact
episodes over both ordinary video and current-frame models without increasing
false-positive episodes, with loss of useful evidence under repeated history;
report mixed/seed-dependent findings without promotion. No gain is scoped to
this implementation/source/budget, not an RGB impossibility. Stop after this
dataset and fixed training comparison, except task-owned mechanical corrections.
If the source or comparison cannot support a claim, record NOT_EVALUABLE.

The root run stores the pre-capture brief, complete input/code identities, native
verification, checkpoints, prediction caches, scores and process-release receipts
under `artifacts.local/nearfield/contact-retina-20260907-v1/`. Predictions are
cached before opening test labels. No App default changes, phone-speed, real-IMU,
natural generalization, user-benefit or avoidance-effect claim.

## Completed result

All 768 frames passed native box visibility/floor and split-boundary verification.
Capture took 475.61 s; the unchanged map and released UE process tree are verified.
The 48 clips provide 432 windows: 216 train, 72 validation, 144 test. All paired
groups stay together. Nine learners completed 100 epochs each in 146.47 s total,
including feature preparation and workload probes, on RTX 5060 Laptop CUDA.
Predictions were cached before scoring test labels; no outcome-driven retraining,
threshold search or extra capture followed. The root pre-capture brief is retained
unchanged; this result section is appended only after scoring.

Below are three-seed **ensembles**, except the fixed MDE baseline. A recovered
episode is a clip/body-part with a correct 3-second warning at least once. The
8 positive clips and 8 negative clips are a small held-out group split within
one synthetic map, not independent natural scenes. An erroneous short-horizon
claim counts as a false alert even inside a genuinely positive episode.

| Arm | Contact episodes recovered | Negative clips falsely alerted | Any false-alert clips /16 | Mean first correct simulated lead |
| --- | --- | --- | --- | --- |
| Frozen MDE + body-corridor adapter | 1/8 | 0/8 | 1 | 1.038 s |
| Current-frame learner | 7/8 | 1/8 | 1 | 2.326 s |
| Ordinary video learner | 0/8 | 0/8 | 0 | unavailable |
| Temporal structure learner | 7/8 | 0/8 | 3 | 2.140 s |
| Temporal structure, repeated history | 0/8 | 0/8 | 0 | unavailable |

The structural seeds recover 6/6/7 episodes with 4/4/2 falsely alerted clips;
all have zero negative-clip false alerts. Their repeated-history interventions
recover 0/0/0. The single-frame seeds recover 0/0/7, exposing substantial training
or operating-point instability in this tiny source. Ordinary video recovers
0/0/0 at its validation-selected zero-FP operating points; silence is not success
or proof that video contains no useful information.

The structural ensemble's **nine false-positive cells are all BODY@2s** in
three true-contact clips. BODY@3s has TP20/FP0/FN14; HEAD@3s TP12/FP0/FN21.
Thus 7/8 episode recovery does not imply high per-window coverage. BODY@2s is
TP5/FP9; HEAD@2s TP10/FP0/FN9. Test BODY@1s has no positive opportunities;
HEAD@1s has three and all are missed. The current data cannot establish broad
urgent-contact competence. Baseline 33/864 UNKNOWN cells remain explicit; its
known-cell Brier score is not a full-coverage score comparable to the learners.

**Decision: useful candidate signal, no retained replacement/default branch.**
None of the seeds or ensemble meets the predeclared combination of extra recovery
over both learners, no added falsely alerted clips, and repeated-history loss.
The result does not negate temporal structure or direct contact prediction.
It identifies short-horizon calibration and unstable small-data baselines as
the next decision points. The repeated intervention removes both image and pose
history, so it establishes dependence on the joint history intervention, not an
isolated causal contribution from visual motion or one structural channel.

A justified successor should separate contact existence from urgency calibration,
provide adequate near-contact opportunities across BODY/HEAD, and isolate visual
history while preserving the observed motion metadata. These are future changes,
not posthoc tuning or promotion of this consumed test set. The existing dense /
fusion baseline and G0–G6 findings remain retained in their original scopes.

All learner arms have 115,686 trainable parameters. Full fixed structural feature
preparation P50 is 12.63 ms/window, including GPU transfers and feature-cache copy;
RGB file decode/hash is separate. MDE inference P50 is 81.83 ms plus 8.11 ms
geometry. Learner model-only **batch-throughput** timings (0.035–0.066 ms/example)
exclude preprocessing and are not single-window latency. No phone speed claim.

Durable output: `artifacts.local/nearfield/contact-retina-20260907-v1/`, including
`capture/verification.json`, `baseline/predictions.json`, `learned/receipt.json`,
all nine checkpoints, `evaluation/result.json`, and the detailed `evaluation/report.md`.
The registered input manifest binds source, predictions, protocol and code hashes.

Reproduction uses a fresh artifact root, the configured research GPU Python,
`contact_retina_spec.py`, `launch_contact_retina.py`, `verify_contact_retina.py`,
`predict_contact_depth.py`, `train_contact_retina.py`, then `evaluate_contact_retina.py`.
The source/model/evaluation commands expose their required paths with `--help`.
Source tests (2), GPU learner tests (7), body-heading geometry test (1), and
cached-evaluator tests (7) pass. Completed capture and Python jobs released their
owned processes; all evidence and checkpoints remain available.
