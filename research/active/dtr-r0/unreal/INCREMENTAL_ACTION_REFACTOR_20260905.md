# Incremental X73 and action-conditioned motion queries

This change separates the cost of maintaining obstacle state from the choice of
the wearer's next motion. The UE fixed replay and live worker now use incremental
X73 by default; the original batch predictor remains the differential oracle.
The shared X24/X25 fitting API now accepts an explicit source window, while its
original defaults preserve the retained X73 behavior. The UE motion default is
`DEPTH_ONLY`, the measured reference; action-conditioned planning stays optional.

## Incremental execution: measured equivalent

One persistent executor per episode retains each layer's trackers and temporal
state. After the first observation's WARMUP, each arriving frame advances all
25 stateful layers once. The first sample is deferred until the observed second
sample supplies the original initial time interval. Per-layer frame history is
bounded to two slots; discarded history cannot be sliced or iterated as a prefix.
Current X73, metric and rigid-footprint outputs are detached from retained state.

The adapter compiles the explicitly selected frame loops from the installed
research functions. This avoids maintaining a second handwritten copy of the
many credential and lineage rules. A contract fixes the normalized complete
bodies of all 33 lifted or mapped functions, including their pre/post-loop code.
Changing one requires reviewing its adapter semantics and differential check;
an unreviewed change fails before execution. Local dependency hashes accompany
validation and closed-loop receipts. The inherited temporary module bindings
require serial execution within one interpreter; the live HTTP server is serial.

The consumed input is the existing V4 fixed-sensory dataset: 16 trajectories,
733 RGB-D observations, poses, calibration and preceding issued plans. One
hash-sealed CUDA detector ledger fed both engines. At every prefix the original
batch predictor ran independently and was compared without a numerical tolerance.

| Check | Result |
| --- | --- |
| Historical compact outputs | 733/733 identical on the common fields |
| Full X73, metric, rigid and compact current-frame dictionaries | 717/717 non-WARMUP prefixes identical |
| Each stateful layer's processed count | Exactly the episode's frame count |
| Retained per-layer frame slots | At most 2 |
| Batch predictor | 370.2786 s |
| Incremental predictor, including construction | 24.0193 s |
| Predictor throughput ratio | 15.4159x |

This isolated measurement preceded the function-body admission guard; the final
guard version separately reproduced all 41 historical rows and 40 full-state
prefixes of one representative episode. The guard changes admission, not the
admitted frame program. The full differential run's 493.65 s includes detector
harvesting and additional rigid/metric oracle runs; it is not a replay-workflow
latency number. These results establish implementation equivalence on consumed
synthetic Development, not new detection accuracy or protected confirmation.

Receipts are under `artifacts.local/unreal/`:

- `incremental-x73-validation-20260905-a/validation.json`
- `incremental-x73-validation-20260905-a/frame-equivalence.jsonl`
- `incremental-x73-validation-20260905-a/candidate-receipt.json`
- `incremental-x73-validation-20260905-guard-a/validation.json`

The final default fixed-replay CLI also completed the whole 733-frame dataset,
including fresh CUDA detection, verification and incremental initialization:
**51.3426 s**, with 18.5347 s in detection and 14.2463 s in frame updates.
All shared fields of all 733 rows match the historical replay. That historical
workflow took 448.9468 s, an observed same-workload ratio of 8.7441x; it was not an
interleaved benchmark and cache/environment timing differences remain possible.
`fixed-perception-v4-incremental-20260905-a/integration-verification.json` records
the input/model join, source hashes at that run and comparison. The process exited.

After consolidating the fitting API, the final source reproduced all 733 compact
rows and all 717 sealed full-state hashes again. Its action-footprint output also
matched the pre-consolidation helper snapshotted by comparison B on all 733
observations (259 supported frames). This check used the existing detector ledger
and no new UE launch or inference. The final source and input identities are in
`incremental-x73-final-api-verification-20260905.json`.

## Live integration and motion candidates

The worker advances obstacle state once per newly observed frame and caches
detector results with source-image and model hashes. Restart reconstructs the
causal state once from those verified observations. Atomically committed policy
checkpoints are authoritative; a journal entry written just before a crash may
not skip an uncommitted policy transition. Repeated committed requests reuse the
response. The incremental path does not retain a complete detector-history list.

`CANDIDATE_DEPTH` and `CANDIDATE_DTR` expose the same observed-depth-admitted
motion set: the existing depth command, waiting and bounded lateral targets.
Both compute the same current rigid footprints and candidate scores. Only the
second mode changes selection when an alternative avoids or delays a supported
intersection of the nominal motion. Existing immediate depth braking, invalid
depth stops and arrival retain precedence. Legacy JOINT/DTR_ONLY/DEPTH_ONLY
behavior is unchanged.

Each query uses the current sensor-derived X25 footprint and relative velocity.
Continuous line sweeps against a convex footprint dilated by a circular body
avoid time-grid tunnelling and square-corner expansion. The first rollout segment
is the command actually proposed; later segments apply the stated lane feedback.
The later rollout is approximate: the real controller may return toward the
center after passing an obstacle and re-evaluates with new observations every
0.2 s. It is not a guaranteed three-second executed plan.
The horizon is 3 s and the body radius 0.30 m, matching the existing depth
corridor half-width. This is a new action-footprint hypothesis, not the retained
X73 0.65 m alert tube. Unknown, stale or malformed support remains unknown.

These are unissued motion hypotheses. Neither an old plan credential nor a raw
X73 route-risk bit authorizes a new path. Footprints do not establish complete
visibility or safety; detector misses and erroneous velocity estimates remain
possible. The comparison therefore records success, contact, timeout, delay and
actual selected motion, including cases where the new mechanism adds no value.

## Source-cadence defect and its isolated correction

Comparison A completed all 32 real branches with frozen inputs and runtime
sources. Both arms reached 8/8 success and identical per-case arrival times, but
**neither arm had any usable rigid footprints or changed a motion**. Both logged
697 sensor frames. This is a degenerate comparison, not evidence of action-risk
benefit. Its untouched report, snapshots and diagnosis remain at
`artifacts.local/unreal/candidate-v4-comparison-20260905-a/`.

The cause is an inherited source contract mismatch: X24 requires four real
samples in a 0.50 s fitting window; UE samples every 0.20 s, so that window can
contain only three. X25 does not emit a track without its fitted position and
velocity. First-20-frame diagnosis found 27 detections and 11 valid rigid
measurements across ten frames, yet no emitted state. One frame had 5,754 valid
foreground points and six historical measurements; the fit window still kept
only three. Changing the geometry gate or the detector score would not repair
that contradiction.

The new `ActionFootprints` component changes the action branch's sampling
contract explicitly. After observing the first two source timestamps it freezes
the fit window to `max(0.50, (minimum_samples - 1) * source_interval)` seconds:
0.60 s for this source. It retains four real samples, the original one-second
history limit, fitting math, association and HOLD rules. Missed detections do not
expand the window or supply samples. A source whose required span exceeds the
history limit is unsupported. The shared `robust_motion(..., window_s=...)` and
`RigidFootprintTracker.update(..., fit_window_s=...)` interfaces carry this
contract; callers that omit the argument keep the original 0.50 s window. This
removes the duplicate pairwise fitter used during comparison B. No sample-count,
span, association or HOLD threshold was relaxed. The 20 Hz component check
matches original emitted tracks even through dropout, and the final full-input
hash check above confirms retained X73 behavior.

On the same first 20 consumed observations and sealed detector rows, the new
contract yields seven supported frames instead of zero, with the same ten
measurement-bearing frames. This checks state availability only; it is not an
avoidance result. The receipt is
`incremental-x73-validation-20260905-a/action-cadence-canary.json`.

Candidate modes default to this separately identified action state. The
`--action-footprint-state frozen` option preserves the original no-support
configuration. Both modes in corrected comparison B use the identical corrected
state, candidate actions, scene and budgets. Comparison A is retained as a
negative control and is not overwritten or promoted by the correction.

The launcher also tracks task-owned process descendants with PID/creation
identities, allows bounded natural exit, then verifies their exit and worker
port closure. Comparison A's immediate port check was premature; later process
and listener-table checks showed it closed without intervention. The new helper
has real wrapper/child/socket tests and an unrelated-sibling preservation test;
it never adopts or kills a process merely because it owns a port.

## Corrected closed-loop result and default decision

Comparison B completed 32 new branches and 1,394 actual sensor frames. Both arms
passed all eight open-loop contrasts. Each assisted arm supplied 369 frames;
119 baseline frames and 120 challenger frames had usable action footprints.
The challenger changed three commands: one sidestep and one wait in occluded
crossing, and one sidestep in narrow passing, each lasting 0.2 s.

| Regression condition | CANDIDATE_DEPTH arrival | CANDIDATE_DTR arrival | Contact in either assisted arm |
| --- | --- | --- | --- |
| Occluded crossing / collision | 10.8 s | 10.8 s | No |
| Occluded crossing / near miss | 8.0 s | 8.0 s | No |
| Sudden stop / collision | 9.6 s | 9.6 s | No |
| Sudden stop / near miss | 8.8 s | 8.8 s | No |
| Narrow passing / collision | 9.4 s | 9.4 s | No |
| Narrow passing / near miss | 8.0 s | 8.0 s | No |
| Low obstacle / collision | 9.6 s | 9.6 s | No |
| Low obstacle / near miss | 8.0 s | 8.0 s | No |
| Successful arrival | **8/8** | **8/8** | |

The decision is **NO_INCREMENTAL_SUCCESS_GAIN**: a functioning motion mechanism
is demonstrated, with no added success or arrival-time benefit on this panel.
The two arms rerendered observations; this is not a claim of identical pixels.
The earlier [three-mode ablation](ALGORITHM_LAB_20260905.md) measured DEPTH_ONLY
8/8, JOINT 7/8 and DTR_ONLY 4/8. Candidate scoring leaves the depth command
unchanged in CANDIDATE_DEPTH. Together these results support making DEPTH_ONLY
the UE launcher/server default and retaining CANDIDATE_DTR as an explicit
challenger. This changes the lab entry point, not the Android application.
All existing modes remain explicitly selectable, including `--controller-mode JOINT`.

The complete report, 61 source snapshots and five frozen input identities are
under `artifacts.local/unreal/candidate-v4-comparison-20260905-b/`. Its
`identity-verification.json` confirms sources and inputs were unchanged through
the run; both `process-release.json` receipts confirm tracked process trees and
worker ports were released. Later API consolidation and CLI-default edits are
identified separately by the final verification and delivery hashes, not
backdated into either frozen comparison.

## Reproduction and boundaries

From the checkout, with its existing CUDA-enabled research Python:

```powershell
python -B research/active/dtr-r0/unreal/ue_fixed_replay.py replay --dataset artifacts.local/unreal/fixed-sensory-v4-20260905-a --output artifacts.local/unreal/<new-replay>
python -B tools/run_street_candidate_comparison.py --engine <UE-engine-root> --scenario-manifest artifacts.local/unreal/scenario-bank-v2-20260905.json --output artifacts.local/unreal/<new-comparison>
```

The first command replays recorded motion only. The second executes the eight
previously consumed regression cases with two modes and fresh open-loop/assisted
branches, freezing code, scene, model and render inputs. Both arms use identical
compute branches; their comparison is about motion selection. No held-out recipe,
CARLA avoidance R1 final source, fitting or final scoring is included.

Focused verification covers analytic swept contacts, relative motion, nominal
depth equivalence, immediate stop precedence, checkpoint crash boundaries,
source admission, input order, WARMUP, bounded histories, plan expiry, parent
lineage, dropout, contradiction release and caller-output isolation. The final
focused DTR suite passed **51/51** tests, including cadence, process ownership
and socket cleanup. These were implementation checks, not extra scene trials.
The [structured disposition](ue_incremental_action_disposition_v1.json) records
the retained executor, source-cadence component, challenger and negative control.

Normal experiment registration was attempted at delivery. The tool rejected the
existing unrelated row `experiments/index.jsonl:252` because its input fingerprint
does not match its references. The registry, central terminals, inheritance and
generated index were byte-identical before and after that attempt. The exact
command and error are retained in
`artifacts.local/analysis/refactor-20260905/registration-result.json`. Local
dispositions are complete; central registration remains an explicit metadata gap.
No manual append or unrelated registry repair was made.
