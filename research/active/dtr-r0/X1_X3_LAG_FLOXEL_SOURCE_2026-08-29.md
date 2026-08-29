# DTR X1-X5 lag-compensated scene-flow source route

Date: 2026-08-29

## Decision

X0's Huang-lane `NO_MOTION_SUPPORT` is not caused by absent sensor evidence
alone.  The responsible pedestrian is present in raw rear LiDAR, but the
frozen source crops everything behind `-1.0 m`.  Extending the source field to
`-10.5 m` restores support, while the old reciprocal pairwise matcher still
produces incorrect velocity.  A lag-compensated five-scan explicit voxel flow
source passed both the positive headroom canary and the frozen X0 error slice.

This authorized one full six-sequence Development replay with the source
frozen.  It did not authorize a source-disjoint, product, real-time, Android,
user-benefit, reliability, or safety claim.

That authorized replay is now complete and did not meet its gate.  X3 is
closed without a parameter rescue.  The deterministic X4 successor described
below also failed its frozen canary in three identical cold-root executions and
is closed before full replay.

## What changed

Only the motion information source changes:

1. source coverage expands from forward `[-1.0, 12.0] m` to body-route
   `[-10.5, 12.0] m`;
2. pairwise reciprocal nearest matching is replaced by an explicit `0.5 m`
   voxel flow field optimized with five scans, distance-transform alignment,
   DBSCAN cluster consistency, and a flow-norm penalty;
3. output at wall-clock frame `t` estimates reference frame `t-2` from scans
   `t-4..t`, then transports positions through the measured two-scan delay.

The last step makes the information boundary online-causal.  The R7 motion
bounds, route-entry geometry, lifecycle, and event evaluator remain unchanged.

The representation and losses are inspired by the published
[Floxels paper](https://arxiv.org/abs/2503.04718).  This repository contains an
independent causal adaptation, not official Floxels code and not a reproduction
of the paper's benchmark numbers.

## Decisive observations

### Rear source availability

In Huang-lane frames `169..191`, native OBB `pedestrian:56` moves from about
`-9.99 m` to `-5.16 m` forward-relative position.  Raw LiDAR contains points in
that OBB in all `23/23` frames (`1,125` points total); the frozen forward crop
contains `0` of those points.

With the rear crop admitted:

| Source | Associated frames | Correct frames | Correct route-entry frames | Minimum velocity error |
|---|---:|---:|---:|---:|
| reciprocal pairwise direct | 13 | 0 | 0 | `1.542 m/s` |
| past-only four-scan causal voxel adapter | 13 | 1 | 1 | `0.102 m/s` |
| symmetric five-scan oracle | 12 | 2 | 2 | `0.126 m/s` |
| two-scan-lag compensated source | 10 | 3 | 2 | `0.095 m/s` |

The past-only X1 gate required two correct route-entry frames and therefore did
not pass.  No threshold, loss weight, crop, epoch count, or acceptance gate was
changed to rescue it.  The symmetric X1b diagnostic changed information
availability, and X1c converted that pattern to a wall-clock-causal two-scan
lag with median delay `0.150 s` (range `0.109..0.210 s`).

### Frozen X0 false-error slice

X2 evaluated one already frozen representative frame for each of X0's 35 false
units.  The source gate applies to the 34 source failures; the one
`REAL_MOVER_NONCRITICAL` unit is outside the source-error denominator.

| Cause | Units | Representative frames suppressed |
|---|---:|---:|
| `BAD_FLOW` | 16 | 13 |
| `STATIC_PSEUDO_MOTION` | 18 | 12 |
| **Combined** | **34** | **25 (73.5%)** |

The frozen gate required at least `ceil(0.70 * 34) = 24` suppressions and the
sealed X1c positive headroom.  Both checks passed.  Suppression is conservative
absence of any route-entering source cell at that representative frame; it is
not yet proof that the complete false segment disappears.

## X3 full-replay gate frozen before execution

X3 runs all `4,787` evaluable output frames across the six already opened C31
sequences.  The candidate must satisfy every check:

- CONTACT recall at least `5/6` (PDC is `4/6`);
- false alert segments at most `16` (PDC is `25`);
- Event F1 at least `0.35` (PDC is `0.2286`);
- median first-alert lead at least `2.0 s` (PDC is `2.291 s`);
- dropout recovery at least `5/18` (PDC is `5/18`);
- false segments strictly below PDC.

Materialization is GPU-required, truth-blind, frame-checkpointed, resumable,
and hash-frozen.  Candidate predictions are sealed before the scorer opens
native labels.  If the full gate fails, X3 closes and receives error
attribution; the same cohort is not used for a parameter rescue.

## X3 full-replay terminal

The sealed replay completed all `4,787` evaluable output frames.  Relative to
the frozen PDC baseline, the lag-Floxel source recovered both missed CONTACT
events, added three dropout recoveries, and moved median first alert `1.525 s`
earlier.  It also expanded false alert segments by `69` and reduced Event F1 by
`0.1154`:

| arm | CONTACT recall | false segments | Event F1 | median first lead | dropout recovery |
|---|---:|---:|---:|---:|---:|
| frozen M1-PDC | `4/6` | **25** | **22.86%** | `2.291 s` | `5/18` |
| X3 lag-Floxel | **`6/6`** | `94` | `11.32%` | **`3.816 s`** | **`8/18`** |

The recall, lead, and dropout checks passed.  The frozen `<=16` false-segment,
`>=0.35` Event-F1, and false-segments-below-PDC checks all failed.  Accept
`DTR_X3_FULL_LAG_FLOXEL_GATE_NOT_MET`: X3 demonstrates useful motion
information on this opened Development cohort, but not a selective full-route
risk source.  Do not tune its loss, crop, motion bounds, route, horizon,
lifecycle, or scorer on these outcomes, and do not use this result to open a
source-disjoint confirmation.

The evidence chain is sealed around one explicit truth-blind amendment.  The
first Huang-lane output had no complete five-scan support; it was therefore
recorded as zero dynamic cells rather than treated as a fatal source error.
The amendment changes no learned value, threshold, route decision, or opened
outcome, and its receipt is hash-bound with the six sequence ledgers,
materialization, predictions, and result.  The sealed hashes are:

- materialization: `e443ded462a604b4abd8226dec3dacd9f579a371e65b22cb8bb3d66d014aa245`;
- predictions: `b05106f8d5df8920018c4540968889e7363339d464f9947c22cdebfe9f639348`;
- result: `e1e7ddcfa9c15e1bd444410cdad5eaf716a5a203f4ddd5c4e91157676a9bc5f7`;
- amended evidence chain: `78f079015e168dc405f433960ccafa89324ffa1468b32ab0eaea4c8f7b57e5e5`.

## X4 deterministic cluster-vote terminal

X4 tested whether X3's source could be made reproducible and selective without
another optimizer run.  It kept the rear ROI, five-scan `t-4..t` information
boundary, two-scan transport, motion bounds, route geometry, and opened X1c/X2
diagnostic slices.  Each DBSCAN component instead received one rigid
displacement selected from fixed multi-scan nearest-occupancy votes.  The
implementation is CPU float64 and single-threaded, with no random seed,
autograd, optimizer, convergence threshold, or early stopping.

Three independent cold roots produced identical canonical arrays and the same
effect-signature SHA-256
`b32c50609e809f255a95c6698a2e87784f8009097861856ac11336ad459e6d61`.
They also produced the same failed effect: of 19 evaluated positive frames,
`17` had associated cells, while zero had correct motion or correct route
entry; minimum associated velocity error was `0.3922 m/s`.  The false-error slice suppressed
`31/34` source-error units (`91.18%`), but this frame-local suppression does not
replace the missing positive headroom or constitute full false-segment scoring.

| cold root | source compute p95 | median scan period | result SHA-256 |
|---|---:|---:|---|
| run 1 | `0.4502 s` | `0.06961 s` | `db575f07ad50778415e1aaf5a046770aeb7cbf3911dce75b4f93bf7be357696f` |
| run 2 | `0.4050 s` | `0.06961 s` | `8b9f483481d1ad6566ecc204c13675fdf3e4dadaaf355cc86a310bd02154b0f1` |
| run 3 | `0.3990 s` | `0.06961 s` | `1b8c2fdd34be4863c773db2096358da233472db6235311574805db0ca5388e0a` |

All three runs failed both positive checks and the one-scan-period runtime
check; only the `>=24/34` diagnostic suppression check passed.  Accept
`DTR_X4_DETERMINISTIC_CLUSTER_VOTE_REPEATABILITY_GATE_NOT_MET`.  The identical
arrays prove that this negative is deterministic rather than optimizer noise.
X4 is closed: do not tune its votes or cluster settings and do not launch a
full six-sequence X4 replay.  The repeatability receipt SHA-256 is
`5ab0acd8fe4daeab1baa575c4635db401f574bebaaf50792b955d95023b968e8`.

Both X3 and X4 remain evidence on already opened Development data.  Neither is
new source-disjoint confirmation, official Floxels reproduction, real-time
end-to-end latency, Android deployment, user-benefit, reliability, or safety
evidence.

## X3 full-replay failure attribution

The post-terminal attribution opened native truth only after verifying the
sealed X3 evidence chain.  It changed no source, threshold, route, lifecycle,
cohort, or scorer.  X3 resolved nine PDC false segments but added 73 new ones,
for a net increase of 69:

| scope | static pseudo-motion | bad magnitude | reversed direction | source failures |
|---|---:|---:|---:|---:|
| all 94 X3 false segments | 66 | 14 | 1 | `81/94` (86.17%) |
| 73 X3-only additions | 54 | 10 | 1 | `65/73` (89.04%) |

The other all-X3 primary causes are five real noncritical movers, four wrong
component bindings, three fragmentation segments, and one route-geometry miss;
11 segments also carry a comparative temporal-fragment flag.  Accept
`DTR_X3_FULL_REPLAY_FAILURE_ATTRIBUTION_COMPLETE`.  The next admissible source
must be `STATIC_AWARE_DIRECTION_CONSISTENT_SCENE_FLOW`, preserving X3's
positive motion while changing the information that creates static and
direction-wrong flow.  No threshold, seed, backbone, tracker, route, lifecycle,
fusion, or scorer sweep is authorized.  The attribution result SHA-256 is
`eb2d91065111c18c24090e80edc56ed8b76d85c993394cad8bae2d21f7b6ba26`.

## X5 overlapping-window cycle terminal

X5 asked one final bounded source question without rematerializing the full
cohort: can a local X3 cell be trusted when its transport agrees reciprocally
with the preceding overlapping five-scan window?  The 60-frame falsifier used
10,964 sealed input cells and retained 3,162.  It suppressed `32/34` frozen
source-error units (94.12%), but the positive slice fell to one correct frame
and zero correct route-entry frames.

Accept `DTR_X5_OVERLAP_CYCLE_SOURCE_FALSIFIER_GATE_NOT_MET`.  Cycle agreement
again obtains selectivity by deleting the positive motion that X3 recovered;
do not tune its consistency threshold or run it on the full cohort.  Together,
X4 and X5 close rigid-cluster motion and same-source temporal-consistency
filtering.  A successor must add a static-world anchor or independently
observable dynamic evidence rather than derive another authority from the same
unsemantic geometric flow.  The X5 result SHA-256 is
`bb0e66a3a4ad35b7ed3e1877f78df43f9852eca4a4cb6b28a582cb4b62538209`.

## Evidence

Ignored local evidence roots:

- `artifacts.local/evidence/dtr-x1/causal-floxel-source-canary/`
- `artifacts.local/evidence/dtr-x1b/symmetric-floxel-oracle/`
- `artifacts.local/evidence/dtr-x1c/lag-compensated-floxel-source/`
- `artifacts.local/evidence/dtr-x2/floxel-error-slice/`
- `artifacts.local/evidence/dtr-x3/full-lag-floxel-replay-mp/`
- `artifacts.local/evidence/dtr-x4/deterministic-cluster-vote-repeat3-20260829/`

Key sealed result hashes:

- X1c result: `6af036ce7c449899f41be5029f75872509b695478f76fe9dbf37d10dfaa2c51e`
- X2 result: `f68c559fed31a97276a40a7452adf0eec41031bd0fca1a645d657d6c8d1fd4c2`
  (X3 also records and verifies it before any full-frame optimization).
- X3 amended evidence chain:
  `78f079015e168dc405f433960ccafa89324ffa1468b32ab0eaea4c8f7b57e5e5`;
- X3 full-replay attribution:
  `eb2d91065111c18c24090e80edc56ed8b76d85c993394cad8bae2d21f7b6ba26`;
- X4 three-run repeatability receipt:
  `5ab0acd8fe4daeab1baa575c4635db401f574bebaaf50792b955d95023b968e8`.
- X5 overlapping-window cycle falsifier:
  `bb0e66a3a4ad35b7ed3e1877f78df43f9852eca4a4cb6b28a582cb4b62538209`.
