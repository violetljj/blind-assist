# DTR X1-X3 lag-compensated voxel scene-flow source

Date: 2026-08-29

## Decision

X0's Huang-lane `NO_MOTION_SUPPORT` is not caused by absent sensor evidence
alone.  The responsible pedestrian is present in raw rear LiDAR, but the
frozen source crops everything behind `-1.0 m`.  Extending the source field to
`-10.5 m` restores support, while the old reciprocal pairwise matcher still
produces incorrect velocity.  A lag-compensated five-scan explicit voxel flow
source passed both the positive headroom canary and the frozen X0 error slice.

This authorizes one full six-sequence Development replay with the source
frozen.  It does not authorize a source-disjoint, product, real-time, Android,
user-benefit, reliability, or safety claim.

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

## Evidence

Ignored local evidence roots:

- `artifacts.local/evidence/dtr-x1/causal-floxel-source-canary/`
- `artifacts.local/evidence/dtr-x1b/symmetric-floxel-oracle/`
- `artifacts.local/evidence/dtr-x1c/lag-compensated-floxel-source/`
- `artifacts.local/evidence/dtr-x2/floxel-error-slice/`
- `artifacts.local/evidence/dtr-x3/full-lag-floxel-replay/`

Key sealed result hashes before X3:

- X1c result: `6af036ce7c449899f41be5029f75872509b695478f76fe9dbf37d10dfaa2c51e`
- X2 result: `f68c559fed31a97276a40a7452adf0eec41031bd0fca1a645d657d6c8d1fd4c2`
  (X3 also records and verifies it before any full-frame optimization).
