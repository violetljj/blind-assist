# C28 visibility-conditioned point memory

Date: 2026-08-28

## Decision

C27 proves that positive-only point memory is not the missing representation.
After correcting its baseline-lifecycle union, it retains `12/12` CONTACT
recall and improves median lead over M1-PDC (`2.967 s` versus `1.624 s`), but
still produces 26 false segments versus 21 and recovers only `27/36` induced
dropout trials versus R7's `30/36`.  The ledger contains many propagated
positives but no causal evidence saying why a point is absent now.

C28 therefore changes the information source, not a threshold.  Each predicted
point support receives a current LiDAR ray state:

```text
HIT          observed occupied now
KNOWN_FREE   a current ray traverses the predicted 3-D cell before its return
OCCLUDED     a nearer return blocks the predicted cell
UNSENSED     no admissible current ray observes the cell
```

The state transition is fixed: `HIT -> present`, `KNOWN_FREE -> departed and
clear`, `OCCLUDED -> propagate with age`, and `UNSENSED -> UNKNOWN`.  Only
`OCCLUDED` may justify persistence.  `UNSENSED` is not a negative, and no
component velocity may be broadcast.

## External mechanisms retained

- [DUFOMap](https://arxiv.org/html/2403.01449) and its
  [official implementation](https://github.com/KTH-RPL/dufomap) establish the
  geometric inversion used here: ray-traversed void space is a causal
  known-empty observation, and a later hit inside prior void space is dynamic
  evidence.  C28 additionally uses the reverse contradiction to clear a
  departed positive.
- [Categorized Grid](https://arxiv.org/html/2407.02192) motivates retaining the
  cause of unknown space instead of collapsing out-of-FoV, unsensed, and
  occluded cells into one value.
- [FreeDOM](https://arxiv.org/html/2504.11073) and its
  [official implementation](https://github.com/LC-Robotics/FreeDOM) motivate
  incremental free-space clearing of historical dynamic remnants.
- [UnO](https://openaccess.thecvf.com/content/CVPR2024/papers/Agro_UnO_Unsupervised_Occupancy_Fields_for_Perception_and_Forecasting_CVPR_2024_paper.pdf)
  models future LiDAR supervision with positive and ray-defined negative
  occupancy instead of treating a point cloud as an unstructured point set.
- [EulerFlow](https://arxiv.org/html/2410.02031v3) explicitly identifies missing
  time-of-flight visibility geometry as a cause of false motion at moving
  occlusion boundaries.  Its continuous motion field remains a later teacher,
  not the next deployed candidate.
- [OCFBench](https://arxiv.org/html/2310.11239) uses ray casting to identify
  unknown voxels and excludes them from occupancy supervision/evaluation;
  `UNKNOWN != FREE` is therefore part of the representation, not a reporting
  caveat.

The earlier M1 reading remains complementary: SeFlow's static/dynamic split,
VoteFlow's local rigidity, ICP-Flow's association-before-motion, and Flow4D's
multi-frame context support motion confidence.  They cannot by themselves tell
whether a missing positive was occluded or physically departed.

## Smallest falsifier

Reuse the five consumed C25 bags only as transparent Development evidence.  At
the existing `0.12 m` 3-D voxel scale, preserve each upper/lower LiDAR origin
and endpoint before merging sensors.  A conservative voxel traversal produces
the four-state sidecar.  PDC remains the baseline alert source; point memory is
an extension only.

The single candidate must satisfy all of:

- CONTACT recall `12/12`;
- false segments `<=21`;
- induced dropout recovery `>=30/36`;
- every event's first alert no later than M1-PDC.

Report how many extension alerts were confirmed by `HIT`, cleared by
`KNOWN_FREE`, retained by `OCCLUDED`, or left `UNKNOWN`.  If the raw rays do not
cover enough remembered support to distinguish these states, record
`NOT_EVALUABLE`; do not sweep angular bins, voxel size, grace, route, lifecycle,
or age.

## Evidence boundary

This is a public-real consumed-cohort Development successor.  Future OBB truth
remains evaluator-only and cannot enter ray state, motion association, or
prediction.  C28 would establish only whether explicit visibility causes add
useful information to point memory.  It would not establish Android runtime,
natural wearer behavior, user benefit, reliability, or safety.
