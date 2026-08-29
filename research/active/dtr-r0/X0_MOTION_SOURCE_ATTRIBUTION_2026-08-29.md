# DTR-X0 motion-source attribution

Date: 2026-08-29

Status: `DTR_X0_MOTION_SOURCE_ATTRIBUTION_COMPLETE`

## Decision first

X0 selects `STRONGER_SCENE_FLOW_SOURCE` as the next dynamic-risk experiment.
Do not train a learned motion-authority head yet, do not tune C31, and do not
open C32 probabilistic body-route occupancy.

The reason is source-level rather than component-authority-level. Of the 35
diagnosed false segments, 34 are `BAD_FLOW` or `STATIC_PSEUDO_MOTION`; only one
is a correctly estimated real mover that is irrelevant to the route. One of
the two missed CONTACT events has no raw cell support at all. A learned
authority cannot recover absent motion and should not be asked to learn around
a source whose false evidence dominates.

The next falsifier should replace only the motion source: compare the current
raw direct flow against exactly one stronger scene-flow representation while
keeping the route scorer, 3 s horizon, body-route radius, lifecycle, cohort,
and evaluator unchanged. Huang-2 also authorizes a separate, bounded continuous
geometry canary; geometry alone cannot address the unsupported Huang-lane miss
or the 34 source-error false segments.

## Frozen diagnostic contract

X0 is post-outcome scorer-side attribution on the already opened six-sequence
C31 confirmation cohort. It does not rescore or modify a prediction.

- Native OBB identity and same-history trajectory are privileged diagnostic
  labels only.
- Cell-to-OBB association reuses the frozen R7 margin
  (`0.0848528137 m`).
- Flow correctness reuses the frozen `0.25 m/s` motion floor as an absolute
  velocity-error limit.
- Miss support is inspected only in the `-3..0 s` window before realized
  CONTACT; two frames are required for sufficient temporal support, and
  `1.5 s` is the existing urgent boundary.
- False-segment attribution uses the first raw route-risk frame in each sealed
  lifecycle segment. Frame-local C31-compatible grouping diagnoses identity
  split/mix only at that frame.
- `UNKNOWN` and `NOT_EVALUABLE` are not negative evidence. No unit in this run
  required `NOT_EVALUABLE`.

## Miss attribution

| missed CONTACT | raw support | correct raw motion | frozen route entry | attribution |
| --- | ---: | ---: | ---: | --- |
| `huang-2-2019-01-25_0:contact:001` | 32 cells / 23 frames | 7 cells / 6 frames; 3 early frames | 0 frames | `ROUTE_GEOMETRY_MISS` |
| `huang-lane-2019-02-12_0:contact:001` | 0 cells / 0 frames | 0 | 0 | `NO_MOTION_SUPPORT` |

Huang-2's responsible pedestrian reaches only `0.145 m/s` native speed in the
window. Correct raw velocity nevertheless appears from the first 3 s origin
(`0.188 m/s` minimum error), but the point/constant-velocity route geometry
never enters the route tube. This is a wearer-route/occupancy geometry gap, not
evidence that C31 needs a longer memory.

Huang-lane has no raw M1-PD cell associated with `pedestrian:56` anywhere in
the `-3..0 s` window. Neither hand-written nor learned authority over the
current source can recover that event.

## False-segment attribution

| sealed set | `BAD_FLOW` | `STATIC_PSEUDO_MOTION` | `REAL_MOVER_NONCRITICAL` | total |
| --- | ---: | ---: | ---: | ---: |
| M1-PDC false segments | 14 | 10 | 1 | 25 |
| C31 non-overlapping incremental false segments | 2 | 8 | 0 | 10 |
| combined diagnostic units | **16** | **18** | **1** | **35** |

The C31 increment is especially decisive: its two Huang-2, five
Huang-basement, and one Memorial additions are static/unmatched pseudo-motion;
the two Huang-lane additions are bad velocity. None is primarily a correctly
estimated noncritical mover, fragmentation, or wrong-component binding.

For PDC, the only `REAL_MOVER_NONCRITICAL` segment is in Huang-lane. The other
24 are split between wrong velocity and pseudo-motion. Across both sets, source
errors are `34/35`; therefore track/residual disagreement remains a useful
future feature, but it is not sufficient reason to train an authority head on
this source.

## Evidence and claim boundary

Run:

```powershell
E:\codex-tools\tools\venvs\blindassist-torch-gpu\Scripts\python.exe `
  research\active\dtr-r0\dtr_x0_motion_source_attribution.py
```

- result:
  `artifacts.local/evidence/dtr-x0/motion-source-attribution/result.json`
- result SHA-256:
  `b19330e3f84764b09cf2cee5f47cde65bec932f468ccbf621b8d16694e71a23e`
- attribution table:
  `artifacts.local/evidence/dtr-x0/motion-source-attribution/attribution.csv`
- table SHA-256:
  `0681af64d4baaa6788fed1def6f8d986be994ea09f2971aa3bf108dfdc352c9b`

This is explanatory diagnosis on consumed, opened truth. It adds no recall,
false-alert, generalization, deployment, product, user-benefit, or safety
evidence. An unmatched native OBB is consistent with static pseudo-motion but
may include an unlabeled mover. X0 chooses the next information-source test; it
does not establish that a particular learned scene-flow method will succeed.
