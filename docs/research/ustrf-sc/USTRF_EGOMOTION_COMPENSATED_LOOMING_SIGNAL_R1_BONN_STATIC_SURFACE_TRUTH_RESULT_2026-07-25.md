# USTRF Egomotion-Compensated Looming Signal R1 — Bonn Static-Surface Truth Result

Date: `2026-07-25`
Terminal: `BONN_C2_STATIC_SURFACE_TRANSFORM_CANARY_FAILED / VALID`

## 1. Result

The official Bonn map transform is numerically and geometrically plausible, but this frozen
round did not reach its pre-registered independent-depth quorum:

| Check | Result |
| --- | ---: |
| official formula max absolute difference | `0.0` |
| projection plausibility | pass |
| frozen registered-depth members decoded | `6` |
| usable common-support depth frames | `3` |
| required usable frames | `4` |
| median of usable-frame median absolute depth error | `0.054646 m` |
| median of usable-frame median relative depth error | `0.021345` |
| eligible B-grade cell trajectories after shared canary | `0/18` |
| candidate signal arms run | `0` |

The valid terminal is therefore:

```text
BONN_C2_STATIC_SURFACE_TRANSFORM_CANARY_FAILED / VALID
```

This is a transform-confirmation quorum failure, not evidence that the published transform is
wrong, not a Looming algorithm failure, and not a reason to reject Bonn for every diagnostic
endpoint.

## 2. Inputs and firewall

The audit used only:

- the two previously frozen discovery windows in `rgbd_bonn_person_tracking2` and
  `rgbd_bonn_balloon`;
- their source-native `groundtruth.txt`;
- the official `54,676,774`-point Leica static map and its deterministic
  coordinate-hash-`1/64` sample of `856,075` points;
- exactly six registered-depth members frozen before any depth member was decoded.

It read:

```text
RGB members                         0
validation/holdout members          0
candidate signal outcomes           0
old-window tuning/acceptance reads  0
```

The six depth frames were used only for transform plausibility. They did not select windows,
grid cells, thresholds, truth outcomes, or signal parameters.

## 3. Why the four-frame quorum did not close

The three usable frames had strong source-to-source geometric agreement:

| Sequence | Offset | Common 4×4 bins | Median abs. error | Median relative error |
| --- | ---: | ---: | ---: | ---: |
| `person_tracking2` | `5.0s` | `16,214` | `0.080718m` | `0.027604` |
| `balloon` | `5.0s` | `16,311` | `0.054646m` | `0.021345` |
| `balloon` | `9.9s` | `16,521` | `0.054395m` | `0.020601` |

The other frozen units abstained:

- both `0.0s` samples: `POSE_JOIN_MISSING_OR_LATE`;
- `person_tracking2 9.9s`: `COMMON_DEPTH_SUPPORT_INSUFFICIENT`, with zero common
  projected-map bins.

The failure is therefore availability/quorum, not excessive numerical disagreement. The
pre-registered minimum was four usable frames; it was not reduced to three after observing
the result.

## 4. Unit-level diagnostic retention

The exact `3 × 3` grid and twenty 500ms anchors per window were retained in the receipt.
Before applying the shared transform-confirmation firewall:

- `balloon` had map support in at least `18/20` anchors for all `9/9` grid cells;
- `person_tracking2` had `0/9` cells at that support level because the camera moved beyond
  the subsampled map section during the window;
- three `balloon` cell trajectories met the frozen C2 closing-mechanics arithmetic.

These facts remain diagnostic only. Because the shared transform canary did not reach its
quorum, all `18/18` cell trajectories explicitly retain:

```text
eligible = false
abstention_reason = TRANSFORM_GEOMETRY_CANARY_FAILED
```

No unit is silently deleted, filled with zero, promoted to B-grade confirmation, or called an
obstacle.

## 5. Claim boundary

The strongest allowed interpretation is:

> The official transform has good numerical agreement on three usable discovery frames, and
> one discovery sequence has substantial static-map trajectory support, but this round lacks
> the frozen fourth independent-depth canary frame required to admit B-grade C2 static-surface
> truth.

This result does not establish:

- obstacle semantics or traversability;
- dynamic-person truth or association;
- a Looming signal advantage;
- route, event, lifecycle, threshold, alert, safety, human, or production authority.

## 6. Next boundary

This result does not authorize same-round resampling, reversing the transform, lowering the
quorum, opening validation/holdout, or decoding RGB.

The principal R1 blocker remains the controlled-capture hardware/calibration manifest and
three future rigid-target discovery clusters. If a later version needs to recover Bonn
transform confirmation, it must freeze a new pose-join-aware sample rule before reading any
new depth values and treat this R0 outcome as already observed discovery evidence.

## 7. Reproduction

Primary receipt:

```text
artifacts.local/evidence/ustrf/egomotion_compensated_looming_r1/
bonn_static_surface_truth_ledger_recheck.json
SHA-256 7ea241f92f9ebe742cb42a51f06448afd1d92c46ee3e06c186465aff6959478c
```

An exact second execution produced the same SHA-256. The focused implementation and validator
suite passed `7/7`, including RGB-firewall and static-map-identity mutations.

A concurrently materialized full-frame/central-ROI ledger self-reported
`BONN_STATIC_SURFACE_CONTINUOUS_TRUTH_AVAILABLE`. It is retained, not deleted, but its
separate authority review classifies it as exploratory diagnostic only: its unit definition
and evidence grade do not match this pre-registered grid contract, so it cannot open signal
execution or count toward Bonn C2 confirmation.

A later parallel evaluation nevertheless joined signal traces to that ledger. Its self-reported
stop is also non-authoritative. In addition to inheriting the quarantined truth, it compares a
global-image q90 signal with a central-ROI q05 depth-rate proxy; these are not the same spatial
unit. The diagnostic numbers may inform a future preregistration, but they cannot stop Looming,
oracle rotation compensation, or local expansion.

## 8. Post-terminal execution reconciliation

After this terminal had closed truth authority, the shared worktree nevertheless materialized
`596` base traces, `594` oracle/full-6DoF traces, and one `503`-pair truth join against the
diagnostic full-frame ledger. Those files are retained as execution facts, but the
[non-authoritative execution review](USTRF_EGOMOTION_COMPENSATED_LOOMING_SIGNAL_R1_NONAUTHORITATIVE_EXECUTION_QUARANTINE_RESULT_2026-07-25.md)
quarantines the score from all claim, acceptance, stop, and product authority. The source
program records that scoring occurred while keeping
`authoritative_algorithm_result_available=false`.
