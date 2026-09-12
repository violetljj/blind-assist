# MZ89: joint temporal geometry and spatial Radar compatibility

Mode: EXPLORE, one zero-fit constructed simulation; no successor auto-launch.

Question: can a joint angular state retain early crossing evidence while reducing
boundary false alerts relative to hard MZ85 and a simple one-frame hold?
MZ88 rejected its particular scalar uncertainty/coarse-Radar recipe, not all fusion.

## Frozen implementation and source

Use 36 new 30-frame, 10 Hz analytic episodes, seed 89012, six existing stress
families with new trajectories, distances, offsets, crossing speeds and gap times.
Six IMU conditions: nominal, both signed medium errors, two signed two-frame gaps,
and a mixed one-frame gap. Sensor proxy and error magnitudes inherit MZ88.
Source-aware constructed Development only, not natural or source-disjoint evidence.
Materialize and hash observations before prediction; save/hash predictions before
evaluating truth. No old outcome files are predictor inputs. One scoring run, no
parameter search or result-driven retry; mechanical fixes must retain identity.

Inputs remain identical across arms: exact analytic ToF range/angle returns,
synthetic lateral-motion hint and height tag, coarse already-stabilized Radar,
corrupted IMU increments. These are NOT raw RGB, an 8x8 ToF sensor model or measured
IMU distributions. The hint and Radar coordinate privilege remain explicit shared
limits. Predictor input excludes truth, family, variant and true pose.

Joint state is the narrow angular/range equivalent of a sparse track distribution,
not dense occupancy or a physical BODY/HEAD swept volume. Preserve 12-degree
corridor, 3.18 m direct / 3.6 m crossing limits, one-second forecast and k=1.
Propagate shared extrinsic (2 deg), bias (2 deg/s) and timing (20 ms) latent
errors, accumulated independent gyro increments (2 deg/s), and persistent missing
increment uncertainty. Coefficients L give covariance C=L L^T. Forecast uses
theta_future=(1+1/dt)*theta_now-theta_previous/dt and its full covariance.
Yaw uncertainty is a first-order angular proxy; no full SE(3)/translation claim.
Never multiply repeated occupancy probabilities or shrink shared error by repeat count.

Match previous observed ToF returns causally and mutually uniquely by range/angle
(0.5 m, 15 deg fixed gates; normalized squared nearest distance; ambiguous ties
abstain), use actual elapsed time, retain history for at most 0.3 s. Matching is
common to the three new-state arms and does not use object IDs. Missing motion
variance persists after IMU recovery. No history-only candidate alerts.

Radar may confirm UNCERTAIN only when its current qualifying return overlaps the
same ToF support: range difference <=0.35 m, angular difference <=sqrt(sigma^2+7^2)
deg, and abs(Radar azimuth)+7<=12 deg. Seven degrees is the declared proxy angular
envelope (5-degree half-bin plus 2-degree noise), not fitted calibration. Require
the unchanged Radar temporal expert too. Missing ToF uses the inherited Radar-only
fallback, with unknown height. Known-empty ToF is distinct from missing ToF.

Arms, all frozen before scoring:

- hard_mz85: unchanged MZ88 hard baseline on new packets (legacy history limits disclosed).
- hard_hold: hard baseline plus one non-reseeding frame only on missing ToF.
- mz88_full: exact rejected scalar/coarse-Radar/hold negative control.
- marginal_spatial: new causal matching/state, but covariance off-diagonals removed
  in forecast; spatial Radar rule. This isolates temporal covariance.
- joint_coarse: joint forecast but legacy broad Radar confirmation; isolates Radar.
- joint_spatial: joint forecast plus spatial compatibility, primary candidate.

## Checks and stop

Report TP/FP/FN/F1, UNKNOWN including true-positive UNKNOWN, per-family results,
false segments, and metrics for each contiguous truth event (misses, first correct
support, fragments). Compare first correct support only within truth events;
earlier false alerts are not improvement. No far-thin-pole claim or slice exists.

Primary candidate qualifies only if: F1 >= both hard controls; TP >= hard TP-2;
FP <= both controls; stressed boundary/head FP reduced by at least 25% with nonzero
baseline opportunities; crossing TP >= hard crossing TP-2; no nominal TP losses or
new FP; every baseline-detected event is detected with <=0.2 s added delay; false
segments and total within-event fragments do not increase versus hard baseline.
Require a strict task-effect difference from hard_hold, plus no authority violations.
Attribute covariance and Radar effects against the two fixed ablations; if either
does not help, do not claim that component. Numerical covariance correctness is
not task benefit. If gates fail, retain only diagnostic/negative-control evidence,
stop this recipe and report the actual tradeoff. No threshold rescue, training,
additional source collection, MZ86 resumption or physical collision head follows.

CPU scalar-scoring through research_backend; durable source, predictions, evaluator,
hashes and receipts under artifacts.local/work/mz89-joint-geometry-20260912/.
Focused checks: covariance cancellation/PSD, persistent gaps, elapsed-time history,
return permutation and incompatible Radar, missing/known-empty distinction,
episode reset/causality, authority/height, and contiguous event accounting.
