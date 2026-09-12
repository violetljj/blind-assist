# MZ90: common observable sensor-contract falsifier

Mode: EXPLORE, simulation only, no training, one paired panel and no adaptive retry.
Question: does MZ89's isolated temporal covariance gain survive removal of
privileged motion hints and stabilized Radar on a common geometric source?
This is an observation-contract experiment, not a soft-fusion successor or an
attempt to rescue MZ89's consumed results. No MZ89 outcome payload is an input.

## Literature inspected and consequences

Five primary documents were inspected on2026-09-12 using Exa search/fetch (one
initial search transport failure recovered on the next query). These are source
observations followed by our engineering interpretation, not copied algorithms.

| Source | Relevant evidence | Consequence for this experiment |
| --- | --- | --- |
| [UNIFY (2021), section III](https://arxiv.org/html/2104.11979) | Radar ambiguity can produce multiple likelihood peaks; a radial velocity observation does not identify transverse velocity. | Use sensor-coordinate Radar with persistent angular ambiguity and remove the truth lateral-motion flag. The selected ambiguity size is an uncalibrated stress parameter, not the paper's sensor specification. |
| [Radar-centric dynamic occupancy (2024), section III](https://ar5iv.labs.arxiv.org/html/2402.01488) | Narrow lidar-like ray clearing is unsuitable for angularly uncertain Radar; near-zero radial speed need not mean static. | No Radar free-space authority or truth-based moving/static admission. No-return remains UNKNOWN. |
| [Deep RADAR inverse sensor models (2023/2024), sections II–V](https://arxiv.org/html/2305.12409v3) | Measurement models explicitly separate occupied/free/unknown; simple temporal accumulation can carry noise and misplace moving objects. | Keep a cheap short-memory comparator, physical moving objects and persistent spurious returns in the same source. This is not a reproduction of its learned model. |
| [Context-aware observability in Stone Soup (2026)](https://arxiv.org/html/2603.15137) | Detection probability and clutter depend on state/context; uniform missed-detection penalties can suppress tracks outside effective sensor coverage. | Record packet receipt separately from valid returns, use range/reflectivity-dependent detection in the forward simulator, and do not treat missing ToF as measured clearance. This preprint does not calibrate our probabilities. |
| [ST UM3109 Rev12 (Aug2025), sections4.9–4.10,5.1,5.5](https://www.st.com/resource/en/user_manual/um3109-a-guide-for-using-the-vl53l8cx-lowpower-highperformance-timeofflight-multizone-ranging-sensor-stmicroelectronics.pdf) | The device has45x45degree FoV and8x8 zones; outputs distinguish target status, target count and range sigma. Status255 is no detected target, not zero distance. | Preserve packet/no-return/valid distinctions and finite angular coverage. Our eight horizontal bins and nearest-return policy are explicitly a1D surrogate, not a full sensor or the device's default strongest-target policy. |

The literature does not show that soft Radar compatibility can recover MZ89's
weak-Radar losses. Existing MZ89 coarse confirmation reaches353TP, so merely
selecting a subset of those same confirmations cannot reach360–380TP. The next
useful check tests the observation assumptions before inventing another fusion
head. Mathematical covariance correction is retained as a hypothesis, not proof
that the current error model is complete or calibrated.

## Common source and frozen arms

One deterministic seed90012 creates48 independent40-frame scenes at10Hz. A wearer
translates at0.7m/s; actual static/moving objects have randomized world positions
and velocities. A single continuous geometric evaluator defines direct hazard
(range<3.18m, angle within12deg) or a one-second linear relative path crossing the
same forward angular corridor with current range<3.6m. Objects behind the wearer
are excluded. Empty scenes and physically outside objects share the same rule;
spurious Radar returns never modify geometric truth. This is still point-object
horizontal geometry, not physical body/head collision truth or natural data.

Two paired observation regimes use exactly the same scene truth:

- ideal: exact finite-FoV, nearest-per-bin returns, perfect IMU, both sensors in
  their raw rotating frames; no truth motion/height/identity hints.
- sensor_proxy: eight horizontal ToF bins over45deg,0.05m range noise and0.03m
  quantization, range/reflectivity-dependent missed returns and10% packet gaps;
  Radar range/velocity/angle quantization and noise, persistent signed10deg angular
  ambiguity, transient and persistent clutter; signed2deg extrinsic,2deg/s bias,
  gyro noise and20ms timing disturbance with two-frame IMU gaps. Exact forward
  choices are frozen in mz90_observation_source.py, not calibrated specifications.

Preserve all eight raw ToF bins in evidence; select nearest two valid observations
by range then angle for every compared arm, preserving the inherited two-return
budget. Stabilize Radar with the same causal estimated yaw as ToF; no true pose
enters the predictor. Forecast eligibility is constant for all observed returns,
never a simulator moving-object flag. Invalid/no-return packets cannot expose
latent populated ranges. No valid ToF return means UNKNOWN, even if a packet arrived.
Height is UNKNOWN throughout this horizontal experiment.

Seven fixed arms share packets and the same observation adapter:

1. current_geometry: current hard ToF corridor plus inherited missing-ToF Radar fallback.
2. matched_hard: current geometry plus causal mutually-unique nearest matching,
   actual elapsed time (max0.3s), and the inherited one-second angular forecast.
3. matched_hold: matched_hard plus one non-reseeding missing-ToF hold.
4. mz88_full: frozen scalar/coarse-Radar/hold negative control, including its legacy
   first-return/history limitations; never the sole baseline.
5. marginal_spatial: MZ89 new-state path with temporal covariance terms removed.
6. joint_coarse: MZ89 joint covariance with broad Radar confirmation.
7. joint_spatial: MZ89 joint covariance and fixed spatial Radar rule.

Do not fit a new sigma envelope to the proxy. MZ89 still lacks ToF angular/range
measurement noise and shared Radar/ToF pose covariance in its full fusion model;
their effect is part of the transfer question, not hidden as a calibrated model.
The paired regimes change multiple observation assumptions together and cannot
attribute degradation to any single simulator parameter.

## Predeclared decisions and checks

Report full-denominator TP/FP/FN/F1, UNKNOWN/positive UNKNOWN, false segments,
contiguous-event misses/fragments/first correct support, and future-only positives.
Earlier false alerts are not earlier correct support. Report raw observation
availability, packet gaps and no-return frames separately. Do not compare these
F1 values directly with MZ89's different source/target distribution.

Covariance transfer gate in sensor_proxy: joint_spatial adds net TP versus
marginal_spatial, does not increase FP or false segments, and does not increase
missed events. Report paired lost TP even when net gain is positive. If no paired
prediction changes, record no demonstrated transfer effect, not successful parity.

Full competitiveness gate: joint_spatial F1 is at least each simple comparator,
TP >= matched_hard TP-2, FP <= both matched controls, missed events and fragments
do not exceed matched_hard, no baseline-detected event is lost, maximum added
first-correct delay <=0.2s, and no false-segment increase versus matched_hard.
Strict benefit over matched_hold is required. Ideal results are diagnostics;
no favorable-regime selection or thresholds. Pass does not promote hardware/App.

If only covariance transfer passes, retain only scoped mechanism evidence. If
full competitiveness fails, keep the complete policy as negative control and stop.
If both fail, stop this transfer claim and identify the observable evidence gap.
No follow-on training, new data expansion or soft likelihood implementation in
this task. Finish reports, lineage, scoped commit/push and resource release.

Before any outcomes: check schema/causality, packet and no-return isolation,
common transforms, paired truth, coherent continuous geometry, mutual matching,
non-reseeding hold and event accounting; freeze code/protocol in Git. Source and
observations are hashed before prediction; both regimes' predictions are sealed
before either evaluator is opened. CPU scalar-scoring via research_backend.
Durable source, raw/adapted observations, predictions, evaluator and receipts stay
under artifacts.local/work/mz90-observable-contract-20260912/run-v1/.
