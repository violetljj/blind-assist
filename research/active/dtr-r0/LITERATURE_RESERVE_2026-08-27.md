# DTR literature reserve: DR41-DR60

Status: `DATED_KNOWLEDGE_RESERVE / NO_ACTIVE_SUCCESSOR /
DOES_NOT_CHANGE_CURRENT_ROUTE`

Date: 2026-08-27

This note adds 20 peer-reviewed papers to the obstacle-risk knowledge reserve.
It is deduplicated against DR01-DR40 preserved at
`archive/pre-agent-surface-2026-08-26` and against the papers already cited in
this route. It does not authorize post-hoc tuning, changes to the current R2
decision or in-progress R3 experiment, a new route, Android integration, or
product/safety claims.

The selection target is the current DTR problem:

```text
causal wearer route tube
        intersect
static/dynamic future occupancy
        within 0-3 seconds
        -> ONSET / HOLD / ESCALATE / CLEAR or UNKNOWN
```

The search reviewed 1,025 Exa result slots across 13 discovery and validation
workstreams. That count includes duplicate mirrors and near-neighbours; it is
not presented as 1,025 independent papers. The final 20 were checked against
primary proceedings, publisher pages, author pages, or arXiv manuscripts.

## A. Perception, missing tracks, and abstention

| ID | Paper | New mechanism for DTR | Smallest useful falsifier | Evidence boundary |
| --- | --- | --- | --- | --- |
| DR41 | [Safety-Oriented Pedestrian Occupancy Forecasting](https://doi.org/10.1109/IROS51168.2021.9636691), IROS 2021 | Combine per-track futures with detector-independent dense scene occupancy, so confidence filtering or NMS cannot silently erase every future pedestrian cell. | On fresh replay, deliberately suppress true tracks and compare track-only versus track-plus-residual-occupancy event recall, false segments, and occupancy calibration. | LiDAR and HD-map autonomous-driving evidence; no wearable RGB, BLV outcome, or safety authority. |
| DR42 | [Pedestrian Trajectory Forecasting Using Deep Ensembles Under Sensing Uncertainty](https://doi.org/10.1109/TITS.2024.3382201), IEEE T-ITS 2024 | Propagate tracking covariance and observation availability into aleatoric/epistemic future uncertainty instead of passing point tracks only. | Inject bounded position noise and causal observation gaps; require empirical coverage and event trade-offs to improve over mean-only input. | Benchmark/Kalman sensing uncertainty is not proof that a phone detector exposes trustworthy covariance. |
| DR43 | [Joint Out-of-Distribution Detection and Uncertainty Estimation for Trajectory Prediction](https://arxiv.org/abs/2308.01707), IROS 2023 | Keep novelty and expected in-distribution forecast error as two independent abstention signals. Either may force `UNKNOWN`; neither may create `CLEAR`. | Freeze development thresholds, then measure cross-dataset/corruption OOD detection, error ranking, retained coverage, and event risk. | Automotive scene encoders and the Shifts dataset; predicted error is not calibrated coverage or a safety guarantee. |
| DR44 | [Scene Informer: Anchor-based Occlusion Inference and Trajectory Prediction in Partially Observable Environments](https://doi.org/10.1109/ICRA57147.2024.10611060), ICRA 2024 | Query only occluded regions that can intersect the planned route, while separating hidden-agent existence probability from its conditional future motion. | Apply causal line-of-sight masks and report occupied/free-anchor calibration plus the additional true and false route events caused by speculative targets. | Simulated occlusions on autonomous-driving representations; it cannot establish an unseen person from wearable RGB. |
| DR45 | [Uncovering the Missing Pattern: Unified Framework Towards Trajectory Imputation and Prediction](https://openaccess.thecvf.com/content/CVPR2023/html/Xu_Uncovering_the_Missing_Pattern_Unified_Framework_Towards_Trajectory_Imputation_and_CVPR_2023_paper.html), CVPR 2023 | Make observation masks and time-since-seen first-class inputs; jointly impute bounded gaps and forecast instead of treating imputed points as observations. | Freeze a maximum supported gap, then stratify random, contiguous, and real occlusions by imputation error, identity continuity, and route-event behavior. | Benchmark missingness does not prove that a fully unseen target exists; gaps beyond the supported window remain `UNKNOWN`. |

## B. Future-occupancy representations

| ID | Paper | New mechanism for DTR | Smallest useful falsifier | Evidence boundary |
| --- | --- | --- | --- | --- |
| DR46 | [Occupancy Flow Fields for Motion Forecasting in Autonomous Driving](https://doi.org/10.1109/LRA.2022.3151613), IEEE RA-L 2022 | Predict a time-indexed grid containing both occupancy and 2-D flow, preserving temporal motion consistency and allowing speculative/disoccluded agents. | Compare 0-3 s occupancy/flow and route-event metrics, stratified by visible, disoccluded, and entering-FOV targets. | Vehicle-scale data and sensors; occupancy values are not automatically calibrated collision probabilities. |
| DR47 | [Trajectron++: Dynamically-Feasible Trajectory Forecasting with Heterogeneous Data](https://doi.org/10.1007/978-3-030-58523-5_40), ECCV 2020 | Produce multimodal, dynamics-aware multi-agent futures conditioned on maps and optionally on the ego plan. | Hold the curved wearer route fixed and compare ego-route-conditioned versus independent target futures with proper distribution and event metrics. | Standard datasets assume usable tracks/maps; benchmark accuracy does not establish wearable deployment or calibrated risk. |
| DR48 | [Reliable Probabilistic Human Trajectory Prediction for Autonomous Applications](https://doi.org/10.1007/978-3-031-91585-7_9), ECCV Workshops 2024 | A lightweight LSTM-MDN makes reliability, sharpness, short-history behavior, latency, and embedded execution explicit outputs of evaluation. | Plot per-horizon residence coverage and sharpness under short histories, then verify whether any gain survives route-event scoring and the runtime budget. | No scene/social context; empirical reliability on its datasets is not universal calibration. |
| DR49 | [SCOPE: Stochastic Cartographic Occupancy Prediction Engine for Uncertainty-Aware Dynamic Navigation](https://doi.org/10.1109/TRO.2025.3578234), IEEE T-RO 2025 | Use one stochastic occupancy representation for ego motion, moving agents, and static geometry, optimized for resource-constrained real-time navigation. | On current native sources, compare the common occupancy future with R3 on calibration, event recall, false segments, lead time, and turn strata. | Robot occupancy grids do not establish phone-RGB inference, head clearance, or user benefit. This is distinct from DR10 SOGMP/SOGMP++. |
| DR50 | [Online Update of Safety Assurances Using Confidence-Based Predictions](https://doi.org/10.1109/ICRA48891.2023.10160828), ICRA 2023 | Update predictor confidence causally and widen a reachable tube when observed motion stops matching the assumed model. | Inject turns, acceleration, outliers, and identity swaps; confidence must fall before current-model misses and improve recall at a bounded false-event cost. | Driving scenarios and offline reachability machinery; the likelihood model and tube validity still need independent validation. |

## C. Collision geometry and risk aggregation

| ID | Paper | New mechanism for DTR | Smallest useful falsifier | Evidence boundary |
| --- | --- | --- | --- | --- |
| DR51 | [Optimal Alarms for Vehicular Collision Detection](https://doi.org/10.1109/IVS.2017.7995732), IEEE IV 2017 | Define collision as the union of overlap events over a horizon and choose an alarm from explicit false-negative/false-positive costs, with Monte Carlo error bounds. | Evaluate `P(any route-tube overlap in 0-3 s)`, calibration, event cost, and sensitivity to time step and sample count; freeze costs before outcomes. | Vehicle simulations and assumed motion distributions; neither costs nor probabilities transfer automatically. |
| DR52 | [Scenario-Based Trajectory Optimization in Uncertain Dynamic Environments](https://doi.org/10.1109/LRA.2021.3074866), IEEE RA-L 2021 | Convert arbitrary obstacle distributions into sampled collision scenarios and geometrically prune only route-irrelevant samples for real-time computation. | Treat pairwise velocities as scenarios; exhaustive and pruned evaluation must recall the same positive events while reducing runtime. | It bounds marginal risk per time step, not automatically the joint probability of a complete DTR event. |
| DR53 | [RADIUS: Risk-Aware, Real-Time, Reachability-Based Motion Planning](https://www.roboticsproceedings.org/rss19/p083.html), RSS 2023 | Combine offline reachable sets with an online closed-form over-approximation of collision risk for arbitrary obstacle-position distributions. | On frozen stochastic replay, empirical violations must respect a preregistered risk level without collapsing into universal alert or abstention. | Autonomous-driving simulation and hardware with modeled distributions; no BlindAssist calibration or safety transfer. |
| DR54 | [Integrating Predictive Motion Uncertainties with Distributionally Robust Risk-Aware Control for Safe Robot Navigation in Crowds](https://doi.org/10.1109/ICRA57147.2024.10610404), ICRA 2024 | Place a distributionally robust ambiguity set around predicted motion rather than trusting one fitted distribution. | Under source-disjoint motion shift, compare calibration and event trade-offs against R3 strict-majority support within the same runtime budget. | Planner/controller evidence; its chance constraint is not a validated BlindAssist warning probability. |
| DR55 | [A Generalized Continuous Collision Detection Framework of Polynomial Trajectory for Mobile Robots in Cluttered Environments](https://doi.org/10.1109/LRA.2022.3191934), IEEE RA-L/IROS 2022 | Solve continuous time-of-impact along polynomial/nonholonomic trajectories instead of relying on discrete route samples. | Generate thin, grazing, and turn-entry contacts between R3's 0.1 s samples; compare with a dense oracle and require fewer interpolation misses without extra event segments. | Known geometry and robot trajectories; no target uncertainty, wearable sensing, or user evidence. |
| DR56 | [Distance and Collision Probability Estimation from Gaussian Surface Models](https://www.kshitijgoel.com/goel-distance-2025/index.html), IROS 2025 | Approximate the body and environment as ellipsoids/Gaussian mixtures and compute distance, gradient, and blended collision probability in microseconds on embedded CPUs. | Compare lower-body/head ellipsoids against Monte Carlo geometry under box/surface perturbations and measure phone-class latency. | Primarily static surfaces; no dynamic intent, hanging-branch truth, or drop-off evidence. |
| DR57 | [Robo-Centric ESDF: A Fast and Accurate Whole-Body Collision Evaluation Tool for Any-Shape Robotic Planning](https://doi.org/10.1109/IROS55552.2023.10342074), IROS 2023 | Use a body-frame ESDF for arbitrary oriented 3-D shapes, including nonconvex bodies, with fast position/rotation queries. | Compare current circular tube plus height bands with a privileged whole-body oracle over turns, narrow gaps, and overhangs. | Robot-body/static-map evidence; it does not supply human dimensions or source-native head/ground truth. |
| DR58 | [OVPC Mesh: 3D Free-Space Representation for Local Ground Vehicle Navigation](https://doi.org/10.1109/ICRA.2019.8793503), ICRA 2019 | Build a conservative watertight free-space mesh from visible points, retaining overhang and rough-terrain structure while keeping unseen space distinct. | Test overhang, occlusion, and drop-off scenes; known clearance must survive and unobserved gaps must remain `UNKNOWN`. | UGV point clouds; no monocular/mobile RGB, pedestrian envelope, or human head/drop-off outcome. |

## D. Distribution and event evaluators

| ID | Paper | New mechanism for DTR | Smallest useful falsifier | Evidence boundary |
| --- | --- | --- | --- | --- |
| DR59 | [Evaluation of Trajectory Distribution Predictions with Energy Score](https://proceedings.mlr.press/v235/shahroudi24a.html), ICML 2024 | Shows that truth-selected minimum-of-N metrics are not strictly proper and proposes Energy Score for complete multivariate trajectory distributions. | Construct mode-collapsed and overdispersed forecasts with similar minADE/minFDE; Energy Score should distinguish them before route-event comparison. | Evaluator only; it creates neither a predictor, an abstention policy, nor an operational risk threshold. |
| DR60 | [Precision and Recall for Time Series](https://proceedings.neurips.cc/paper/2018/hash/8f468c873a32bb0619eaeb2050ba45d1-Abstract.html), NeurIPS 2018 | Range-based precision/recall separates existence, overlap, position bias, and fragmentation/cardinality penalties. | On canonical timelines, early contiguous alerts must outrank late, fragmented, or excessively long alerts; freeze weighting before model comparison. | Generic evaluator whose configurable weights can hide failures if selected after observing results. |

## Highest-value reading order for the current R3 boundary

1. **DR55** tests whether R3's 0.1-second route sampling misses thin or grazing
   contacts before any richer predictor is considered.
2. **DR42 + DR43** define a better tracker-to-forecast contract: covariance,
   missingness, novelty, expected error, and explicit `UNKNOWN`.
3. **DR41** is the most direct answer to detector/NMS disappearance: add a
   dense residual occupancy head rather than pretending a missing track is free
   space.
4. **DR59** prevents best-of-K trajectory metrics from making an overdispersed
   model look useful.
5. **DR60** maps directly to DTR's event fragmentation, useful onset, nuisance
   duration, and CLEAR behavior.
6. **DR49** is the strongest longer-term common representation for static and
   dynamic future occupancy if R3 establishes that richer prediction is worth
   the complexity.

## Deliberate exclusions

- Multi-agent conformal reachability papers were not counted because DR12
  already covers the conformal risk-tube mechanism closely.
- AgentFormer and Quo Vadis remain useful general references, but complete-track
  interaction and greater-than-three-second re-association add less immediate
  falsification power than the selected missingness and abstention papers.
- Generic detector/mAP papers, end-to-end robot-navigation systems without an
  explicit collision mechanism, and papers that only rename distance as risk
  were excluded.

## Claim ceiling

These papers support candidate mechanisms and evaluators only. Most evidence is
from autonomous driving, mobile robots, privileged maps, point clouds, or
simulated occlusion. None proves BlindAssist detector performance, phone
latency, BLV avoidance benefit, natural-distribution reliability, or safety.
`No alert` remains distinct from `known safe`; missing or unsupported inputs
remain `UNKNOWN`.
