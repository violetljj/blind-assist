# NF-G6 — Suspended Structure Information Probe

EXPLORE, controlled synthetic Development. G5 ruled out two tested substitutions,
not all depth models. Test whether sparse RGB structure supplies near evidence
when dense metric depth misses a suspended bar. No training or model sweep.

## Fixed source and observation boundary

At most 36 posed captures: twelve three-view static-scene windows. Bars at 1,2,3,
6 m with forward translations (0,.1,.2 m) and lateral translations (-.1,0,.1 m);
near2/far6 yaw-only (-2,0,2 degrees); forward-motion distant wall-mark and empty
scene controls. Frozen Willow map, existing bar dimensions and lighting. These
are geometrically ordered poses, not real-time optical-flow or warning-latency
sequences. Preserve every clip, visibility gap and failed match.

Matcher sees RGB, calibration and ideal metric camera poses only. Native depth
and object geometry are evaluator-only. Known translation scale and attitude
are privileged inputs: IMU alone is not assumed to supply metric translation.
Pure rotation or unobservable motion must yield UNKNOWN distance. No motion
signal cannot prove RGB intrinsically insufficient; texture, aperture, occlusion
and matcher failure remain alternative explanations.

## Fixed information test

One gradient maximum per 16x16 image tile, magnitude >=.04 on grayscale [0,1],
up to 920 candidates at 640x360. Nine-pixel patches; GPU normalized correlation
over 96 inverse-depth hypotheses from .05 to 2 per metre, shared across the two
earlier views. Compare best score with rotation-only correspondence (edge
persistence) and report rotation-compensated displacement (parallax).

Accept a depth only when both views have correlation >=.75, best-versus-other
inverse depths outside +/-20% has margin >=.05, the score-within-.03 hypothesis
interval spans <=50% of the best inverse depth, and final-baseline parallax
>=1 pixel. Missing support remains UNKNOWN. Reject hypotheses outside images;
no GT-selected candidates or per-clip thresholds. These are diagnostic fixed
settings, not calibrated operational confidence probabilities.

Compare fixed Hypersim Small518 at each endpoint (12 new calls maximum) for
metric depth and local inverse-depth contrast (centre versus four neighbours
eight pixels away). A contrast or edge alone never creates a near alert.
Dense models and patch matching run CUDA; metadata and native-label summaries
may use CPU. Cache before evaluation and record compute separately from capture.

Primary: endpoint target-bar candidate coverage, accepted near/far/UNKNOWN counts
and clip recovery, reported separately by motion/distance. Report false near
evidence on far/mark controls, all-candidate native agreement, edge persistence,
parallax and dense residual distributions. Target-mask membership is derived
only after predictions from native depth and known object bounds; it is not
learned instance accuracy. No retroactive changes to G0–G5 denominators.

Keep the sparse mechanism as a candidate only if it recovers at least one near
bar missed by dense depth with no target false-near on far/mark controls; disclose
all non-target false near matches too. Success supports an ideal-pose geometric
signal, not deployability or full thin-obstacle coverage. No gain retains the
failure and identifies observability/matching gaps. Stop after these 12 clips,
one fixed matcher and 12 endpoint model calls; mechanical bugs may be repaired
with receipts, no outcome-driven parameter sweep. Do not integrate App defaults.

## Results

Completed: 36 captures in 60.55 s, twelve endpoint MDE calls, twelve sparse
matches. Map unchanged, all targets visible in native geometry, all owned UE
processes released. The pre-score copy remains at the experiment root. Matching
uses 9x9 patches; evaluator-only attribution was extended once from cached
predictions to expose overlapping rejection reasons, without changing matches.

| Motion / initial bar distance | Target candidates | Accepted near body/head | Best optical depth median (including rejected) | NCC median | Ambiguity margin median |
| --- | --- | --- | --- | --- | --- |
| forward / 1 m | 14 | 11 | .762 m | .962 | .190 |
| forward / 2 m | 5 | 0 | 1.843 m | .984 | .032 |
| forward / 3 m | 2 | 0 | 2.718 m | .946 | .024 |
| forward / 6 m | 0 | 0 | unavailable | unavailable | unavailable |
| lateral / 1 m | 10 | 1 | 10.983 m | .992 | .0008 |
| lateral / 2 m | 1 | 0 | 4.668 m | .998 | .0042 |
| lateral / 3 m | 1 | 0 | 20.000 m | .980 | .0021 |
| lateral / 6 m | 0 | 0 | unavailable | unavailable | unavailable |
| yaw / 2 m | 3 | 0 | unobservable | .982 | 0 |
| yaw / 6 m | 1 | 0 | unobservable | .963 | 0 |
| forward / far wall mark | 4 | 0 | 1.258 m (rejected) | .919 | .0078 |

The empty-scene control has no target denominator. Forward endpoints are .2 m
closer than initial distances; optical depth differs from gravity-aligned range.
Best-depth medians include rejected hypotheses and are not accepted estimates.

**No retained sparse alert branch yet.** Only the two 1 m clips have accepted
target near evidence, and MDE already has near pixels on both (12,306/12,584 and
2,814/8,006 native target pixels respectively). None of the four translational
2–3 m cases missed by MDE is recovered. The predetermined incremental-recovery
criterion fails. Dense predictions have zero near target pixels on these four
cases; their target-candidate median depths are 12.25/14.52/14.09/16.53 m.

There is nevertheless diagnostic signal: forward 2/3 m target correlations
improve from rotation-only .504/.575 to .984/.946, with best-hypothesis parallax
3.33/2.97 pixels. At 2 m, one candidate fails correlation and four fail ambiguity;
at 3 m both fail ambiguity. These high matching scores are insufficient to make
unique range claims. Lateral 2/3 m scores barely improve over rotation-only
.966/.980, consistent with aperture ambiguity along a horizontal bar; this is
an interpretation of these views, not proof that lateral motion never helps.

Far-bar translational clips have **zero selected target candidates**, despite
224/220 visible native target pixels: proposal coverage is itself a bottleneck.
The four far-mark candidates and pure-yaw candidates remain UNKNOWN. Zero target
false-near matches is therefore not evidence of robust far rejection. Across all
8,300 image candidates there are 45 accepted near versus native far mismatches,
16 of them also in predicted body/head height. These are pixel-reference errors,
not distinct obstacle or warning counts; edge-centre background and patch support
can disagree at boundaries. They remain reported rather than discarded.

Local dense inverse-depth residual medians on forward 2/3 m targets are
.000515/.003540 per metre; lateral 2/3 m .000315/.000138; far-mark .000188. Small,
uneven target coverage and no translational far-bar candidates prevent claiming
a near/far separator from these values. No residual-triggered alert was added.

Sparse CUDA timing P50 **13.87 ms**, mean61.97 ms including593.28 ms first-call
startup; MDE P50 **93.65 ms**, mean116.41 ms. These exclude camera acquisition,
image decoding and pose estimation; they do not establish Android latency.

Next decision: preserve dense/fusion baseline. The next candidate should improve
coverage along connected line segments and assess distance observability along
edge normals/endpoints, rather than lower the confidence threshold. G6 neither
establishes an RGB information impossibility nor warrants training a residual
network. Such a successor requires a separate scope; no tuning or extra capture
was performed after these results. No App promotion or dynamic-warning claim.

Evidence under `artifacts.local/nearfield/structure-information-20260907-v1/`:
`capture/verification.json`, `predictions/result.json`,
`evaluation/result.json`, `evaluation-attribution/result.json`.
Prediction SHA-256: `07b480268f415af8b24bae83896dbcbbe61e8da6491c23c61321083c1037e25c`.
Attributed result SHA-256: `8efbee40a1df9f4e6202ccae98a064be82af599ddf469c6c8c8430423cc143f0`.
Three focused tests cover projection/pitch, textured-plane depth recovery, and
zero-baseline UNKNOWN. A NumPy-bool/Torch dispatch
bug was repaired during unit checks before any experiment prediction.
