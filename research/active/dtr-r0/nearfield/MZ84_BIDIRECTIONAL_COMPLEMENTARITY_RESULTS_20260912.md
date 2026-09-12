# MZ84 bidirectional ToF/radar complementarity falsifier

Decision: `BIDIRECTIONAL_TOF_RADAR_COMPLEMENTARITY_CANARY_PASSES`.

The zero-training responsibility-separated late fusion passes all four
predeclared gates on a new 480-frame analytic source. Fusion frame F1 is 0.944,
strictly above ToF 0.807 and radar 0.428. ToF supplies unique true evidence for
weak reflectors and low-radial-velocity lateral crossing; radar supplies unique
true evidence during strong-light wall loss and multi-target ToF gaps. Fusion
keeps the better single expert's eight FP and adds no fabricated height labels.

## Fixed design

[Protocol](MZ84_BIDIRECTIONAL_COMPLEMENTARITY_20260912.md): six families, four
20-frame episodes each, zero fit and no cutoff selection. The analytic source,
sensor observations, predictor outputs and evaluator arrays are separate sealed
artifacts. Radar keeps the MZ83 range/closing-velocity/coarse-angle rule. ToF
uses fine range/angle/height when valid plus a fixed one-second angular forecast
for lateral motion. Missing ToF stays UNKNOWN.

The fusion rule is exactly:

`ToF_alert OR (ToF_UNKNOWN AND Radar_alert)`.

Thus valid-clear ToF geometry can veto coarse radar clutter/ghosts, while radar
can speak when ToF is physically unobservable. Radar-only positive output remains
`GENERIC_FORWARD_HAZARD / HEIGHT_UNKNOWN`.

## Aggregate result

| Method | TP | FP | FN | TN | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| ToF | 121 | 8 | 50 | 301 | 0.807 |
| Radar | 79 | 119 | 92 | 190 | 0.428 |
| **Fusion** | **160** | **8** | **11** | **301** | **0.944** |

All gates pass:

- fusion F1 is strictly best;
- distinct ToF-win and radar-win families both exist;
- fusion FP equals the higher-F1 ToF expert, within the allowed +2 frames;
- every positive event's first correct fusion alert is 0.0 s later than the
  earlier correct single-expert alert, below the 0.2 s limit.

## Failure-direction decomposition

| Family | Truth-positive frames | ToF TP/FP/FN | Radar TP/FP/FN | Fusion TP/FP/FN | Unique direction |
| --- | ---: | ---: | ---: | ---: | --- |
| weak reflector | 47 | 47/0/0 | 0/0/47 | 47/0/0 | ToF wins |
| off-corridor clutter | 0 | 0/0/0 | 0/57/0 | 0/0/0 | ToF valid-clear veto |
| wall / multipath | 26 | 0/0/26 | 24/37/2 | 24/0/2 | Radar wins under ToF loss; ToF vetoes ghosts |
| lateral crossing | 48 | 44/0/4 | 12/0/36 | 44/0/4 | ToF wins over low radial velocity |
| head motion | 0 | 0/8/0 | 0/25/0 | 0/8/0 | shared residual failure |
| multi-target | 50 | 30/0/20 | 43/0/7 | 45/0/5 | both directions; radar dominates gaps |

ToF has unique true frames in weak-reflector, lateral-crossing and multi-target
families. Radar has unique true frames in wall/multipath and multi-target.
Therefore the result is not another single-sensor dominance case.

## Attribution and temporal audit

ToF supplies valid BODY/HEAD attribution for 121 true frames. Thirty-nine true
radar-only fusion frames remain HEIGHT_UNKNOWN. Fabricated radar height labels:
zero.

Twelve of fourteen positive episodes have one continuous fusion-alert segment.
Two multi-target episodes have two segments because the ToF gap begins one frame
before radar's two-of-three activation. Their first correct alert is still not
delayed. Preserve this fragmentation defect; do not posthoc add fusion hold time
on the consumed source. It is a concrete responsibility for a later collision
state estimator.

All eight remaining fusion FP are from the head-motion family and already occur
in ToF. Radar produces 25 head-motion FP, but valid ToF suppresses the additional
17. This shows both that the observability gate is useful and that neither sensor
alone separates ego-motion cleanly. IMU is now a mechanism-level requirement for
the next temporal state-estimation stage, not evidence of measured benefit.

## Interpretation and limits

MZ84 establishes the desired logical property under constructed conditions:
heterogeneous sensing plus explicit responsibility allocation can outperform
both experts without buying recall through extra FP or delayed first alerts.
It supports the research direction `ToF geometry + radar generic dynamics + IMU
ego-motion`, rather than learned modality voting.

It does not estimate real-world performance or failure prevalence. Every RCS,
multipath, sunlight, gap, target-motion and head-motion effect is an analytic
proxy chosen to exercise the declared failure direction. No natural imagery,
UE rendering, actual radar, actual ToF, people or IMU is present. The result is
policy-logic evidence and a falsifiable sensor contract, not hardware validation.

Retain the fixed fusion as a `COMPONENT_OR_CHALLENGER`. A successor should change
the information source: either measured/credible device-calibrated sensor traces
or a bounded ego-motion-compensated state estimator with fresh simulation. Do
not train learned fusion or claim alert, deployment, user-benefit, or safety
improvement from this canary.

## Evidence and verification

Canonical evidence is
`artifacts.local/work/mz84-bidirectional-complementarity-20260912/run-v2/`.
It contains the source and receipt, observable packets, sealed predictions,
evaluator arrays, result, source snapshot and final hash receipt. Three focused
tests pass for the six-family source, clip-isolated causal hysteresis, confusion,
F1 and event accounting. Python compilation, knowledge validation and
`git diff --check` pass. CPU execution took under two seconds.
