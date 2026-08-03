# M3D-CF route status

Date: 2026-08-03

Decision: `KEEP_CANDIDATE_SIDE_LANE_CURRENT_OCCUPANCY_ONLY`

## Conclusion

The route has produced a real candidate, but it has not earned mainline
promotion.

The supported core is:

```text
ordinary calibrated RGB at 10 FPS
-> UniDepthV2-S metric depth + confidence
-> class-free 3D clearance bands
-> causal RAFT motion evidence
-> frozen current occupancy probability
```

It does not require ARCore, phone-native depth, YOLO boxes, object classes, or
an RGB-D camera at inference time. “M3D-CF” now names the metric collision-field
research problem, not a requirement to use the Metric3D model specifically;
the frozen A0.1 evidence uses UniDepthV2-S. Independent consumed metric-depth
quality tests favor Metric3D, while the new same-runtime Android screen favors
UniDepth on deployment cost. Those are different claims and neither authorizes
a silent source swap.

## Evidence ledger

| Stage | Exact outcome | Meaning |
|---|---|---|
| Deterministic 3D clearance | fresh fail | metric depth alone was too noisy for hard collision decisions |
| A0.1 motion-conditioned current occupancy | fresh pass | first supported algorithm result |
| A1 current-risk arm comparison | mixed fail | probability field led Brier/F1/recall; 2D corridor narrowly led MCC |
| A1 consumed incremental ablation | not supported | motion raised recall 0.59 pp but worsened pooled Brier 3.94%, F1, and MCC |
| A2 future occupancy | Development fail | useful 0.5 s signal, but FPR exceeded gate |
| A2.1 2D+3D+motion future occupancy | Development pass, fresh fail | did not transfer walking to sitting regime |
| Bonn cross-dataset proxy | not evaluable | source pose was not a calibrated RGB optical camera floor reference |
| PC runtime | complete | supported route about 50.25 ms mean component sum on RTX 5060 |
| Android A0.1 probability head | supported | exact parity on 1,716 rows; SM-S9280 P95 0.001615 ms; heavy inference not covered |
| Android metric-depth dual arm | executes, not real-time | ORT CPU P95 Metric3D 5367.79 ms versus UniDepth 1721.97 ms; NNAPI registration slowed both |
| Candidate-only CLI | smoke pass | 30/30 RGB frames, JSONL and MP4, evaluator-consistent probabilities |

The strongest fresh A0.1 result on 1,716 known opportunities was:

- Brier reduction versus deterministic clearance: 40.45%;
- log-loss reduction: 71.97%;
- ECE: 0.02909;
- high-confidence-clear coverage: 14.92%;
- high-confidence-clear false-clear: 4.30%;
- occupied recall at `P>=0.50`: 88.43%.

This is same-dataset-family TUM Development evidence. It is not external-camera,
user-utility, reminder, or safety evidence.

## What is runnable now

- Deterministic seven-window public contact sheet:
  `render_motion_occupancy_a0_demo.py`.
- Any calibrated 10 FPS RGB video:
  `prepare_external_rgb_video_manifest.py` followed by
  `run_motion_occupancy_a0_candidate.py`.
- Candidate outputs: current left/centre/right occupancy probabilities at 1.0,
  1.5, and 2.0 m, UNKNOWN propagation, JSONL, summary, and annotated MP4.

The unsupported A2 future model is not present in the candidate CLI.

## Mainline promotion boundary

Do not compare this route with the traditional mainline by model novelty or one
TUM score. It becomes eligible to replace the mainline only after all are true:

1. final external camera intrinsics are calibrated;
2. two controlled sessions cover static obstacles, lateral passage, approach,
   wearer motion, rotation, blur, and low light with independent geometry truth;
3. current-occupancy calibration, false-clear, recall, coverage, and UNKNOWN
   behavior transfer without fitting or threshold changes;
4. actual target-device latency, memory, power, and thermal behavior are
   measured;
5. under the same event ledger, the field improves user-facing error/coverage
   trade-offs over the current mainline.

Until then it remains an independent, promising side lane. Final-camera data
is still required for quality transfer. While that capture is unavailable, the
only useful next branch is a separately frozen Metric3D compression or
teacher-to-student deployment screen using consumed inputs, with quality
retention checked before another phone run—not another TUM feature or threshold
search.
