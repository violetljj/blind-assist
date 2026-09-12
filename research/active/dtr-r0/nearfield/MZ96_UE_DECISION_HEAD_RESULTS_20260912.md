# MZ96: the small tree suppresses false alerts but does not beat strong rules

Decision: `FRESH_UE_TREE_GATE_NOT_MET`. Keep the learned head as a negative
control for full risk-decision replacement. Horizon A and A+B remain frozen
Development challengers; retained matched_hold/default application is unchanged.

The [protocol](MZ96_UE_DECISION_HEAD_20260912.md) compares all methods on one fresh
UE source:128 independent scene instances,40 frames each,80/16/32 scenes assigned
to train/validation/test before collection. This is5,120 new frames, not the
consumed1,920 MZ90 frames. Families are shared across splits: no unseen-family,
real-hardware or physical-body-collision claim follows.

## Held-out scene-instance results

Test:32 scenes,1,280 frames,409 positive frames,117 future-only positive frames.
A+B was selected as reference using validation only (F1 .8559); the learned
threshold0.55 was selected using validation only. All test predictions, model,
selection and local implementation hashes were sealed before test scoring.

| Method | TP | FP | FN | F1 | Future-only recall | False segments | Fragments | Missed events |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| matched_hold | 306 | 200 | 103 | .6689 | 41/117 =35.0% | 48 | 35 | 0 |
| range-only3.6 | 321 | 234 | 88 | .6660 | 53/117 =45.3% | 54 | 38 | 0 |
| horizon A | 342 | 176 | 67 | .7379 | 70/117 =59.8% | 47 | 39 | 0 |
| coverage B | 320 | 233 | 89 | .6653 | 46/117 =39.3% | 39 | 28 | 0 |
| A+B | 368 | 205 | 41 | .7495 | 86/117 =73.5% | 39 | 27 | 0 |
| XGBoost | 297 | 84 | 112 | .7519 | 54/117 =46.2% | 19 | 23 | 2 |

Relative to A+B, the tree reduces FP by121 but also loses71 net TP. F1 improves
only0.0024, below the predeclared0.02 gain. It misses2 events the reference detects;
maximum added delay among jointly detected events is0.9s, above0.2s. It passes
FP/false-segment/fragment gates and fails F1/TP/event/delay gates. Paired episode
bootstrap95% interval for F1 difference is[-.0761,.0982], not evidence of a stable
advantage. There was one fixed model configuration and no test-driven retry.
Fewer fragments do not independently prove better continuity when complete
events disappear.

Horizon A improves both TP and FP versus matched_hold on this new panel, whereas
range-only does not improve F1. That supports retaining A as a challenger, but
its fragments increase35 to39 and its earlier consumed proxy gate failed.
A+B improves recall further while FP rises176 to205 relative to A. Neither is
automatically a robust overall replacement. Standalone B again loses F1.

## What the learned head used and where the tradeoff appears

The34 observable features include causal range/bearing/Doppler, geometry and
coverage indicators, track length and1–3-frame support history. No episode ID,
family, split, source identity, true velocity, ghost flag or future packet enters
the head. One300-tree depth3 XGBoost classifier is followed by frozen2-of-3 /
2-empty hysteresis. Unlike rule arms, it has no sensor-priority outer hold; this
is the predefined end-to-end readout comparison, not an isolated tree ablation.

| Test slice / method | TP | FP | FN | F1 |
|---|---:|---:|---:|---:|
| ghost-present scenes, A+B | 95 | 71 | 6 | .7116 |
| ghost-present scenes, tree | 69 | 1 | 32 | .8070 |
| ghost-absent scenes, A+B | 273 | 134 | 35 | .7636 |
| ghost-absent scenes, tree | 228 | 83 | 80 | .7367 |

Ghost-present denotes the whole scene, not attribution of each FP to a ghost.
The impressive slice F1 therefore does not prove ghost identification; it includes
26 fewer TP. Mean absolute TreeSHAP ranks ToF support age, horizon-support count,
yaw increment, direct ToF support and nearest Radar radial speed highest. These
are associations, not causal mechanism proof; synthetic motion correlations may
matter. No post-test threshold or feature adjustment was made.

The common no-ToF/no-alert marker yields UNKNOWN640 for matched_hold,626 for A+B,
764 for the tree; positive UNKNOWN67/31/80 respectively. This bookkeeping marker
is not calibrated abstention or proof of free space. F1 above includes all frames.

## Provenance, execution and limits

UE5.8 native collision rays and transforms supply geometry. ToF uses8x8 rays
aggregated to8 horizontal bins; Radar uses actor-directed native visibility rays
with hypothetical quantization/noise/dropout and persistent synthetic ghosts.
GT is horizontal point-center route occupancy in[0,1]s, including t=0, within
3.6m. It is not actual Radar RF, material optics, rendered-RGB perception,
height-specific BODY/HEAD recovery or finite human swept-volume collision.

Capture source frozen at `87152040`; feature/runner freeze at
`7d5e9f1cab5a9384169adf43a29ccb930a5257c7`. Spec SHA256:
`a87173e43287eb6f5cb18cf3adf101337d6b7a52a7817a198734faab3266fb06`.
Native collection recorded30,398 ToF ray hits and6,409 Radar occlusion hits.
Full capture completed in25.625s wall time (7.281s native work); editor/Zen
release receipt confirms task-owned processes exited. The earlier80-frame
engineering canary used a different seed and was excluded.

XGBoost3.2.0 was installed only under the artifact root. Equivalent fixed train
fits measured CPU0.194s and actual CUDA0.342s, one fit each, no predictive
selection. The CPU model was retained as `CPU_FASTER_MEASURED`. These are host
training measurements, not edge inference latency. Comparison took3.039s.
Five capture tests and one feature test passed; feature checks include causal
prefix and Radar slot permutation. Readonly review added exact source/split
checks, dependency sealing and future-only denominators before execution.

Evidence root: `artifacts.local/work/mz96-ue-decision-20260912/`.
Keep source-v3, capture-v1 streams/receipt/process-release, comparison-v1 split
payloads/model/predictions/selection/seal/results/TreeSHAP/backend/receipt,
engineering canary logs and local dependency directory. All are task-owned
durable reproducibility evidence; no running collection/training or paid worker
remains. Capture raw SHA256:
`2516b822c40e3f59f01b0084894a72656cd47aa3b04ec07780fc9fa30137a80e`.

This bounded attempt is complete. The useful next decision would concern how to
retain future-risk recall while reducing false alerts; this run does not justify
claiming that a general learned fusion boundary is established or automatically
starting another fit. A new experiment would need an explicit changed hypothesis
and a new evaluation boundary; the present test is now consumed Development.
