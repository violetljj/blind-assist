# Original FoundationStereo geometry challenge on historical stereo panels

2026-09-23 EXPLORE. User explicitly adopts this execution task: challenge retained
SGBM spatial evidence on all576MZ101/MZ102 pairs, with MZ104 TAO as a negative
reference and MZ105 local-score failure preserved. Not a current-A comparison.

One candidate: official NVlabs original23-51-11 ViT-Large, source6e8806816b533e4d13ddbb95ffa907b797060a62,
official Google Drive config/checkpoint linked by the project. Actual hashes are
frozen in configuration before inference. No TAO/small/Fast substitution, fitting,
checkpoint sweep or confidence-threshold rescue. Existing original source is
reused unchanged. Full640x360 RGB, official divisibility32padding/unpadding,
32iterations, original max_disp416 and mixed-precision CUDA; preserve full raw
disparity before applying unchanged visibility and0.5..4m metric-range masks.
Original70degree horizontalFoV and0.10m baseline give axialZ=fB/disparity.

Compatibility adaptations may isolate exact upstream helper definitions from
visualization-only imports, use the bundled DINO source instead of a mutable
torchhub download, and suppress redundant pretrained initialization only when
the full official checkpoint is loaded strictly. Record source/runtime/hash
details; no network-layer or learned-weight alteration. Official xFormers-disabled
PyTorch attention fallback is allowed. GPU is required. No torch downgrade or
shared-environment mutation. One memory-only low_memory=1 recovery is allowed
before any task-score access, if original low_memory=0 fails; preserve logs and
freeze the actually feasible configuration. Do not downscale or switch model
silently. If still unavailable, report engineering NOT_EVALUABLE.

Source construction creates a hardlinked RGB-only subset with exact image hashes
and a calibration-only manifest. The producer consumes those images, official
model/source/configuration and no scene/native/ToF/labels. Fixed canary IDs:
mz101/head_bar_textured_00, mz101/thin_left_flat_03, mz102/small_head_flat_00
(verify existence; if named ID absent choose its documented first frame, before
inference). Canary verifies dimensions, left/right metadata, padding restoration,
disparity units, finite outputs, backend and memory. It does not select on depth
accuracy or task scores. Then one full576pair replay with sealed original outputs.

Evaluation keeps exact historical commonFoV, known camera/body poses, BODY/HEAD
boxes, full64zoneToF and two-on/two-off rules. Reproduce historical SGBM+ToF raw
435TP/62FP/11FN and final402/40/44; report each panel separately. TAO history is
437/273/9raw,399/242/47final. All figures belong to consumed historical scope.
Seal candidate support/alerts before joining truth. No filtering by evaluator
depth and no favorable source exclusions.

Report four linked observations: far>4m surfaces estimated near; eligible near
surfaces outside the actual query estimated inside; true native-in-query pixel
coverage/position, including thin_left/right/small_head/occluded_thin; and final
events, false segments, first-correct times and processing cost. Attribute support
even on task TP, so an accidentally correct alert with entirely false geometry
is visible. Native query pixels remain the denominator for5/10/20cm Z accuracy,
missing predictions count as failures; conditional error and coverage are separate.
Native depth is evaluator-only, not independent physical truth or model input.

Decision is descriptive and tied to SGBM, not merely beating TAO: joint useful
geometry/task gains retain a Development challenger; partial spatial gains retain
the specific component for a same-depth structure comparison; persistent false
geometry or gains dominated by lost thin/small evidence reject this configuration.
Report all tradeoffs rather than inventing a complicated joint promotion gate.
No current-A/App promotion, fresh-layout confirmation or structure/fusion/state
change in this task. A later structure comparison may use useful imperfect output;
it need not wait for universally correct depth. No automatic threshold successor.

Run all cohort stages through governed research-ue. Retain acquisition/canary
failures, source/input/output seals, raw arrays, baseline parity and independent
evaluation checks. Release task-owned processes; keep resumable weights and
durable research evidence on canonical F:-backed artifacts.local. No paid
allocation unless already explicitly authorized; local existing CUDA is first.
