# Current decision: GRAIL active multiview appearance

Status: `G1_GPU_PROTOCOL_FROZEN / DEVELOPMENT_RUNNING / NO_FINAL_TEST`

## Authorized question

Can a fixed anchor-plus-left-plus-right reference scan reveal symmetry-breaking
RGB/mask appearance that materially improves direct owner-canonical
PRESERVE/FLIP prediction over a matched single-anchor model?

This is a fresh synthetic ProcTHOR Development experiment. It tests added visual
observability, not camera-pose transport or a next-best-view policy.

## Frozen surface

- Protocol and code: `research/active/grail-r1cg/`
- Source: the pinned ProcTHOR-10K train revision used by R1C-L and G0
- Renderer: AI2-THOR `Linux64` through WSLg Mesa D3D12 on the local NVIDIA GPU
- Rosters: 96 train houses plus 24 Development houses
- Exclusions: all 180 R1C-L train/validation houses and all 24 G0 houses
- B1 input: anchor RGB/masks plus query RGB/masks
- G1 input: anchor/left/right RGB/masks plus the same query RGB/masks
- Model: identical shared pair encoder and `mean+max -> MLP` direct mode head
- Seeds: 1701 and 2701
- Training: discriminative samples, balanced mode/object sampler, eight epochs
- Primary metric: discriminative balanced accuracy

The scan is generated on the anchor-camera lateral axis, never an owner or
canonical axis. Each side requires 0.20--0.45 m lateral displacement and at most
0.20 m longitudinal drift. Missing side frames make the scan not evaluable;
anchor duplication is forbidden.

## Frozen advancement gate

All conditions must hold:

- at least `+8pp` balanced-accuracy uplift over B1 in each seed;
- rescue greater than collateral in each seed;
- no more than `5pp` PRESERVE-accuracy loss in either seed;
- positive mean balanced-accuracy uplift for both Drawer and Doorway.

No result-dependent alternative gate is available. Camera/owner pose, depth,
object coordinates, scan geometry, and canonical sign are not model inputs. No
NBV, pose head, G0 fusion, backbone/loss/threshold sweep, or final test is open.

## Current state

The protocol and fresh house-disjoint rosters were frozen before collection or
model outcome. A Linux64/Xvfb startup attempt was interrupted before merge,
training, or evaluation after CPU contention caused a Unity timeout. The GPU
renderer amendment changes runtime only; roster, pixels consumed by both arms,
inputs, model, seeds, metrics, and gate remain matched. Development collection
and matched two-arm training are running. No outcome claim is available yet.

## Preserved prior terminals

- G0 remains
  `STOP_G0_POSE_TRANSPORT / DEVELOPMENT_GATE_NOT_MET / NO_FINAL_TEST`.
- R1C-L remains
  `STOP_R1C_L_WITHOUT_FINAL_TEST / DEVELOPMENT_GATE_NOT_MET / FINAL_UNOPENED`.
- The unseen-location Router remains
  `MSLS_SOURCE_ADMITTED / ROUTER_DEVELOPMENT_GATE_NOT_MET / TEST_UNOPENED`.

None of those consumed cohorts is reopened, tuned, rerun, or fused into G1.
