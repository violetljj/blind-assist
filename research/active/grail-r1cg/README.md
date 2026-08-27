# GRAIL R1C-G1 Active Multiview Appearance

Status: `G1_GPU_PROTOCOL_FROZEN / DEVELOPMENT_RUNNING / NO_FINAL_TEST`

R1C-G1 asks whether a short fixed three-view reference scan reveals asymmetric
RGB/mask appearance that a single reference image hides. It is a new
information-source experiment, not a rescue of G0 camera-yaw transport.

## Frozen comparison

Both arms use the same fresh train and Development samples, pinned DINOv2-S
weights, shared pair-encoder architecture, direct two-class permutation head,
training budget, seeds, sampling, and checkpoint rule:

- `B1 single`: anchor reference plus query;
- `G1 triplet`: anchor, left, and right references plus the same query.

Each reference/query pair produces an evidence vector. Both arms aggregate with
`concatenate(mean, max) -> MLP`; for B1, mean and max are the same single
evidence vector. Thus the model parameterization is unchanged and only the
number of reference observations differs.

The fixed rosters contain 96 training houses and 24 Development houses. They are
house-disjoint from one another and exclude all 180 R1C-L train/validation
houses plus all 24 G0 Development houses. The protected R1C-L final data is not
opened.

Pixel collection uses AI2-THOR `Linux64` through WSLg Mesa D3D12 on the local
NVIDIA GPU. The
initial Linux64/Xvfb startup attempt was interrupted before merge, training, or
evaluation after CPU contention produced a Unity timeout; it is not G1 evidence.
The roster, model inputs, gate, and all scientific conditions were unchanged.

## Scan leakage boundary

The old R1C-L `_ranked_positions` route is forbidden because it uses owner yaw.
G1 instead chooses anchors without owner orientation, then locates side frames
on the anchor-camera lateral axis. Each side must have 0.20--0.45 m signed
lateral translation and at most 0.20 m longitudinal drift. A triplet missing
either side is `NOT_EVALUABLE`; copying the anchor is forbidden.

Simulation metadata may identify the physical owner group, keep the object in
view, and score the permutation. Owner yaw and canonical sign do not choose the
scan. The model receives only RGB, owner-union masks, and sibling-centroid masks;
camera yaw, owner yaw, depth, object coordinates, scan geometry, and view-role
labels are not inputs. The triplet aggregator is permutation-invariant.

## Development gate

Primary evidence is balanced accuracy on discriminative PRESERVE/FLIP samples.
The report also includes class accuracy, rescue, collateral, Drawer, Doorway,
ambiguous coverage, and owner-group macro balanced accuracy.

G1 advances only if all frozen conditions hold:

- balanced-accuracy uplift over matched B1 is at least `+8pp` in both seeds;
- rescue exceeds collateral in both seeds;
- G1 loses no more than `5pp` PRESERVE accuracy in either seed;
- mean balanced-accuracy uplift is positive for both Drawer and Doorway.

No subjective "obvious effect" escape clause, NBV policy, pose regression, G0
fusion, backbone/loss/threshold sweep, or final test is authorized.

The frozen roster and exact protocol are in
`grail_r1c_g1_manifest_v1.json`. Development outcomes will be added only after
the complete frozen run terminates.

## Preserved prior terminal

G0 remains `STOP_G0_POSE_TRANSPORT / DEVELOPMENT_GATE_NOT_MET / NO_FINAL_TEST`.
Its exact manifest and result remain in `grail_r1c_g0_manifest_v1.json` and
`grail_r1c_g0_development_result_v1.json`.
