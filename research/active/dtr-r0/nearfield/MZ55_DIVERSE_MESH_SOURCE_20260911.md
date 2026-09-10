# MZ55 diverse rigid meshes — source only

2026-09-11. One bounded Development source acquisition; no fitting, model
inference, threshold change or fresh-site claim. Registration and root review
of the frozen manifest must precede any UE launch. MZ48 remains unchanged.

Use the three measured native CitySample assets in the bound design at
`work/mz52-diverse-source-design-20260911`: shallow_awning, sign_panel,
square_grille; retain the original oblique rod as retained_rod. Preserve mesh
material slots, opaque source geometry and the original rod material; do not
substitute assets or override materials after seeing renders. Native material,
back-face and opening agreement are canary admission questions, not established
sensor realism. The grille is a rotated tree prop, not natural wall hardware.

The exact matrix is 4 families × 4 relations × 2 front-anchor ranges ×
2 support contexts × 8 consumed sites × 5 profiles = 2560 unique frames:
1920 new-mesh and 640 rod controls. Preserve the eight MZ48 camera/floor poses.
Five TRAIN_CANDIDATE sites provide1600 frames; site06-002 is CALIBRATION320;
05-002 and06-004 are HELDOUT_SITE640. Pairs never cross roles. All are controlled
Development and share previously consumed backgrounds.

Profiles use the bound design's five scales, yaw offsets, lateral offsets and
near/far anchors. Awning/sign local yaw is90° plus profile offset, grille pitch
90° plus yaw offset. Front anchor means the camera-forward AABB minimum;
lateral and height anchors mean AABB centers. BODY center1.19m, HEAD1.625m,
BOTH one copy at each height with second front+0.05m; VISIBLE_NONINTRUDING
center2.10m. Rod shape/material/relation construction is unchanged from MZ48;
only five fixed position profiles differ. The same twelve-part lateral rig is
added for supported cases in every family/relation; target dictionaries remain
identical within each context pair.

Distance is an anchor stratum, not exclusive near/far truth. Actual native
four-query events use the unchanged full-image native contract and >=3 pixels.
Intent collapses BODY_NEAR|BODY_FAR and HEAD_NEAR|HEAD_FAR. Preserve additional
range bits, intent mismatches, source failures and UNKNOWN; no failure-driven
geometric replacement or relabeling. Native-depth labels never enter prediction.

The first64 frames count within2560. For site i and family j choose relation
(i+j)%4, range(i+j)%2, profile(i+2j)%5, both contexts. Two32-frame region shards
precede eight312-frame site shards. Canary requires all64 source/render/floor
and actual mesh placement checks,32 equal target dictionaries and native query
mask/depth pairs, at least56/64 collapsed-intent matches, and each new family
× each of four relations correctly represented. Inspect every canary RGB/native
pair; explicitly check grille openings, awning underside/back faces, sign-face
depth, visibility and native material plausibility. Missing/failed visual review
or any hard source gate stops expansion. Preserve the failed canary as evidence;
do not override materials, retune geometry or expand a replacement budget.

Predeclare80 native audit rows: all64 canary plus one additional context pair
per site (awning, same canary relation/range, next profile modulo5). Retain all
raw RGB/native on the capture owner; return unchanged RGB bytes in compact
training.source ZIPs, observable original45° packet, four native labels,
contributor labels and small supervision. Only those80 raw native arrays return.
After source admission, MZ52's native-only full-frame8×8-count recipe can be
applied on each owner; it does not expand predictor ToF coverage.

Initial capture is worker-only with the frozen MZ48 runtime/plugin and its warm
DDC. Primary may take only explicitly handed-off unstarted site shards after
MZ54 ends and its GPU is free. Bind runtime/plugin/spec/mesh/material hashes
before rendering. Never change case payloads for host placement; only map_file
path may be remapped with equality and same map SHA proof. Each shard has one
durable attempt, timeout2400s, exclusive process ownership and UE/Zen release
verification. Do not save or mutate the existing project/assets.

Budget:2560 captures maximum, zero model/training steps. MZ48 measured raw
4.74GB and compact1.10GB for the same frame count; reserve12GiB on the worker
for raw/package/logs and1.5GiB controller returns, excluding existing shared
engine/DDC. These are planning estimates, not MZ55 throughput measurements.
64-frame canary disk share is roughly120MB raw; compilation/DDC overhead can
dominate. Actual bytes/time and resource release receipts decide completion.

Prepared manifest and per-shard specs live under
`work/mz55-diverse-mesh-source-20260911/spec-v1/`. Source helpers are
`mz55_prepare.py`, `mz55_dataset.py` and `mz55_canary_gate.py`; the task-owned
worker command checks registration/admission locks and frozen hashes before UE.
Root owns registration, admission review, source index and final delivery.
