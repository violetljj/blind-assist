# MZ0: clean multi-zone information ladder on existing UE sources

2026-09-10 EXPLORE, user explicitly prioritizes algorithmic information gain over
sensor emulation. MultiZone-ToF-64 is a generic simulated8x8 metric source, not a
claim of reproducing VL53L8CX. No noise/dropout, training or checkpoint selection.

Use every EVAL_ONLY row (1500) of admitted body-query-5000 dataset-v1/index.json,
in source order. This is consumed controlled Development, not fresh confirmation.
Source provides BODY_ONLY,HEAD_ONLY,BOTH,CLEAR and low/above/lateral/far controls,
four fixture families and near/far placements. Do not invent strict paired
distance sequences, continuous trajectories or natural hard negatives from it.
Keep all rows; missing/corrupt inputs stop execution rather than silently exclude.

Freeze original10k B and current JOINT decoder, existing normalization and alert
thresholds, RGB256x144 preprocessing. A is JOINT four spatial events at0.5.
Save B alerts independently and preserve them exactly across all arms. B/C/D
spatial readouts may correct old range events; positive-only OR is deliberately
not used because it cannot remove wrong-far activations. This changes spatial
evidence only. No spatial absence is a clearance assertion.

B: same45x45degree crop compressed to one zone. C: same crop split into8x8 equal
angular bins. D: all native depth pixels in the same crop. An uncropped full-depth
reference diagnoses FoV loss separately, not64-zone compression. Depth is axial
camera-X; convert to radial range before packet output. Known finite native
depth0<d<100 and radial<=4m are eligible. Preserve validity and missing observations.

For B/C report ALL three readouts, with no winner selection: nearest valid radial
range, pixel median, and two-surface readout. The latter uses fixed0.1m radial bins
with>=3 native pixels, outputting mean range in the first/last supported bin.
This is clean geometric summarization, not a SPAD histogram. No target-conditioned
mask/identity, evaluator truth or native per-pixel coordinates cross the packet
boundary. Model-visible input is onlyzone directions, distance(s) and validity.

First zero-fit decoder projects returns on zone-center directions and queries the
four physical BODY/HEAD near/far volumes. A valid decoded point asserts an event.
D uses>=3 observed native pixels per physical range query. Both are gated by the
same frozen B BODY/HEAD category alerts. Also report depth-only outputs to show
whether gains merely come from geometry. This is an interpretable first readout,
NOT a mathematical upper bound on all learned methods using64 zones. D is a
privileged geometric reference, not evaluator truth fed to fusion.

Primary metric: exact four-event Spatial Exact Accuracy on all1500 frames.
Also report near/far and BODY/HEAD event-vector accuracy, per-event TP/FP/FN/TN,
near-only->far activations and BODY-only->HEAD/HEAD-only->BODY errors. Preserve
denominators and absent metrics; no temporal stability result from static frames.
The illustrative user gate is exact+5pp,wrong-far reduced>=50%,cross-body errors
reduced>=50%,original alert parity1500/1500. Zero baseline error stays zero.
Do not choose a new readout/threshold after seeing scores. Failure of this simple
decoder does not prove no information exists; use D/crop comparisons to identify
coverage or correspondence limits before making that broader conclusion.

Stop MZ0 after this single fixed inference/observation pass and independent checks.
MZ1 learned fusion and MZ2 corruption/resolution curves depend on the observed
information gain and remain distinct experiments.
