# Matched 3 m crossing observability diagnostic

2026-09-23 EXPLORE. The user authorized testing whether current/past RGB and
8x8 ToF can distinguish the 3 m crossing before a distance-admission branch.
Reuse complete consumed LOCAL stability and rescue-fresh captures; no new
capture, model fit, threshold selection or modification of old experiments.

Within each clip, verify identical objects/materials and camera y/z/angles.
Only camera x moves. Entry pairs are frames 2/3, exit pairs 9/10, oriented near
versus far using source geometry only in evaluation. Keep all 48 clips per
source: 64 intended crossing pairs and 32 OUTSIDE pairs. Entry pairs duplicate
across trajectories; report all192 recorded pairs and deduplicated144 geometry
pairs separately. Eight geometry groups per source, not independent frames.
Distances span near2.65--2.90m and far3.08--3.26m; not centimetre-level coverage.

Public readouts are unchanged global nearest (confounding control), nearest
query-compatible nominal zone, and median of three nearest compatible zones.
Compatibility uses public zone footprint at its measured distance with fixed
union X[-.3,.3], Y[-.2,.9]; it does not prefilter Z<=3. A readout supports a
nominal crossing when both endpoints are available and .3<=near<=3<far.
Missing returns never become8m, known free space or successful comparisons.
Full-zone possible compatibility is not measured return ownership.

For causal history, report current-minus-previous distances in the CURRENT
selected public zones where both frames are valid. No future sample, direction,
shape, object identity, target mask or frame index enters the readout. Geometry
and index only construct paired evaluation and reset histories. Past change
can describe approach/retreat but does not itself establish absolute range.

Keep saved observations plus16 deterministic resimulated noise realizations per
needed pose (pair endpoints and immediate predecessors). Use unchanged
simulate/sample_native; native depth is source construction, never a predictor
input. Original identities must reproduce saved ToF first. Seed identity keeps
the old shared trajectory noise-key semantics. Repeated draws are noise tests,
not new scenes. Save public readouts BEFORE joining target attribution/truth.

Evaluator-only native XYZ within rendered target_a/b cuboids attributes each
selected winning-bin contributor as target or other; background movement can
otherwise masquerade as useful target range. Never use this mask to select a
zone. Report ranking and correct crossing with/without target witnesses, all
missing endpoints, OUTSIDE false alerts, shape and entry/exit costs. Public RGB
whole-image paired L1 and same-pose dwell L1 are descriptive only: perspective
and backdrop changes do not prove a usable RGB range decoder.

A useful public range component needs >=80% joint crossing accuracy in BOTH
sources on saved and16-draw pooled deduplicated intended pairs, and <=10%
OUTSIDE endpoint false-alert rate on each. This is only a gate for further
range-admission research, not an App or system-performance gate. Any failure
retains partial signal and diagnoses limits without declaring all input
information absent. No cutoff rescue or automatic learner follows a failure.

CPU TASK_NOT_GPU_SUITABLE for fixed NumPy/JSON sensor simulation and scalar
statistics; use governed research-ue exact observation/evaluator roles, hashes
and receipts. Tests cover public footprint, missing-return handling, attribution,
pair deduplication and exact 3m semantics. Preserve A/LOCAL/UNKNOWN and all old
negative controls. Retain outputs and terminate task-owned work on completion.
