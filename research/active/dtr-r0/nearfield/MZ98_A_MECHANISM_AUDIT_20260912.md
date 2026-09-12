# MZ98: consumed A admission decomposition, not a policy search

EXPLORE diagnostic only on sealed MZ96/MZ97 test payloads,1280frames each.
Both are consumed Development. No new collection/training/tuning, no changes to
frozen A, no new test or promotion claim. Verify all original receipt/seal hashes
and exact reproduction of baseline/A before scoring diagnostic exclusions.

Question: how much of A's final effect requires raw current occupancy, fitted
current correction, or actual later corridor entry? Return-level exclusive labels:
R = A's first raw-current branch; F = raw fails, fitted admission passes and fitted
position is inside at t=0; P = raw fails, fitted admission passes and fitted
position starts outside and enters later. Split R at3.18m. Record current legacy
range,20deg bearing and-.35 Doppler gate failures as overlapping flags, not
exclusive causal categories. Shared tracking/history remain identical.

Fixed diagnostic replay controls only: baseline; full A; R only; R|F; R with
legacy range; R with legacy Doppler; R with both. Every replay uses unchanged
Radar hysteresis, ToF selection and one-frame outer hold. This finite set is
declared before diagnostic scoring and is not a threshold/algorithm search.
Full minus R|F isolates dependence on strict future admission in this implementation;
R|F minus R shows dependence on fitted-current correction. Increments are
order-dependent with interactions, not independent additive causal effects.

Report TP/FP/FN/F1/future-only recall/events/segments/fragments for both panels,
paired added/lost TP/FP at each step, actual support branch mixtures, selected
sensor and current-vs-history for final added/lost/retained alerts. Trace changed
frame records and return branch records. For every FP path report corresponding
TP, not just error counts. State propagation may move differences to later frames;
same-frame branch counts are descriptive, not final-effect attribution.

Use native bounds only in evaluator-side diagnostics to split GT future_only into
current extended-wedge occupancy versus strict future entry. A current3.18–3.6m
target can be GT future_only without trajectory prediction. Sensor code lacks
translational ego velocity, so target/wearer contributions are not separately
identifiable. Gyro yaw compensation is shared with baseline. No fabricated source
identity for individual raw Radar returns: capture discarded slot provenance.
Ghost-present scene and family slices are context, not per-return cause labels.

Preserve outputs under artifacts.local/work/mz98-a-mechanism-audit-20260912/run-v1.
Scalar CPU audit, no paid resources. Freeze code before diagnostic outcomes;
stop after one audit and report/inheritance/scoped delivery. Any apparent simpler
variant remains consumed-data diagnostic evidence, not a selected successor.
