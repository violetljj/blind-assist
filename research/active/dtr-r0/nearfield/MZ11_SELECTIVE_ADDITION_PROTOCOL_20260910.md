# MZ11 selective addition: one bounded Development candidate

Question: can frozen local SOURCE evidence plus observable angular candidate
extent recover missed obstacles without adding false positives to the complete
MZ5 output? MZ10 showed availability alone fails and score-only threshold repair
loses far TPs. Its post-hoc audit suggested extent, while a baseline-confidence
veto would reject the intended thin pole.

Keep MZ5 positive outputs unchanged. Only where MZ5 is negative, MZ9 SOURCE has
geometric support and its frozen calibrated margin is nonnegative, allow one
small gate to add a positive. No negative SOURCE sentinel means CLEAR. This
design cannot remove any existing MZ5 false positive, including premature near
warnings; report that ceiling explicitly.

Gate: four independent logistic readouts, each with four inputs and bias (20
parameters). Inputs are SOURCE margin/4, log1p angular hypothesis count/6,
distinct candidate zones/8, and distinct candidate returns/8. These counts are
possible supports, not measured occupancy. No baseline confidence, native depth,
target identity, truth, pose, or temporal information enters the gate.

One fit: seed111, 600 full-batch AdamW steps, learning rate .03, weight decay .001.
Only old TRAIN2500 and consumed MZ6 training-regression200 labels enter fitting.
Mean BCE is balanced across positive/negative eligible examples separately for
each query and each cohort, then averaged over nonempty groups. Frozen MZ5,
SOURCE, visual encoder and source thresholds never change. No architecture,
seed, step or loss sweep and no continuation after the stop gate.

Old DEV1000 chooses a per-query gate threshold strictly above the highest score
of a false addition candidate. This enforces zero additional DEV FPs for the
whole composition, whose original FP budget is already occupied. All DEV
selection is disclosed; it is not independent evidence. If there are no eligible
DEV examples use +infinity; if no negative examples use -infinity. Save exact
thresholds and the zero-add baseline. Source score and sign are replayed from
frozen MZ10/MZ9 where available; extraction uses original 32-frame query batches.

Report original MZ5, MZ10, MZ9 SOURCE and equal-new-data MZ5 adaptation alongside
the candidate on old DEV and clean/stressed MZ6. MZ6 is training regression, not
confirmation. Gate for a subsequent experiment: no per-query FP increase in DEV,
clean or stress; no original positive-bit losses; strictly improved DEV far TP
sum; clean and stress thin far TP >=48/49. Reject a zero-add solution as no gain.
Run the frozen four-column wrong RGB correspondence control on the sequence;
to attribute thin recovery to correct correspondence it must recover fewer thin
bits than the correct version. This tests the complete frozen chain, not the
unique contribution of gate extent features. No causal extent claim without an
additional matched ablation; no such ablation is budgeted here.

At completion stop this experiment, register its disposition, and deliver the
result. No new collection is in this turn's budget. A pass only nominates a
fixed candidate for a separately scoped new-placement confirmation; a failure
retains MZ5 and the MZ9 component. Temporal work and App promotion remain closed.
Do not report 49 adjacent thin opportunities as independent obstacle events.

GPU for frozen extraction and gate fitting; CPU for scalar metrics. Durable
inputs/checkpoints/receipts stay in artifacts.local. Mechanical failures get
separate attempt records and do not change the frozen scientific recipe.
