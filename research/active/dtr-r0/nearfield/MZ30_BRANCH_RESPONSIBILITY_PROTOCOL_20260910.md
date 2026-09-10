# MZ30: choose a responsible sensor branch for unsupported baseline alarms

2026-09-10 EXPLORE, consumed Development. MZ29 locates80/81 placement baseline
false bits outside geometric support, alongside48 correct placement BODY_NEAR
bits and28 correct oldDEV BODY_NEAR bits. Missing support cannot justify a veto.
MZ22 did not train or change this branch. Test whether RGB/ToF disagreement
provides an observable correction path without deleting agreements or additions.

First reconstruct frozen MZ5 RGB and ToF logits/hidden128 vectors from existing
full-image feature maps, fixed query projection/context decoder and sensor
packets. No RGB backbone inference or source collection. Reproduce MZ5 baseline
logits/signs on all required TRAIN and normal DEV/clean/stress rows. Only the
original TRAIN IDs and normal consumed cohorts are read; no protected EVAL.

Fixed pre-fit falsifier: among81 placement baselineFP, at least41 must be both
geometrically unsupported and RGB/ToF-sign-disagreeing. Also require at least20
TRAIN eligible query examples with RGB correct and20 with ToF correct. If either
fails, stop with diagnostic evidence and no fit; branch selection cannot address
enough of the named error surface. Do not lower the criterion after inspection.

If the gate passes, fit one shared branch selector on eligible TRAIN queries:
baseline>=0, no original MZ20 geometric support, RGB/ToF signs disagree. Target1
means RGB sign matches the dataset event truth; otherwise ToF is correct. Inputs
are frozen RGB hidden128, ToF hidden128, eight branch logits, and query one-hot4
(268 total). TRAIN-frame-only normalization, std floor0.1, computed over unique
TRAIN frames; query one-hot remains unnormalized. MLP268->32ReLU->1, seed123,
Adam0.001, original1200x16 MZ20 batch stream, inverse-frequency weighted binary CE
for the two TRAIN eligible target classes. Both sensor branches remain frozen.
No seed/architecture/step/weight sweep; record in-sample base-training limitation.

At runtime the selector estimates which branch to trust. An eligible baseline
alarm can only be removed when the selector favors the existing negative branch
with calibrated confidence. All other MZ28 decisions remain exactly unchanged.
This is branch selection, not invented local evidence. Use oldDEV only to choose
one confidence cutoff per query: maximize false alarms removed with zero correct
baseline alarms removed; consider cutoffs>=0.5, break ties by fewer total switches
then higher cutoff; disabling a query is allowed. Freeze that cutoff elsewhere.
Task-negative output remains a research event decision, never a safety/CLEAR claim.

Evaluate all normal cohorts plus existing wrong-visual-zone controls using frozen
MZ28 decisions as starting point. Global RGB/ToF branch inputs do not change in
those local-visual controls. Report exact frames, totalFP removed/new, TP retained/
lost, positive additions, per-query/group/site and trained pole49. A useful
replacement component requires at least41/81 placement baselineFP removed, zero
newFP, no baselineTP loss on normal cohorts, unchanged MZ28 far additions and
pole>=48/49 clean/stress. Report raw tradeoffs if any criterion fails; no promotion
or rescue. Calibrated oldDEV retention does not establish transfer retention.

Save branch inputs/logits, masks, labels, frozen hashes, gate results, original
batches, normalization, initial/final selector, calibrated cutoffs and task arrays.
Check reconstruction, target routing, calibration optimality and scalar outcomes
independently. Numerical/mechanical failures preserve outputs; no scientific
budget extension. Complete report/terminal/scoped delivery and release resources
after the gate or fixed fit. No App, device, temporal, independent-source or safety
claim. Output: artifacts.local/work/mz30-branch-responsibility-20260910/run-v1/.
