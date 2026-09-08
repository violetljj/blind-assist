# HEAD-P1 conditional matched fits

EXPLORE. Authorized continuation of HEAD_PAIRED_20260908.md. This protocol is
frozen before optimizer steps. The completed primary diagnostic triggered:
L damages3/16 units across TRAIN/EVAL and three families; M damages0/16.
Result SHA256:48a167394842477129a7234c494d70fec4aaf03d867f3e063255ab627422c744.
All64 native appearance comparisons passed, including UNKNOWN correspondence.

## Matched intervention

Start two fresh fits from the original G13 seed17 checkpoint, not Coverage1198.
Both use the original model/runtime, all parameters trainable, frozen BN running
buffers, AdamW lr1e-5/weight_decay1e-4, seed17, batch32 and final2000 steps.
Both consume the same saved sequence:20 uniform replacement draws from original
1198 TRAIN plus two distinct uniformly drawn TRAIN units, each all six frames
(CLEAR/HEAD_ONLY by A/L/M). There are eight paired TRAIN units/48 images. No
diagnostic EVAL fitting. The16 A parent geometries in TRAIN were already exposed;
the static rerenders do not create new independent geometry. Fixed2000-step
exposure differs from original Coverage1198, so only the two new arms isolate
the consistency penalty; comparison with Coverage is contextual.

ERM uses original near BCE plus .25 class-balanced known-pixel support BCE.
CONSISTENCY adds .1 times (near probability MSE plus support probability MSE).
Pairs are A-L and A-M within the same unit/relation. Support MSE balances
positive/negative classes per pair/head before averaging active pair/heads.
Common-known pixels only; UNKNOWN contributes zero gradient. No features,
cross-relation matching, MIL, Dice, architecture or optimizer-policy change.

## Selection, outputs and interpretation

Finish both fits before final evaluation. Calibrate each final checkpoint only
on original DEV128 using the original per-head empirical FPR<=.1 selector.
Evaluate original TRAIN1198, DEV128, original coverage EVAL64, and all diagnostic
TRAIN/EVAL appearances. Retain raw near/support logits and probabilities,
pixel recall/false activation, signed relation ordering and alert counts.
Save actual exposure, checkpoint/optimizer/RNG, source/input hashes and receipts.

Compare CONSISTENCY against matched ERM first. Reduced appearance deltas alone
are insufficient: require relation correctness and useful recall with false
alarms visible. Report all original EVAL64 and paired EVAL counts, including
regressions. No aggregate model promotion rule is introduced after outcomes;
one-seed small consumed same-world comparison supplies descriptive evidence.
No claim of independent-world generalization, statistical significance or App
readiness. No plaza reopening, extra seeds, alternate weights or rescue fits.
At fixed budget completion, report the result and close this experiment.

Source note: native masks are exactly invariant under background WPO freeze,
but an A/A four-frame probe retains temporal RGB noise (~2.1–2.3/255 native
mean absolute difference). M changes can approach this scale. A/A inference,
if reported, is a supplementary noise reference and does not replace any
predeclared diagnostic sample, trigger or fitted model selection.
