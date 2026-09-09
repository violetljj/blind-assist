# Range supervision R0

2026-09-09 EXPLORE. Preserve expanded B as the working alert baseline.
Cached TRAIN/DEV diagnostics (no EVAL access or fitting in that diagnostic)
show HEAD-near recall at within-split FPR<=10% of62/500 and61/200. HEAD-far
fires at .5 on497/500 TRAIN near-only positives and182/200 DEV near-only
positives. Threshold calibration alone is not established as sufficient.
The initial diagnostic was exploratory, not preregistered confirmation.

Run one new candidate from the same G13 seed17 initial tensors, exact saved
expanded-B sampling schedule and2000-step batch32 budget. All model parameters,
BN policy, original losses, optimizer and data roles remain unchanged. Add
0.25 times mean BCE on four range-event logits: BODY/HEAD x near/far. For each
range the event is that its three independently predicted capped counts sum
to>=3. Truth is the same deterministic event on the existing capped counts.
This is a reorganization of existing truth, not new depth input or parameters.
Do not continue from the final expanded-B checkpoint or sweep weights.

Baseline checkpoint SHA256:
c7aef143bcc7ac464097e95219dc5239523df7ce5716fc95926f37592f94e776.
Reuse accepted cache-v1 from body-query-expanded-b-20260909. Use the same new
DEV selector and persist both arms' near cutoffs before EVAL. Baseline must
reproduce its cached scores/cutoffs (floating roundoff tolerance only).

Report final-alert confusion/AUC/groups and range-event AUC/BCE, wrong-range
activation, query recall and condition errors. Calibrate four diagnostic range
cutoffs on DEV only; retain fixed.5 and within-split low-FP curves as diagnostics.
A candidate qualifies to replace the alert baseline only if EVAL HEAD TP>=565,
BODY TP>=585, HEAD FP<=34, BODY FP<=58 and complete groups>=228; additionally
HEAD-near query recall must improve by>=20 percentage points and erroneous
HEAD-far activation on native near-only frames at.5 must at least halve.
These are scoped engineering tolerances, not statistical significance.
If geometry improves but alerts fail this condition, retain a research
challenger without replacing the working baseline. If TRAIN range separation
does not improve, stop this recipe. One fit only, no automatic new geometry
capture, depth model, attention, extra steps or seeds.

All results remain shared-asset controlled Development with repeated geometry
designs. No reliable range display, physical free-space, safety or protected
confirmation claim. Preserve UNKNOWN and all frames in final alert metrics.
