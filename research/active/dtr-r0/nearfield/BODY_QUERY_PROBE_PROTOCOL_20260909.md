# Frozen B/R1 linear attribution probe

2026-09-09 EXPLORE. No main-model optimization, new capture or R1 weight sweep.
Reuse B Q2 point features; extract R1 pooling-before32-D features once on320 RGB
frames and reproduce cached near scores. Main checkpoint hashes remain frozen.

Probe two explicit label definitions, with identical labels for both models:
1. Native nearest-ray OWN_CELL versus known visible OTHER; invalid rays ignored.
2. Local footprint membership >=0.5 versus <0.5, using Q3 bilinear20x20 cell
membership mass. Ignore out-of-FOV and any footprint touching native UNKNOWN.
The second is a coarse majority-footprint target, not exact ray truth or the
network's true receptive field. No FREE class: absence/no return is not free.

For each target fit B32-D, R1 32-D, and XYZ-only3-D control, six fixed probes.
TRAIN-only feature standardization, zero initial weights/bias, full-batch Adam
lr0.03 for1000 steps, unweighted binary CE plus1e-4 mean squared weight penalty.
No hyperparameter search or model selection. Record TRAIN objective endpoints;
poor fitting cannot prove feature information absent. Only probe weights update.

Report rank AUC, class denominators, fixed0.5 decisions and original-policy-style
DEV-selected point FPR<=0.05 cutoff, then frozen DEV cutoff on EVAL. Report all,
BODY/HEAD, near/far and condition strata; null AUC when a stratum has one class.
Save selected cutoffs before EVAL reporting. Point errors are correlated, not
independent frames; point FPR is not final alert FPR. XYZ control checks whether
query geometry priors alone explain apparent separability.

Decision concerns linear readability of these targets under this protocol only.
High point AUC does not prove a count readout will work; weak linear probes do
not prove the backbone lacks information or causally identify shortcuts. Do not
turn27 duplicated/correlated samples into native pixel counts. One-source consumed
Development remains consumed; no R2 main fit or promotion follows automatically.
