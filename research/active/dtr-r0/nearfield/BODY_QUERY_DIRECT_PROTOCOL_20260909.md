# Frozen direct readout of attribution probes

2026-09-09 EXPLORE. Zero fitting/inference; reuse saved B/R1/XYZ ray/local probe
logits and frozen point DEV cutoffs. Existing consumed320-frame source only.

Exactly four query statistics: max probability, valid-point mean probability,
top3 mean probability, and number of valid samples above the saved point cutoff.
Divide the last by27 only to express it in[0,1]; it is not a native pixel count.
Sigmoid is applied before probability aggregation. Mask only fixed projection
FOV validity; never use native UNKNOWN, ownership or local-membership GT masks.

For every method, frame BODY/HEAD score is maximum over that head's six query
statistics, including both distance bins. This fixed OR-like score is not a
probabilistic count model and is never passed into near_from_counts. Evaluate
all24 combinations (two target definitions,three feature sources,four statistics).
Use original DEV frame selector FPR<=.10/min_count8 separately for both heads.
Persist cutoffs before EVAL reporting. Within each target/features choose one
whole readout on DEV: maximum minimum-head recall, then macro recall, lower
macro FPR, then max/mean/top3/count order. No EVAL winner selection or new sweep.

Report all combinations, baseline B/R1 final near outputs, query AUC, full-frame
TP/FP/AUC/groups/condition errors, and matched HEAD_ONLY-CLEAR/BOTH-BODY_ONLY
score differences on both frames and fixed HEAD query positions. XYZ differences
must be exactly zero. Include descriptive frame TP at EVAL FP<=2, never a new
deployment cutoff. No abstention filtering or hidden-frame exclusion.

Stop after this comparison. If no useful relation/alert tradeoff emerges, retain
B and stop these fixed direct-readout replacements; do not tune k or fusion.
Any improvement is consumed Development evidence, not solved transfer. CPU is
TASK_NOT_GPU_SUITABLE for this small cached scalar/vector scoring task.
