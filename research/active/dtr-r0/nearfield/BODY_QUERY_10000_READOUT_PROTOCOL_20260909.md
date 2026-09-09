# 10k B readout-only continuation

EXPLORE, consumed Development. Test whether the trained 10k B representation can
support better spatial count attribution by updating only its shared linear
query_readout. This is a 2000-step adaptation of trained 10k B, not the original
same-G13-initialization source comparison. The unchanged 10k B is the comparator.

Start both arms from SHA256 db39ccfcb9f3d1c8f3722711bab314ff0cf2da708e00d91f7fb263e6fec0ece0.
Freeze every parameter except query_readout.weight/bias and every buffer. Keep
TRAIN5000/DEV2000/EVAL3000, seed17, exact saved 2000x32 TRAIN schedule, AdamW
lr1e-5/weight_decay1e-4. Preserve near BCE + .25 support BCE + .25 count CE;
support is constant with respect to the trained layer. Record separate alert/count
gradient norms and cosine at step1 and each100 steps. Require initial prediction
parity against cached TRAIN before any optimizer step, and final exact frozen
parameter/buffer identity. Select final-step thresholds on DEV independently using
the original <=10% empirical FPR rule, min_count48, and save before EVAL.

Only one correctly configured fit, no extra seeds, losses, thresholds or data.
Mechanical recovery preserves failed attempts; evaluation can resume from saved
final weights without fitting again. No independent distance-source rerun needed:
report wrong-range events on main-source native near-only HEAD frames.

Success criteria fixed before corrected run-v2: EVAL HEAD-near TP at least294/1788
(at least10 percentage points above115/1788); wrong HEAD-far events at0.5 on
native HEAD-near-only frames decrease; HEAD_ONLY BODY false alerts below38;
complete groups at least508/600; BODY TP>=1187 FP<=69 and HEAD TP>=1150 FP<=35.
Report all values even if a criterion fails. A failure rejects this exact
continuation as a replacement, not the existence of readable upstream information.
A gain would support readout adaptability, not uniquely causal attribution: there
is no matched additional full-network continuation arm in this bounded experiment.

run-v1 was misconfigured: it initialized from G13, freezing an untrained query_point,
and compared against expanded B. It completed2000 steps and scored the already
consumed EVAL before the setup error was caught. Its receipt PASS describes script
completion only. It is INVALID_FOR_REQUESTED_COMPARISON, not a negative readout
result. Preserve its source snapshot, outputs and correction record. run-v2 uses
trained10k inputs; no hyperparameters were selected from invalid-run outcomes.
