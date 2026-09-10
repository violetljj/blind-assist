# MZ35: fixed optimization endpoints for expanded responsibility supervision

2026-09-10 EXPLORE, consumed Development. MZ33 finds66/1165 raw TRAIN
responsibility errors at the MZ32 endpoint, with positive aggregate shared
gradient pair cosines. Its fixed local-conflict gate does not admit MZ34's
query separation. This supports testing remaining optimization opportunity,
without asserting the TRAIN errors are all optimizable or that more training
must improve transfer. Earlier MZ26 duration findings concern another model and
objective; they remain evidence of possible TRAIN/DEV divergence.

One continuous4800-step trajectory of the unchanged MZ32 shared8641-parameter
selector. Use its exact initial tensors, normalization,1165 TRAIN disagreement
mask,478/687 class weights and Adam0.001. The original1200x16 MZ20 batch stream
is repeated four times in the same order. No new samples, architecture, loss,
runtime eligibility, query reweighting, augmentation or stochastic schedule.

At step1200 save the checkpoint and compare every tensor bit-for-bit with
frozen MZ32 selector.pt and every first1200 loss/exposure row with its saved
training-loss.npy. Continue to4800 only if both match. On a mismatch preserve
the attempted prefix and stop as INVALID_FOR_REQUESTED_COMPARISON; no repeated
fit or relaxed tolerance. Save optimizer state at the two specified endpoints.

At4800 apply the same oldDEV-only zero-TP-loss cutoff selection algorithm and
unchanged MZ30 runtime repair mask on MZ28 task decisions. Compare all normal
and wrong-local-visual cohorts with frozen MZ28/MZ30/MZ32; report FP/FN, exact
frames, query/site/family tradeoffs and trained pole. Also measure final raw
TRAIN responsibility errors and weighted BCE on all1165 disagreements to
distinguish optimization response from task transfer. Initial/1200 metrics
come from the frozen MZ33 diagnostic; no intermediate selection is introduced.

The fixed decisive-improvement gate is placementFP<=20, no MZ28 TP loss on
normal cohorts, no new positive bits versus MZ28, unchanged additions and
trained pole>=48/49 clean/stress. Report raw benefits and losses even if this
gate fails. One trajectory, no later endpoint, cutoff rescue, seed or new fit.
Independent audit verifies exact prefix parity, repeated schedule/supervision,
weights, calibration, final task counts and saved TRAIN metrics. GPU neural
work; record actual runtime. Preserve all evidence and release processes.

Output: artifacts.local/work/mz35-responsibility-convergence-20260910/run-v1/.
In-sample TRAIN predictions and consumed DEV do not establish independent
generalization, phone behavior, temporal performance or a safety guarantee.
