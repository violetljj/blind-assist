# MZ17: reward a supported witness, penalize high-scoring wrong candidates

2026-09-10, EXPLORE, consumed Development. User authorized the diagnostic and,
if it supports the proposed conflict, one matched training-objective intervention.
MZ16 HIGH_DETAIL remains a failed challenger, MZ5 the composed baseline.

Hypothesis: the positive max-pooled query loss rewards an incorrect spatial
winner, while its local negative loss is diluted among many known negatives.
This can improve final positives without learning reliable echo attribution.
It is a specific optimization hypothesis, not an established dominant cause.

First inspect the frozen final MZ16 HIGH_DETAIL checkpoint on the first64 saved
TRAIN batches (1024 draws). No optimizer, updates, DEV or checkpoint selection.
Differentiate the original local loss and weighted0.25 query loss with respect
to candidate logits. Report supervised positive queries, known wrong winners,
unknown winners, ties, each gradient sign, and the summed gradient. Negative
gradient means gradient descent would raise that candidate logit if independent.
It does not prove a shared-parameter optimizer step or past training trajectory.
Proceed to the single fit only if at least10% of supervised positive outputs
have a known wrong winner and at least50% of those have net upward logit pressure.
Otherwise stop the proposed fit and deliver the diagnostic as insufficient support.

If admitted, reuse MZ16 native-detail224x224 ROI cache, frozen encoder,10740parameter
QUERY readout, original MZ15 initialization, saved1200x16 batches, Adam0.001,
normalization and existing local BCE. Only replace the weighted query-loss term:

- Positive witness term: softplus of minus the highest score among known valid,
  geometrically eligible actual query contributors, on full-query positive frames.
- Wrong-candidate term: softplus of the highest score among known valid eligible
  non-contributors, on both positive and negative frames.
- Sum present terms per frame/query, average over frame/queries with either term;
  multiply by0.25 and add the unchanged balanced local loss. This coupled objective
  changes gradient allocation and magnitude; it is not a pure one-coefficient test.
- No known positive means no positive witness reward. Unknown cells are neither
  positive nor negative. Multiple true cells may coexist; no unique-source softmax.

Inference remains the identical unmasked maximum over geometric candidates and
receives only RGB features and ToF ranges/validity. Native labels are training
supervision/evaluation only; no witness restriction at inference. The comparator
is the completed MZ16 HIGH_DETAIL run, not another refit or warm-start extension.
Cache/batch/initial-state identity and frozen code hashes must be verified.

One1200-step candidate only, no coefficient sweep or rescue fitting. Calibration
uses the same oldDEV1000 zero-added-MZ5-FP rule, then freezes. Replay oldDEV1000,
trained clean/stress200, consumed relationDEV2000 and distanceDEV1000 plus wrong-zone
RGB controls. Preserve stress echo-slot contributor alignment. Training pool and
exposure match MZ16:7562 unique sampled TRAIN frames including200 diagnostic frames.
No capture, EVAL, encoder training, temporal module or App promotion.

Primary acceptance: zero added FP per query on both placementDEV cohorts and
clean/stress, positive far recovery on both placement cohorts, pole clean/stress
at least48/49, and retained baseline near positives. These are Development-only
criteria, not an assertion that oldDEV calibration guarantees unseen-set FP.
Report exact frames, all four TP/FP, family/group/site aggregates and wrong-zone
results. Report actual-contributor AP and local precision/recall/FPR independently;
AP is explanatory, not an additional veto on useful task effect. Wrong-zone
disruption cannot alone prove the winning angular cell has the true source.

Source of the general hard-example idea: Shrivastava et al., CVPR2016,
https://arxiv.org/abs/1604.03540. This objective is a task-specific adaptation,
not a reproduction of that detector or a claimed new learning principle.

Stop after the diagnostic and admitted budget. Verify changed gradient behavior,
unknown handling, scalar task outputs and AP independently. Deliver scoped code,
report, structured terminal inheritance and normal push; release task processes
and disposable delivery resources while preserving evidence and checkpoints.
