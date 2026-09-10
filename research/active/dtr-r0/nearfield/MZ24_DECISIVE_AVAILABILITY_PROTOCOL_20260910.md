# MZ24: protect true witnesses and penalize the highest unavailable candidates

2026-09-10 EXPLORE, consumed Development. MZ23 preserved74/75 far additions and
pole49/48, but left7/8 addedFP. Six residual locations are19..25native pixels
from actual support. A stricter common availability cutoff would leave14/75
far additions. This motivates a training objective aligned with extreme local
scores, rather than cutoff adjustment or an assumed one-cell boundary defect.

One matched1200-step fit, same10577-parameter AngularAvailability, seed123,
initial state, Adam0.001, normalized native224ROI features and1200x16 batch IDs
as MZ23. Freeze MZ20/MZ5. No extra capture, feature extraction, EVAL access,
training extension, architecture change, seed/weight sweep or operating-point
selection. Artifacts: `artifacts.local/work/mz24-decisive-availability-20260910/run-v1`.

Change only the objective, with two coupled terms:

- Retain MZ23 class-balanced per-frame dense availability BCE at weight1.
- At weight0.25, penalize softplus(max availability) over known-unavailable
  geometrically eligible cells, separately for each frame/query with candidates.
  This supervises absent <=4m evidence, not absent obstacles or negative UNKNOWN
  query labels. Echo duplicates share one angular availability output.
- At weight0.25, preserve one actual query contributor per eligible positive
  frame/query: select the contributor with the largest frozen MZ20 raw score,
  then penalize softplus(-availability) at that location. Selection reads only
  training contributors and frozen scores, never calibrated DEV cutoffs. An
  unavailable/empty positive bag contributes no positive term.

Weights are fixed before fitting as a bounded contrast; no tuning claim. The
two terms change together, so improvement would not isolate their contributions.
Training-only native availability and contributor labels never enter inference.
Save inputs, initialization, batches, objective history and full first/last-batch
loss inputs/availability gradients for independent scalar and gradient checks.

Inference remains exactly MZ23: availability logit>=0 restricts MZ20 geometric
candidates; fixed MZ20 per-query cutoffs; add-only unchanged MZ5 fallback. Every
baseline positive must survive, and no new positive beyond MZ20 is possible.
UNKNOWN remains UNKNOWN. Evaluate oldDEV, clean/stress, both placements and
the same both-chain wrong-zone controls; report raw gains/losses and contributor
rejection. Useful effect requires zero addedFP on all normal cohorts, at least
68/75 placement far additions with gain on both cohorts, and trained pole>=48/49
clean and stress. These are experiment utility criteria, not safety guarantees.

Run focused objective tests and independent output audit with selected fixed
model inference. If useful, retain the changed mechanism for further authorized
validation; if not, stop this recipe and diagnose whether the added pressure
separates TRAIN extrema versus whether that separation transfers. A low loss,
zero alerts, or fewer FP obtained by deleting true detections is not success.
Do not rescue the frozen run by a new cutoff, weight or additional steps.
Finish the result/terminal/scoped delivery and release task-owned resources.
No default-App, device latency, physical-sensor, temporal or safety claim.
