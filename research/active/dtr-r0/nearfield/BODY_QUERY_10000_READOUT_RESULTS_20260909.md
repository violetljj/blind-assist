# 10k B readout-only adaptation: no spatial recovery

2026-09-09 EXPLORE. Decision: **retain unchanged 10k B; keep this exact readout-only
continuation as a negative control, not a replacement**. No additional fit follows.
[Protocol](BODY_QUERY_10000_READOUT_PROTOCOL_20260909.md),
[operator](body_query_10000_readout_train.py),
[independent analysis](body_query_10000_readout_analysis.py).

## What was actually tested

Corrected run-v2 starts from trained10k checkpoint `db39ccfc...ece0` and updates
only query_readout.weight/bias with the unchanged TRAIN5000 source, exact original
2000x32 sampling schedule, seed17, AdamW lr1e-5 and weight_decay1e-4. Backbone,
query_point, support parameters and all buffers are fixed. The original joint
loss remains; support loss is constant with respect to the trainable layer.
This is an additional readout-only adaptation, not a same-G13-initialization
retraining experiment. No extra full-network continuation is compared.

Initial TRAIN prediction parity is exactly0, frozen parameters/buffers are checked
after fitting. Final step2000 is selected without a checkpoint sweep. DEV cutoffs
are selected independently and saved before EVAL scoring: baseline BODY/HEAD
0.039468/0.511381; adapted0.049600/0.569344. All partitions are already consumed
shared-asset Development, not fresh confirmation or natural-scene generalization.

## Main results

| EVAL metric | 10k B | Readout-only |
| --- | ---: | ---: |
| HEAD-near TP / positives at0.5 | 115/1788 (6.43%) | **105/1788 (5.87%)** |
| Wrong HEAD-far events on native HEAD-near-only frames at0.5 | 564/600 | **568/600** |
| BODY TP / FP | 1187 / 69 | 1187 / 65 |
| HEAD TP / FP | 1150 / 35 | 1148 / 35 |
| HEAD_ONLY BODY false alerts | 38/600 | 35/600 |
| Complete groups | 508/600 | 509/600 |
| BODY AUC | 0.997122 | 0.997202 |
| HEAD AUC | 0.986954 | 0.986826 |

HEAD-near also fails to recover on TRAIN191/2978 to167/2978 (6.41% to5.61%) and
DEV93/1191 to87/1191 (7.81% to7.30%). Small alert/control improvements do not meet
the spatial recovery criterion (at least294 EVAL near query hits). HEAD alert
preservation also fails by two positives. Do not retain this arm merely because
complete groups improve by one or BODY false alerts decrease.
Native three-cell HEAD-near event TP changes282 to287, a different unit from
nonempty query-cell recall; the wrong-far increase remains visible. Events are
computed by count-distribution convolution, not by treating1-S_far as near.

## Gradient evidence and interpretation

Separate alert/count gradients on the readout are recorded at step1 and each100
steps. All21 sampled batch cosines are negative, range -0.99946 to -0.77715;
step1 -0.99576, step2000 -0.99763. This shows recurring objective tension at the
readout during this specific continuation, not only a frozen endpoint snapshot.
Batch gradients are diagnostic samples, not proof of a unique cause or evidence
that a particular gradient surgery will improve attribution.

The [previous frozen diagnostic](BODY_QUERY_10000_FROZEN_DIAGNOSTIC_20260909.md)
showed range-associated feature readability. This run shows that updating the
existing shared linear readout under the unchanged objective and budget does not
turn it into correct spatial count attribution. It does not prove that all
readouts fail, that the backbone lacks information, or that pooling is responsible.
Do not extrapolate a negative result to a different decoder, loss, initialization,
or optimizer budget. A next mechanism remains a separate decision; no weight
sweep, seed extension, or new capture was performed.

## Invalid first attempt and execution evidence

run-v1 was an operator configuration error: G13 initialization plus old expanded-B
comparison froze an untrained query_point. It completed2000 steps and evaluation,
but is **INVALID_FOR_REQUESTED_COMPARISON**. Its execution receipt PASS cannot be
used as scientific validity. Its original executed source was reconstructed and
verified against the recorded SHA256, then preserved alongside all predictions,
weights and invalid-comparison.json. Its outcome is not method-negative evidence.
Corrected run-v2 changes only setup to the intended trained10k starting point,
with preflight identity/parity and explicit requires_grad freezing. No hyperparameter
was selected from run-v1 outcomes. Total task compute includes two fits, one invalid
and one valid; do not report the task as having executed only one fit.

run-v2 checkpoint SHA256:
`34a690ed5a7c483efce41a45d68bf5e6ab0795ed79caebd280c874212e6de326`.
Valid fit48.70s; full corrected run83.28s on CUDA / NVIDIA GeForce RTX5060 Laptop GPU.
Invalid full run196.86s. Durable artifacts are under
`artifacts.local/work/body-query-10000-readout-20260909/run-v1` and `run-v2`.
No model was promoted; baseline files remain unchanged. Processes exited and
no worker or persistent GPU service was allocated by this task.

Independent analysis recomputes DEV selections, split confusion/group counts and
range-event probabilities. Direct CPU checkpoint tensor comparison finds exactly
query_readout.weight and query_readout.bias changed; all other state tensors are
identical. HEAD_ONLY counts are checked against the condition field and original
result. Validation PASS means consistent execution/scoring, while acceptance
all_pass=False. This does not relabel the negative method result as a success.
The range-event helper passes all64 deterministic three-cell count combinations;
syntax, report links and scoped diff checks pass. Staged-source knowledge tests
pass10/10 and library validation passes. The decision engine retains the previously
verified history-recall failure0.70 versus0.80, with the same three cases as the
prior diagnostic's unchanged-HEAD check. No unrelated repair or broad retesting
was added. `delivery-checks.json` preserves this delivery gap.
