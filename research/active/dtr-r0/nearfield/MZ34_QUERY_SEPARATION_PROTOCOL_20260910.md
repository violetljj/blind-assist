# MZ34: test query parameter sharing after a fixed gradient diagnostic

2026-09-10 EXPLORE, consumed Development. MZ32's wider supervision restores
BODY_NEAR retention but loses HEAD_NEAR false-alert correction while improving
HEAD_FAR. Its shared268->32->1 network conditions on a query one-hot. The
hypothesis is that different query responsibilities compete for shared weights.

Before fitting, require completed MZ33 TRAIN-only gradient diagnostics at the
frozen initial/final endpoints. At the final endpoint at least one pair of
nonzero per-query gradients over shared parameters (excluding the disjoint
one-hot input columns) must have cosine<=-0.1, and at least one query must
still have a TRAIN responsibility error. This establishes a local conflict and
remaining fitting opportunity, not proof that sharing causes transfer loss.
If the condition fails, do not fit this candidate or lower the admission rule.

If admitted, make one structural change: four separately trainable copies of
the original MZ32 shared selector, one per query. Each copy retains268 inputs,
32 ReLU units and one output; its weights are cloned from the same frozen
MZ30/MZ32 initial state. The initial four-query function must match the shared
model within1e-6 on all cached TRAIN features, with identical output signs.
Independent copies have34564 parameters versus8641; the increased parameter
count is an inherent part of removing sharing, not an isolated capacity test.

Keep the exact MZ32 original TRAIN set,1165 disagreement supervision mask,
478/687 responsibility classes, global inverse-frequency formula, feature
normalization,1200x16 batches, Adam0.001 and global eligible-query batch-mean
loss. No query reweighting or supervision narrowing. Runtime still edits only
baseline-positive, geometrically unsupported RGB/ToF disagreements. Keep MZ28
additions and the original oldDEV-only zero-TP-loss calibration algorithm.

One1200-step fit; no architecture/seed/step sweep or posthoc threshold rescue.
Evaluate the same normal/wrong-local-visual views and all paired FP/TP changes
against MZ28, MZ30 and MZ32. A decisive improvement requires placementFP<=20,
no MZ28 TP loss across normal cohorts, unchanged additions, no new positive
bits versus MZ28, and trained pole>=48/49 clean/stress. Report raw tradeoffs if
the gate fails. Independent audit checks initial cloned tensors/function,
supervision/weights/batches/calibration and task outputs. Record actual runtime.

The diagnostic and this fit use consumed in-sample branch predictions and
development outcomes. No current truth/site ID is an inference input. No new
source, protected EVAL, App/default change, device or safety claim is included.
Output: artifacts.local/work/mz34-query-separation-20260910/run-v1/.
