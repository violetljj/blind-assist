# 10k B frozen TRAIN gradient and feature diagnostic

2026-09-09, consumed controlled Development, zero fits and zero optimizer steps.
Decision: retain 10k B for its existing alert scope; retain this diagnostic as
component evidence for a future readout/optimization intervention, not a model
replacement or proof that pooling/backbone is the unique cause.

## Question and fixed scope

The [10k result](BODY_QUERY_10000_RESULTS_20260909.md) retains alerts but leaves
HEAD-near at 6.43% EVAL query recall. The preceding cache audit found TRAIN
191/2978 (6.41%) versus HEAD-far 2847/2937 (96.94%): fitting failure exists inside
TRAIN. This diagnostic asks whether the frozen checkpoint has opposing alert/count
gradients and whether frozen sampled features retain readable range information.
The older [probe](BODY_QUERY_PROBE_RESULTS_20260909.md) and
[Range R0](BODY_QUERY_RANGE_RESULTS_20260909.md) remain scoped historical evidence;
this is a new checkpoint/data diagnosis, not a repeat of their fits or a rescue.

Inputs: unchanged checkpoint SHA-256
`db39ccfcb9f3d1c8f3722711bab314ff0cf2da708e00d91f7fb263e6fec0ece0`,
10k cache manifest `2b20909395f997b79e56dd44039e96811c7c3224f1a03f40fd65178ab9532985`.
Only the 5000 TRAIN rows and saved NEW TRAIN predictions were consumed; DEV,
EVAL and distance-pair source were not opened by the diagnostic operator.

[Operator](body_query_10000_frozen_diagnostic.py) computes original mean alert BCE
and weighted `.25 * mean count CE` gradients on all TRAIN count logits. Shared
parameter gradients use 128 fixed seed17 samples (64 HEAD_ONLY near-only and
64 far-only), averaging four 32-example batches. No optimizer is constructed.
The support loss has no path to count logits or query_point/query_readout; its
shared-backbone contribution is not included in the reported pairwise comparison.

Feature retrieval uses all 1000 exclusive-range HEAD_ONLY examples (500/500),
cosine nearest neighbour excluding the entire query region, with no fitted probe.
Five TRAIN regions are represented. Each region was already seen by the encoder
during training: this is cross-region retrieval inside consumed TRAIN, not unseen
region generalization. One fixed shuffled-label control is descriptive only.
Stop after fixed-gradient and frozen-feature checks; no training or threshold
selection, no automatic architecture change or additional data.

## Gradient evidence

On 500 HEAD_ONLY near-only frames, all 1500 far query cells have opposing
alert/count directions for the class3-minus-class0 logit contrast. Alert descent
raises it, while count descent lowers it. This demonstrates a local wrong-range
incentive from final alert aggregation; it does not describe the entire nonempty
probability over count classes1/2/3.

However, the weighted count gradient is not generally too small. On the correct
near queries its norm is 16.53 times the alert gradient; on wrong far queries it
is 1.45 times. The combined output-space gradient raises the class3-minus-class0
contrast in only 66/1500 wrong far cells. Thus most wrong far cells already have
a correcting direction at the independent output level. A simple explanation
that alert loss overwhelms all spatial supervision is not supported.

The two objectives can interact differently after sharing parameters:

| Parameter block | Alert/count gradient cosine | Weighted count / alert norm |
| --- | ---: | ---: |
| query_point | -0.2156 | 0.8869 |
| query_readout | **-0.9634** | 0.8604 |
| deep_projection | -0.1102 | 0.8169 |
| detail | 0.4337 | 2.2186 |
| backbone | 0.9050 | 0.2685 |

The shared readout has strong pairwise local gradient opposition on this balanced
HEAD_ONLY subset. It is not a reconstruction of the 2000-step trajectory, the
full TRAIN sampling distribution, support gradients, or AdamW's optimizer state.
Negative cosine alone does not prove the historical failure's cause or predict
that gradient surgery/loss reweighting will improve the retained model.

## Frozen feature evidence

Raw descriptors are masked mean sampled image features per query, before XYZ
conditioning/query_point; embeddings are immediately before query_readout.
Three lateral query descriptors are concatenated within the named range.

| Fixed descriptor | Correct / 1000 | Shuffled-label correct / 1000 |
| --- | ---: | ---: |
| Raw HEAD-near | **857** | 498 |
| Raw HEAD-far | 866 | 498 |
| Pre-readout HEAD-near | **827** | 527 |
| Pre-readout HEAD-far | 795 | 499 |
| RGB channel means | 548 | 517 |

Balanced accuracy equals accuracy because the two classes have equal size.
All five regions have above-half raw-near retrieval (75/100, 220/264, 247/280,
290/328, 25/28). The frozen descriptors retain range-associated information
readable by this simple retrieval rule. This argues against declaring the current
near sampling features information-free. It does not establish metric depth,
object-local causal evidence, or independence from shared fixture/appearance
correlations. Nor does the 85.7% to 82.7% difference isolate pooling: the raw
descriptor already uses averaging and the two descriptors have different geometry.

## Interpretation and next decision

Preserve 10k B. The diagnosis narrows the next question toward converting existing
range-associated evidence into correct query outputs, with explicit attention to
the shared readout and its joint objective. It does not yet choose query competition,
a new backbone, larger loss weights, or longer training. A future single mechanism
must justify itself by correct HEAD-near attribution, fewer wrong-range activations,
HEAD_ONLY BODY false alerts, and preservation of retained alert/group performance.
No such intervention was implemented or trained in this task.

## Reproduction, failure and validation

Artifacts are under `artifacts.local/work/body-query-10000-frozen-diagnostic-20260909/`.
`run-v1/failure.json` preserves a count-parity failure of 3.6538e-5 after changing
batch composition. The mechanical correction replays the original full TRAIN
32-frame batches and selects feature rows afterwards. `run-v2` succeeds without
relaxing the original tolerance: count prediction parity is exactly 0; probability
readout from cached counts differs by at most 1.2666e-7. Both attempts have zero fits.

Successful run-v2 took 11.82 seconds on CUDA / NVIDIA GeForce RTX 5060 Laptop GPU.
State tensor digest, input manifest and checkpoint hashes remain unchanged.
Independent float64 NumPy recomputation reproduces raw-near 857/1000 and
embedding-near 827/1000. One embedding neighbour index differs at higher precision,
without changing its class or score. The selected 500 near and 500 far frames
each have all 1500 opposite-range native count labels exactly zero.
Protocol, gradient summary, result, feature arrays and hash-bound PASS receipt
are preserved. No workers or persistent process were allocated; the command exited.

Delivery checks: Python syntax, report links, receipt hashes and scoped diff pass.
The staged-only knowledge snapshot passes all 10 unit tests and library validation.
Its decision-engine history recall is 0.70 against the 0.80 gate; an unchanged
HEAD snapshot reproduces the identical three failed cases and recall. This is
recorded as a pre-existing unrelated delivery-check gap, not repaired here.
`baseline-delivery-check.json` retains the comparison. Temporary Git environment
isolation failures were corrected before these checks; no checks were relabeled PASS.

Command (from checkout):

```powershell
E:\codex-tools\tools\venvs\blindassist-torch-gpu\Scripts\python.exe research/active/dtr-r0/nearfield/body_query_10000_frozen_diagnostic.py --cache artifacts.local/work/body-query-10000-b-20260909/cache-v1 --run artifacts.local/work/body-query-10000-b-20260909/run-v1 --pretrained artifacts.local/work/body-query-v1-20260908/model-inputs/pretrained --output artifacts.local/work/body-query-10000-frozen-diagnostic-20260909/reproduction-fresh
```

Output directories must be fresh. Reproduction does not authorize further model
fits or restore the consumed source's freshness.
