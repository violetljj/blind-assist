# NF-G5: suspended-structure frontend diagnosis

EXPLORE, consumed eighteen NF-G3 static views; all cases and denominators kept.
Primary failure: native suspended-bar center/head surface at1.96 m is predicted
around14.29 m by Hypersim Small,so no support/fusion can recover it. The new
cane-complementary priority makes this a frontend target,not a low-boundary task.

First compare the official same-size VKITTI Small checkpoint at input518 and
its documented max-depth80 m with cached Hypersim Small input518/max20 m.
Official source:
https://github.com/DepthAnything/Depth-Anything-V2/blob/main/metric_depth/README.md
The official models differ in training domain and output range; this is a model
configuration comparison,not a pure causal domain ablation. Both use the existing
ViT-S/DPT implementation; no bigger backbone or changed postprocessing.

VKITTI source revision:c725b8589bdf6ab04072cab74c0467830db80d6d.
Weight SHA-256:9203e538d35255c90dda4b7fedb47ff33fe725497bcca3b1e53b3a65ee63f0cb.
Verify official LFS digest before weights_only loading. Do not commit weights.

If this candidate fails to recover center/head bar evidence or introduces
reference FP, test one predefined resolution diagnostic: existing Hypersim Small
at input1036,max20 m. This tests extra detail cost,not a larger learned model.
Maximum36 new calls (18 each),no crops,per-frame scaling,threshold sweep or new
capture. Stop after these configurations; do not relabel existing misses.

Keep raw,ground and non-destructive fusion unchanged. Report all9-cell and
directional reference scores,body/head subgroup and per-case regressions.
Recovered bar must have near center/head evidence; direction recall alone can
be satisfied by a lower object and is insufficient. Retain a frontend only if
it recovers the bar without added reference FP; disclose other regressions and
compute cost. Successful consumed comparison is not natural generalization.

For the fixed native supported center/head mask, report each candidate's raw
forward-depth distribution and eligible/supported near evidence. Native is
evaluation-only,never supplied to prediction/fitting. Cache every new prediction
before scoring; subsequent comparison reads cached branches,not repeated models.
Model and dense geometry use CUDA; metadata/9-cell fusion use CPU
TASK_NOT_GPU_SUITABLE. Report actual device and timing. No App promotion.

## Results

Completed 36 new CUDA inferences, zero simulator launches. Cached baseline and
both candidates use identical 18 RGB/native identities and camera contracts.
All scores below use unchanged raw/ground non-destructive union.

| Configuration | Cell TP/87 | Cell FP | Direction TP/44 | Direction FP | Body/head TP/47 | Body/head FP | Bar forward median | Inference P50 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Hypersim Small 518 baseline | 52 | 0 | 39 | 0 | 16 | 0 | 14.288 m | 91.21 ms |
| VKITTI Small 518 | 0 | 0 | 0 | 0 | 0 | 0 | 51.855 m | 93.55 ms |
| Hypersim Small 1036 | 70 | 9 | 44 | 8 | 30 | 1 | 13.429 m | 925.28 ms |

Neither candidate recovers the suspended-bar center/head alert. The same 1054
native supported pixels have forward median 1.960 m. High resolution produces
39 valid-depth pixels there but zero near supported head pixels; baseline and
VKITTI have zero valid pixels there. Ground fits fail 3/18,18/18,2/18 respectively.
The apparent 44/44 directional recall therefore does not mean bar recovery:
other height evidence satisfies that collapsed cell, and false alerts increase.
Body/head scores remain branch-height diagnostics, not verified physical height.

Decision: retain the existing frontend baseline; adopt neither candidate. The
frontend defect is unresolved. This bounds two simple fixes, not all monocular
frontends. A future separately scoped test should examine local structure and
range evidence without relying entirely on one globally regressed depth map;
no such branch, adaptive policy, or threshold change is implemented here.
Stop the predefined experiment; do not sweep more settings on these views.

Evidence: `artifacts.local/nearfield/frontend-domain-20260907-v1/comparison.json`
contains every case's gained/lost cells, branch/fusion scores, model receipts,
verified cache hashes and reference-mask statistics. Per-candidate prediction
and evaluation directories retain all arrays and timings. Candidate inference
totals were 5.41 s and 20.31 s (model loading included, Python startup excluded).
Native geometry is evaluator-only. Static consumed synthetic frames establish
neither dynamic/first-warning performance nor natural-scene accuracy.

Receipt limitation: the reused runner copied the old NF-G3 brief into each run
directory. The NF-G5 pre-score brief was separately saved at the experiment root
before inference and remains unchanged there; these historical receipts are not
rewritten. The runner now accepts `--protocol` for explicit future attribution.
The comparison script does no inference and verifies all 18 cache identities.
