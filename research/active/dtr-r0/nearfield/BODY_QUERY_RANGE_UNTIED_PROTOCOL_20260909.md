# Range-untied count readout

2026-09-09 EXPLORE, consumed shared-asset Development. The broad research goal
authorizes this new experiment; it does not reopen any historical frozen run.

Hypothesis: sharing a linear count readout across near/far queries impedes use of
range-associated information retained in frozen 10k B features. Existing count-only
continuation does not resolve this, ruling out removal of alert gradients alone
under its recorded budget. Split the original linear layer into two exact copies,
routed by fixed near/far query index; retain body/lateral sharing and all count
semantics. Both ranges may be positive. No mutually exclusive distance softmax.

Original checkpoint is SHA256
`db39ccfcb9f3d1c8f3722711bab314ff0cf2da708e00d91f7fb263e6fec0ece0`.
Keep encoder, query_point, support and buffers frozen. Count-only CE, seed17,
the original 2000x32 schedule, AdamW lr1e-5 and wd1e-4 are unchanged. Exactly one
valid fit, final step2000 only. Historical original B, shared count-only and
combined-loss readouts remain comparators; no refit of them. Verify starting RGB
prediction parity, schedule identity, unchanged frozen tensors and hashes.

Use the existing 5000/2000/3000 TRAIN/DEV/EVAL cache. Train only on TRAIN. Freeze
independent DEV thresholds with the original selector before scoring EVAL.
Preserve all cells, controls, UNKNOWN and complete groups. Report all four query
strata and range events from capped-count convolution, not complement shortcuts.

Replacement gate unchanged: EVAL HEAD-near query TP>=294/1788, native near-only
wrong-far<564/600, HEAD_ONLY BODY FP<38, complete groups>=508/600, BODY TP>=1187
and FP<=69, HEAD TP>=1150 and FP<=35. Report TRAIN fit and historical shared
count-only near12/1788, wrong-far562/600 separately. Partial spatial improvement
may be a component result but cannot silently satisfy this full replacement gate.

Stop this experiment after one fit and independent probability/tensor analysis.
A failure rejects this exact range-untying recipe and budget, not all visual
information or all decoders. No threshold rescue, extra seed or capture within
this run. Further hypotheses require separate explicit briefs and preserved
terminals. No natural-scene, calibrated distance, safety or deployment claims.

Operator: [training](body_query_10000_range_readout_train.py),
[module](body_query_range_readout.py). Output:
`artifacts.local/work/body-query-range-untied-20260909/run-v1`.
An exception writes failure.json; outputs require a fresh canonical directory.
No persistent worker/service; command completion releases its CUDA process.
