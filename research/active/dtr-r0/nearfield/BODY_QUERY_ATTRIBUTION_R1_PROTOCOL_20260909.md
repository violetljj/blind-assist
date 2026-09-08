# Attribution R1: supervise local query evidence without changing pooling

2026-09-09 EXPLORE, consumed same-world Development. Q3 found full local HEAD
coverage in DEV47/47 and EVAL46/46, yet owned strong-point rates decline across
regions. Test an auxiliary supervision intervention, not a pooling replacement.

Baseline: completed BODY_QUERY B, original G13 initialization, seed17 schedule,
240 TRAIN frames,2000 AdamW steps, batch32, lr1e-5, frozen BN statistics. New
challenger has identical initialization and forward path, mean pooling, count
labels/readout and original near/support/query losses. Add one Linear32->1 head
on each query_point output, used only during training, with coefficient0.25.

Target: each cell's native visible membership max-pooled into20x20 blocks, then
sampled with the existing18x32 bilinear projection weights. This soft target is
local visible-evidence attribution, not exact point occupancy or true receptive
field ownership. Ignore out-of-FOV samples and any sample whose weighted local
footprint includes invalid native depth. Do not interpolate depth. TRAIN labels
only enter optimizer. Known samples use unweighted BCE with logits; no balanced
CE, attention, pooling change, added capture or backbone replacement.

One new2000-step fit from original initialization, not continuation of frozen B.
Final checkpoint only; original DEV selector FPR<=.10/min_count8, persist cutoffs
before EVAL. Compare cached B and challenger on TRAIN/DEV/EVAL query recall,
HEAD/BODY TP and FP, AUC, group correctness, and LOW/ABOVE/BODY_ONLY controls.
Primary useful effect: improved HEAD recall at a comparable low-FP operating
point without BODY regression; disclose region-dependent and consumed-data
results. No automatic model promotion or further recipe sweep. End this fit at
2000 steps even if negative; preserve terminal, checkpoint and receipts.
