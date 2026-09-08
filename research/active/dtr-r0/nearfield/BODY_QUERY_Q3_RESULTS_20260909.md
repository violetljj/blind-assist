# Q3: local coverage is high; correct query responses do not transfer reliably

2026-09-09 EXPLORE. Consumed same-world TRAIN240/DEV40/EVAL40, frozen Q2 logits.
All320 original native arrays were retrieved from the worker (294,952,960 bytes)
without worker compute, capture or process interruption. Local GPU geometry
reproduced all320 cached count labels. No new model inference or training in Q3.

## Results

| Nonempty query cells | TRAIN | DEV | EVAL |
| --- | ---: | ---: | ---: |
| Total |576|93|97|
| At least one owned nearest native pixel |462|74|75|
| At least one locally covered projected sample |575|93|96|
| Original missed cells |17|68|68|
| Misses with exact-ray coverage |15|56|60|
| Misses with local coverage |16|68|68|
| Misses with an owned strong point |8|12|15|

Exact-ray ownership samples the nearest native pixel; local coverage max-pools
each cell's native membership into20x20 blocks then applies the exact bilinear
feature projection. The local quantity is a proxy, not the network receptive
field or proof of accessible discriminative information. No depth interpolation.

HEAD local coverage is TRAIN290/290, DEV47/47, EVAL46/46. HEAD-near exact-ray
coverage is75/82,12/12,6/6. Thus the near held-out misses are not attributable to
all reference rays missing the obstacle. Yet only3/12 DEV and3/6 EVAL HEAD-near
misses have an owned strong point under the frozen decomposed readout.

Among all owned points, strong responses are TRAIN2989/3502 (85.35%),
DEV204/530 (38.49%), EVAL245/727 (33.70%). These are point-weighted descriptive
rates from a classifier trained on pooled features; points are correlated and
the rates are not calibrated detection accuracy or independent statistical units.
Known-other points also respond strongly:12742/58091,1181/9642,1278/9172.

BODY_ONLY HEAD strong rays are especially informative:

| Native surface category | TRAIN | DEV | EVAL |
| --- | ---: | ---: | ---: |
| BODY height, depth <=3.18m |313|27|7|
| LOW height, depth <=3.18m |65|24|8|
| Farther than3.18m |605|55|234|
| UNKNOWN native depth |9|0|131|

No remaining categories have strong rays in this slice. Far distance takes
precedence over height; these categories describe ray endpoints, not the full
feature content. Not all incorrect HEAD responses are the BODY obstacle being
mistakenly claimed: far surfaces and unknown rays dominate EVAL. UNKNOWN is
never negative/free-space truth.

## Decision

Do not expand sampling or choose attention solely from these results. Existing
local coverage is nearly complete; correct-region response transfer and
selectivity remain weak. Test one controlled auxiliary attribution intervention
while retaining mean pooling and the runtime count path. This tests whether
stronger local query supervision helps, without claiming the backbone or pooling
has been causally isolated. The successor has its own
[R1 protocol](BODY_QUERY_ATTRIBUTION_R1_PROTOCOL_20260909.md) and new budget;
the original B checkpoint and completed fit remain immutable.

## Evidence and validation

[Protocol](BODY_QUERY_Q3_PROTOCOL_20260909.md),
[implementation](body_query_ownership_audit.py).
Payload root: `artifacts.local/work/body-query-q3-20260909/`.
`run-v1/` contains per-split owned/known/local/category arrays, result and receipt.
`native/` retains original source arrays. `validation.json` independently checks
103,680 nearest-pixel memberships and3,840 explicit native-tile footprints using
NumPy, and verifies receipt hashes. GPU geometry and cached labels agree for all
frames. No retained worker allocation or Q3 process remains.

This is sampling/attribution evidence, not a network recall ceiling, a full
occupancy ground truth, fresh confirmation, or device performance.
