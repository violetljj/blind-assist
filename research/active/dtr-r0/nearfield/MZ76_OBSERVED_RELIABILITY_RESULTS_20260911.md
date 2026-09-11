# MZ76: no added benefit in the final attempt; paused

The two256-step continuations completed, but the registered strict scorer failed
historical frozen-output parity. The attempt is INVALID_FOR_REQUESTED_COMPARISON
under that unchanged criterion. No tolerance was relaxed, no scorer was replaced,
and no model was promoted. Both checkpoints and the failed score remain intact.
User requested a pause after this final attempt: no successor training, inference,
recut or collection is started.

A separate posthoc saved-output diagnostic found no added final-OR benefit from
three observed neighbor-geometry features over matched ordinary continuation.
Five source/profile conditions have identical decision bits; MZ67 IDEAL loses
one BODY_FAR true positive, with no false-positive change. This diagnostic does
not turn the registered comparison into a PASS or establish a hardware result.

Each source has1024 existing consumed HELD frames:1024 positive query bits and
3072 negative query bits, with256 positives and768 negatives per query. Query
UNKNOWN is zero on this subset; native cell UNKNOWN remains distinct. Values
below are TP/FP for the full OLD_NEG OR candidate pipeline at unchanged cuts.
Frozen means recomputed originalD within this run, whose decision signs match
historical results but whose IDEAL numeric parity failed.

| Source/profile | Frozen TP/FP | BASE TP/FP | Neighbor geometry TP/FP |
| --- | ---: | ---: | ---: |
| mz61/IDEAL | 971/78 | 973/78 | 973/78 |
| mz61/CLOSEST_REPORTED_PROXY | 988/31 | 995/31 | 995/31 |
| mz61/FARTHEST_REPORTED_PROXY | 987/31 | 996/31 | 996/31 |
| mz67/IDEAL | 861/46 | 875/45 | 874/45 |
| mz67/CLOSEST_REPORTED_PROXY | 911/57 | 925/57 | 925/57 |
| mz67/FARTHEST_REPORTED_PROXY | 912/56 | 928/56 | 928/56 |

All per-query TP/FP/FN/TN and exact bidirectional event IDs are retained in the
diagnostic result. The sole lost positive is BODY_FAR on
`mz67-mz36_dense_candidate_06_site_004-open_bike_stand-BOTH-far-g24-supported`.
The continuations improve some counts over originalD, but the extra geometry
cannot claim that ordinary-training gain. Old-source IDEAL HEAD_FAR decreases
225 to223 for both continuations; improvements are not universal retention.

Strict failure: MZ61 IDEAL originalD candidate has153/4096 tolerance violations,
maximum absolute difference0.00418401. The diagnostic also finds257 violations
for MZ67 IDEAL candidate, maximum0.005632. All24 historical decision tables have
zero sign changes. MZ37 and OLD_NEG agree exactly; both partial profiles agree
within the original tolerance, and packets agree exactly. Differences localize
to originalD's IDEAL full-RGB/head path versus MZ70. Saved winner indices also differ (14/6 IDEAL entries and one MZ67 entry
per partial profile); identical frame decisions do not establish identical
localization. The exact numerical cause
is unresolved: no causal attribution to batching, hardware, or model mutation
has been demonstrated. The original tolerance2e-5 absolute/1e-6 relative remains.

Implementation checks passed three CPU tests before fitting;32 TRAIN frames
passed prefit original/expanded parity. Both arms started from the same originalD
parameters, used4096 TRAIN frames each exactly once, and256 Adam updates. The
three extra BASE inputs are zero, so equal11116 parameter counts do not establish
equal effective capacity. The backbone, originalD and cuts remain frozen. No
ALL_INVALID profile, new hardware-quality fields, or fresh source frames entered.
Historical packets contain only ranges/valid/frame identity. Neighbor geometry
is not measured signal, ambient, sigma, reflectance or calibrated confidence.

The RTX5060 Laptop run took118.366s (124.102s launch wrapper). Recorded RGB decode
time was54.960s, encoder/view time35.054s, and both arms' updates6.239s. These
components exclude other overhead. There were6176 PNG RGB loads totaling
2,565,185,160 bytes read; feature memory was limited to16 frames (14,745,600 bytes),
with no permanent dense cache. Existing RGBStore still consumes PNG: earlier
lossless-WebP size checks do not mean this training used compact WebP. No dataset
was deleted or full-scale compressed in this attempt.

The posthoc CPU diagnostic checked saved packet transformations, identities,
scalar candidate/OR arithmetic and scalar metrics; it performed zero inference,
fit or cutoff changes. The strict scorer exited1 and its failure is preserved.
Training exited0; local learner processes and compact handles were released.
The separate [source-contact smoke](SOURCE_CONTACT_SMOKE_20260911.md) completed
four worker frames and released worker processes, without dataset admission.

[Protocol](MZ76_OBSERVED_RELIABILITY_20260911.md).

[Run receipt](../../../../artifacts.local/work/mz76-observed-reliability-20260911/run-v1/receipt.json). SHA256:b13e1eb299a20f01de4d187cd3a54717f9e249c81cf0d50a2a02b9a13dc287ab.

[Strict score failure](../../../../artifacts.local/work/mz76-observed-reliability-20260911/score-v1/failure.json). SHA256:613027692fb23907a74798d2e4f75a2539b6839986dbb5c57d32ace1a553e33b.

[Score execution](../../../../artifacts.local/work/mz76-observed-reliability-20260911/score-execution-v1.json). SHA256:13ad0f8952035ace4363285e83e7607932404528f718a7ddd0cc0c4af6b5e156.

[Posthoc diagnostic and per-query exchanges](../../../../artifacts.local/work/mz76-observed-reliability-20260911/baseline-diagnostic-v1/result.json). SHA256:bb72acf2ea1e8c8e9d781987ec5f7bdfa8043cb3797ccefe31db07bc12e447f2.

[Diagnostic script](../../../../artifacts.local/work/mz76-observed-reliability-20260911/baseline-diagnostic-v1/diagnose.py). SHA256:53bfb00fc18b475292ed09e94436fdb05ddc3a300616766f42d16e71ff1dc61e.

[Diagnostic receipt](../../../../artifacts.local/work/mz76-observed-reliability-20260911/baseline-diagnostic-v1/receipt.json). SHA256:d64dcb30e5b17c703c0743f42dbb3e62286a0cf9d53659056a32414ad3c2d8f7.
