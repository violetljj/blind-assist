# MZ66 execution and resource cost

The [protocol](MZ66_REPLAY_PROJECTION_20260911.md) and
[results](MZ66_REPLAY_PROJECTION_RESULTS_20260911.md) cover one actual PROJECT
fit and one independent CPU score. Both processes exited0 with no surviving
task-owned learner/scorer Python. No unrelated process was stopped. Mechanical
PASS does not override the **5/11 retaining failure**.

The learner uses CUDA on the primary NVIDIA RTX5060 Laptop GPU (8GB);
projection arithmetic stays on the same device. The saved-array scorer uses CPU
because its work is small reductions, byte comparisons and metadata checks.

| Measured interval | Seconds |
| --- | ---: |
| CPU source preparation, no images/model | 12.906 |
| Shared feature build | 79.134 |
| PROJECT1536 updates, including replay measurements | 55.474 |
| Complete new-head evaluation | 136.009 |
| Run receipt, including input/output work | 292.266 |
| Learner wrapper through exit0 | 297.456 |
| Independent CPU score receipt | 9.977 |
| Score process wrapper | 10.654 |
| Owned cache cleanup | 7.770 |

One temporary cache holds6676 fit-unique RGB frames:1095 MZ48,3533 old replay,
2048 MZ61. MZ55 has no training-cache entries. Evaluation reuses3143 cached
frames and encodes10497 misses. Actual total RGB reads/encodes are17173 and
7,002,792,656 encoded PNG bytes, sharing each image across profiles. The
encoder-only statistic is80.375 seconds; timings overlap in scope and are not
an independently isolated I/O decomposition or Android latency measurement.

Both state and schedule remain bound to the prior experiment: initial checkpoint
SHA `b7106449e4246b7bd7933659cbf0dd815691072c03a8b522633b7aee9a456d2e`;
MZ64 groups/schedule are byte-identical. A32-frame TRAIN initialization check
(16 MZ48×three profiles,16 MZ61×four) verifies raw/support against the saved
warm start at atol2e−5, rtol1e−6. This is not a full baseline replay. The head
has11020 parameters; missing anchors remain exactly zero before/after fitting.

The scorer independently preserves2486 historical arrays byte-for-byte and
checks18954 historical metric rows,360128 known scalar decisions,126976 native
winner lookups and the original1,256-row cutoff. MZ36's80 UNKNOWN query bits
remain; local UNKNOWN cells remain7,562,077 /7,417,741 /12,188,903 for
MZ48/MZ55/MZ61. No RGB or native depth is decoded by scoring. Dense logits are
not stored, so the full winning argmax is not regenerated.

All1536 saved assigned-dot bounds are checked. Sixteen fixed, outcome-independent
steps additionally preserve before/proposed/gradient/assigned FP32 vectors of
length11020. NumPy independently reconstructs the closed form in FP64: inactive
projection requires identical bytes; active projection permits one FP32 ULP plus
the explicit propagated FP64 reduction bound. The other1520 parameter vectors
are not independently reconstructed. Ideal-dot roundoff and final FP32 rounding
remain separate; the residual bound includes positive ideal-dot residual.
Actual nonlinear replay increases are descriptive and never select updates.

Preparation checks cover11-clause cases, exact cutoff reconstruction,786
historical aggregate metrics and50 corresponding candidate/OR aliases, plus
synthetic vector corruption rejection. Two preparation-only bookkeeping/test
assertion corrections are preserved under `scorer-preparation/`; neither
started a fit nor scored scientific outputs. The actual fit and score had no
rerun, criterion change or source/cutoff modification.

After sealed outputs and exit0, only the sole-link temporary
`scratch-v1/training-full.npy` was removed. Its logical size was6,152,601,728
bytes; measured allocated storage released was **6,152,605,696 bytes**. The
empty scratch directory and feature plan remain. The exclusive handle was
disposed before deletion; target absence and no owning processes were verified.
All28 checked preserved files, including19 durable run/score files, remained unchanged.
Concurrent capture writes mean the volume-wide free-space delta is not solely
attributable to this deletion; the target allocation is the isolated measure.

Evidence root: `artifacts.local/work/mz66-replay-projection-20260911/`.

- [Run receipt](../../../../artifacts.local/work/mz66-replay-projection-20260911/run-v1/receipt.json): `0e562511c0396baa3f2e3fa87e6d5c0e3ef9f34a61b9ed8e36ca0dee5c8ebec4`.
- [Independent audit](../../../../artifacts.local/work/mz66-replay-projection-20260911/score-v1/audit.json): `7d46c36b257e596ec73b981a6044f3119c54ebb74ff47fdb6fa2d61cd5016b11`.
- [Cleanup receipt](../../../../artifacts.local/work/mz66-replay-projection-20260911/cache-cleanup-v1/receipt.json): `7abe58de00c75c2d478863417e16bea429f49aa9946df4e6696416e78a0b9c06`.
- Cache SHA before release: `45c7fe1507ea1702c04874b0f2c109650e6541bf78e5c19f5271758bf6c44a00`.

The numerical projection works within its declared bounds. Its failed Pareto
result remains a scoped `NEGATIVE_CONTROL`; MZ64's geometry-learning
`CHALLENGER` and all original comparator evidence remain available.
