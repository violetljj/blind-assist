# MZ64 execution and resource cost

[Protocol](MZ64_GEOMETRY_LEARNING_20260911.md) and
[results](MZ64_GEOMETRY_LEARNING_RESULTS_20260911.md) cover the same registered
two-arm comparison. All fitting/inference ran on the actual RTX5060 Laptop GPU
using the existing CUDA runtime; source I/O and independent scoring ran on CPU.
Primary UE did not overlap the model job. Worker capture optimization was
prepared independently; this does not claim simultaneous GPU execution.

| Measured part | Seconds |
| --- | ---: |
| CPU source smoke, no RGB or model | 14.916 |
| Shared feature build | 116.051 |
| CONTROL1536 updates | 46.512 |
| GEOMETRY1536 updates | 49.508 |
| Complete learned-head evaluation | 148.980 |
| Model run receipt, including input and output work | 381.847 |
| Process wrapper through verified exit0 | 388.159 |
| Independent CPU score receipt | 13.483 |
| Score process wrapper | 13.958 |

Encode8276 unique training RGB references once, shared by both arms and all
profiles. Evaluation reuses4743 cached frames and encodes8897 misses. Actual
RGB reads/encodes total17173, consuming7,002,792,656 encoded RGB bytes. The
32-frame initialization probe reuses those training features and does not add
RGB reads. The encoder statistic is82.046 seconds; source extraction includes
decoding, transfers and I/O, so this is not an end-to-end throughput number.
The gaps between timers are not a separately measured causal I/O breakdown.

The owned FP32 feature file has shape[8276,64,45,80] and7,627,161,728 logical
bytes. After actual model and score exit0, the sole-link file was verified under
an exclusive stream, then removed with native PowerShell. Actual allocated
storage released was7,627,165,696 bytes; the observed F: free-space increase
matched in this instance. No original RGB, native depth, support, source archive,
checkpoint, prediction, feature-index or receipt was deleted. The empty scratch
directory and feature plan remain as evidence.

Both initial states are exact copies of MZ62 CONTROL; the32-row/14-comparison
raw/support check passed at atol2e-5,rtol1e-6. Both fitted heads retain11020
parameters and the original loss/normalization/replay/cutoff rule. Missing anchor
context is exactly zero before and after fitting. The CPU source check rejected
eight corrupted schedule, owner or role cases. The independent scorer's synthetic
checks and686 historical metric rows passed before registration.

The actual score preserves1993 MZ62 arrays,138 prefixed MZ63 arrays and two MZ61
evaluator truth/known arrays exactly. It independently checks12150 old metric
rows,720256 known scalar decisions and380928 native winner lookups, and rebuilds
both1256-row cutoff vectors. Local UNKNOWN cells remain7,562,077 on MZ48,
7,417,741 on MZ55 and12,188,903 on MZ61. MZ36's80 attempted UNKNOWN bits remain.
Dense logits are not stored, so winning-cell argmax is not independently rerun;
native attribution uses the saved winning-cell index and unchanged source labels.

Evidence root: `artifacts.local/work/mz64-geometry-learning-20260911/`.

- Run receipt SHA: `3e917c4e2d6d56f134ab52633e572ae528e75487bc68405d2ce5a6a247d2508e`.
- Audit SHA: `048b94d8a74328c34904bb6d72372b52b4b0cec9e88415e915ca39e53501dab9`.
- Feature file SHA before release: `d7b02b9df70fd1ab1fd2ac489f94918654ac94376d2474c3ac7f3add5cb0ce10`.
- Source/index plan is retained in `feature-plan-v1`; exact allocation and
  release evidence is in `cache-cleanup-v1/receipt.json`.
- `execution-v1.json` and `score-execution-v1.json` record actual exit0 and no
  surviving task-owned Python process. No unrelated process was stopped.

No retries of fitting, source-role changes, threshold search, new source
calibration or outcome-selected profile were performed. The false-alert limits
failed in five clauses; successful execution does not override that result.
