# MZ74: useful surviving returns, with a slot-representation confound

The fixed-model run and independent score passed. Both original MZ70 arms and
all original cutoffs remain unchanged. Retain this as a descriptive input
sensitivity COMPONENT, not a new trained model or measured VL53L8CX behavior.

Each consumed Development HELD source has 1,024 frames, 256 positive and 768
known-negative events per query, zero frame-query UNKNOWN. Totals below count
four query events per frame, not uniquely detected physical obstacles.

| Source / input | CONTROL TP / FP | DIVERSE TP / FP |
| --- | ---: | ---: |
| Old MZ61 / IDEAL | 978 / 86 | 971 / 78 |
| Old MZ61 / MERGE_CLOSE | 987 / 33 | 987 / 31 |
| Old MZ61 / DROP_CLOSE | 858 / 38 | 832 / 38 |
| Old MZ61 / closest survives in slot 0 | 987 / 33 | 988 / 31 |
| Old MZ61 / farthest survives in slot 1 | 900 / 123 | 881 / 106 |
| New MZ67 / IDEAL | 769 / 48 | 861 / 46 |
| New MZ67 / MERGE_CLOSE | 804 / 57 | 911 / 57 |
| New MZ67 / DROP_CLOSE | 664 / 71 | 753 / 71 |
| New MZ67 / closest survives in slot 0 | 805 / 57 | 911 / 57 |
| New MZ67 / farthest survives in slot 1 | 723 / 97 | 760 / 83 |

Keeping one near return instead of deleting both produces materially different
decisions with the same weights. New-source DIVERSE HEAD_FAR changes from
158 TP / 19 FP under DROP to 217 / 19 under CLOSEST: 65 positive bits gained,
6 lost; 8 false bits added and 8 removed. Of the 65 gains, 23 already fire through
the new profile's OLD_NEG path; the remaining 42 are head-only beyond that path,
with 33 native winner cells and 9 locally UNKNOWN cells. Saved winner lookup is
descriptive evidence, not a recomputed argmax or branch-causal explanation.

The FAR profile has different costs. New-source DIVERSE HEAD_NEAR becomes
246 / 32 versus DROP 223 / 15; HEAD_FAR becomes 95 / 28 versus 158 / 19.
Thus its total gain of 7 TP comes with 12 additional FP and hides a large
far-head loss. On the old source it adds 49 TP and 68 FP versus DROP. Neither
endpoint should be selected as a hardware simulator merely because its result
is better. The raw profile outputs and all positive/false bit exchanges remain
in the [independent score](../../../../artifacts.local/work/mz74-return-survival-20260911/score-v1/report.md).

## Why the far-return result needs a packing control

The follow-up TRAIN-only census found zero slot-1-only zones in either MZ70
arm's 65,536 actual training-frame presentations (4,194,304 zone presentations
per arm). The original source pools and fixed availability retrieval bank also
contain none. In contrast, FAR creates slot-1-only zones in 1,993/2,048 old-source
TRAIN frames and 1,813/2,048 new-source TRAIN frames.

The complete predictor is slot-sensitive: ordered packet channels enter linear
and nonlinear layers in OLD_NEG and AnchorQuery, and earlier branches also use
flattened packet coordinates or echo identity. Permutation-invariant eligibility
alone does not make the whole predictor invariant. The observed difference
therefore combines changed surviving distance with an unfamiliar representation.
It cannot yet be attributed solely to loss of the near return.

The necessary next control is to move the same surviving FAR value into slot 0,
without changing any measured-distance multiset, RGB, model or cutoff, and compare
against this frozen FAR result. Keep MZ74 immutable and register that control
separately. [Slot census and code evidence](../../../../artifacts.local/work/mz74-return-survival-20260911/slot-audit-v1/report.md).

## Execution and verification

One CUDA run processed all 8,192 frames under both new profiles, with 32
additional TRAIN parity frames across the three original profiles. The three
RGB feature views were shared across profiles and heads: 8,224 RGB decodes,
not a separate image load for each model/profile. All parity decision signs
and discrete outputs matched; float outputs used the declared 2e-5 / 1e-6
tolerance. No new fit, cutoff, dense feature cache or native-depth read.

Run time was 153.116 s (159.953 s including the command wrapper). RGB decoding
took 62.524 s, shared visual features 56.714 s and readouts 25.094 s. Decoding
is a concrete remaining throughput cost; the timing does not prove a speedup
from an unimplemented prefetcher. Independent CPU scoring took 14.719 s and
checked 2,097,152 packet slots, 262,144 candidate decisions, 1,310,720 scalar
metric bits, 432 unchanged historical metric tables and 327,680 saved native
winner lookups. Predictor and scorer processes exited; archive handles closed.

| Evidence | SHA-256 |
| --- | --- |
| Run receipt | c2a67b4c7326f9beb9f30d12552c7ab493445b6c4053c00edfa577da40baf79d |
| Score receipt | 5e720c247bb3df8ec511b2d6ad23f3c4eedae83caf69185a138d796eed22b1bb |
| Score result | c0c8dce450e80c83b61d73d1d23ec6d890f4f29ba3d4deec8bde90a15fb7844d |
| Paired event IDs | c3d29e375e10d7c5773d748cd21be64689127d85d6785389a9388c3887f9d746 |
| TRAIN slot census receipt | fb6d8e43ceab19f77af6c88662e7351b296307f5920a2507217c5154b9c6516f |

[Protocol](MZ74_RETURN_SURVIVAL_20260911.md), [runner](mz74_return_survival.py),
[scorer](mz74_score.py), [command exit/release](../../../../artifacts.local/work/mz74-return-survival-20260911/execution-v1.json),
[score exit/release](../../../../artifacts.local/work/mz74-return-survival-20260911/score-execution-v1.json).

## Independent source repair work

The secondary worker completed a read-only four-asset LOD0 surface export:
wood 8,726 triangles, landing frame 1,937, duct 1,134 and birch 28,438. A faulty
bulk triangle-vertex API returned six IDs; the task-owned corrected probe read
each triangle corner through the original vertex-instance mapping. Failed
attempts are retained. The successful editor job took 60.809 s, the metadata
probe itself 1.345 s, and the returned compressed evidence was 897,758 bytes.
All original asset hashes were unchanged; 29 returned files were verified and
owned UE/Zen processes exited.

These are actual editor source triangles, not new rendered samples or proof of
physical support/native visibility. They support subsequent real-surface
placement and contact checks; MZ72's failed 64-frame source remains unadmitted.
No new source entered MZ74. [Asset-probe evidence](../../../../artifacts.local/work/mz74-return-survival-20260911/asset-probe-v1/REPORT.md).
