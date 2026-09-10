# MZ11 selective addition results

The 20-parameter gate recovers missed positives without adding DEV or sequence
false positives, but fails the predeclared thin-retention gate: clean46/49 and
stress44/49 versus required48/49. Stop this fit; no threshold rescue, second fit,
new capture, temporal work or baseline promotion. Keep MZ5 and the MZ9 component.
Retain this completed recipe as a scoped negative control with useful measured
recall gains, not an admitted replacement or confirmation candidate.

## Change and evidence

[Protocol](MZ11_SELECTIVE_ADDITION_PROTOCOL_20260910.md): four independent linear
gates read frozen SOURCE margin and candidate hypothesis/zone/return counts.
One600-step fit, seed111, uses old TRAIN2500 plus consumed MZ6 training200.
All original MZ5 positive bits are immutable. Only supported SOURCE-positive,
MZ5-negative bits can be added. No native depth, target identity or pose enters
the gate. The20 parameters are additional to the existing heads and backbone,
not the size of the complete algorithm.

The effective eligible training counts are small: old TRAIN positive0/12/2/13,
negative7/21/20/2; sequence positive7/29/0/38, negative0/0/0/0, in BODY_NEAR,
BODY_FAR, HEAD_NEAR, HEAD_FAR order. Full-batch fitting balances nonempty
cohort/query/class groups; it does not create missing negative examples. This
limits confidence in transfer, especially from a single trained thin-pole layout.

DEV chooses cutoffs above every false-addition score. Therefore zero added DEV
FP is calibrated, not independent validation. Inputs were also chosen after the
MZ10 DEV audit. Sequence frames are training regression. No unseen obstacle
configuration was evaluated, and49 thin bits are adjacent opportunities in one
25-frame clip, not49 independent events.
The same cutoffs add8 FP bits on old TRAIN (4/3/1/0); zero additional error is
specific to the checked DEV/sequence cohorts, not a structural guarantee.

## Complete output, not standalone branch budgets

| Cohort / method | Four correct | TP BN/BF/HN/HF | FP BN/BF/HN/HF |
|---|---:|---|---|
| old DEV / MZ5 | 912/1000 | 187/180/183/165 | 6/11/10/7 |
| old DEV / adapted MZ5 | 917/1000 | 180/180/185/173 | 5/10/7/6 |
| old DEV / MZ9 SOURCE | 899/1000 | 169/189/196/170 | 5/9/10/7 |
| old DEV / MZ10 | 900/1000 | 197/189/196/170 | 11/20/19/14 |
| old DEV / MZ11 | **928/1000** | **195/190/190/172** | **6/11/10/7** |
| sequence clean / MZ5 | 143/200 | 2/6/9/0 | 5/3/19/0 |
| sequence clean / adapted MZ5 | 199/200 | 9/36/9/38 | 0/0/0/1 |
| sequence clean / MZ9 SOURCE | 198/200 | 9/35/9/38 | 0/1/0/0 |
| sequence clean / MZ10 | 177/200 | 9/35/9/38 | 5/1/19/0 |
| sequence clean / MZ11 | **174/200** | **9/35/9/36** | **5/3/19/0** |
| sequence stress / MZ11 | **172/200** | **9/34/9/35** | **5/3/19/0** |

MZ11 adds32 old DEV TP bits (8/10/7/7), including17 far bits, while exact frames
increase16. Sequence clean adds72 TP bits and31 exact frames; stress adds70 bits
and29 exact frames. This is a nontrivial addition result, not a reject-everything
solution. No baseline-positive bit is lost. The zero-add comparator is exactly
original MZ5, and every one of its false positives is necessarily preserved.

In particular, the approaching bar's16 premature HEAD_NEAR activations remain,
as do all19 sequence HEAD_NEAR FPs. MZ11 improves recall; it does not repair
distance attribution or reduce existing false alerts. It is not uniformly better
than MZ9 or adapted MZ5 on sequence exactness. The adapted comparator shares
underlying new examples but not total training steps or auxiliary supervision:
SOURCE had contributor supervision and MZ11 adds600 gate steps. This comparison
cannot isolate a unique spatial-method contribution from equal training effort.

## Thin loss and correspondence control

| Condition | BODY_FAR | HEAD_FAR | Total thin TP |
|---|---:|---:|---:|
| Original MZ5 | 0/25 | 0/24 | 0/49 |
| Frozen MZ9 SOURCE clean/stress | 24/25 | 24/24 | 48/49 |
| MZ11 clean | 24/25 | 22/24 | **46/49** |
| MZ11 stress | 23/25 | 21/24 | **44/49** |
| MZ11 clean/stress wrong RGB | 1/25 | 0/24 | **1/49** |

Wrong correspondence recomputes SOURCE scores and SOURCE-positive eligibility,
then uses the same trained gate and cutoffs. MZ5 and geometry counts stay fixed.
Whole-sequence exact drops174->149; added far TP65->2; added FP stays0. This
supports sensitivity of the frozen chain to correct correspondence. It does not
isolate the causal contribution of the new count features without a matched
ablation, which was not run.

Gate passes no added FP, preservation of all baseline positives, and DEV far
gain. Both clean and stress thin-retention checks fail. The gate rejects two
otherwise available clean thin positives and four stress positives. A two-bit
clean-to-stress loss now exists, but temporal work remains closed: the single
frame candidate has not passed its own retention criterion or new-placement
confirmation. No automatic successor is started.

## Verification and delivery

Artifacts: `artifacts.local/work/mz11-selective-addition-20260910/run-v1/`.
Frozen input hashes, protocol/source identity, training group counts, gate weights,
exact cutoffs, feature/score arrays, all predictions and per-clip results are
retained. CUDA extraction plus fitting/scoring took7.705s on the local GPU; this
is experiment wall time, not online frame latency. No encoder was trained.

The independent audit replays composition, metrics, threshold budgets, rejected
positive counts and wrong-correspondence eligibility across17,200 stored output
bits. Eligible positive rejections are2/4/6/6 on DEV,0/0/0/2 on clean sequence,
and0/1/0/3 under stress. No
additional fit was performed. Processes exit normally; no UE, worker, ports or
other reserved runtime were started. Invalid packets remain missing evidence,
not CLEAR. These synthetic Development results establish no hardware or safety
performance claim.
