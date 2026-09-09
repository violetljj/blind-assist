# MZ5: fixed unimodal ensemble and compact readout

2026-09-10 EXPLORE plus engineering export. **The fixed equal-logit ensemble
reaches 1378/1500 exact (91.87%), versus MZ1 fusion's 1353/1500 (90.20%).**
Wrong-far falls12->4 and cross-body19->8. Retain its compact implementation as
the next controlled learned-readout baseline, with reduced far recall explicit.
MZ0 remains an information component; original alerts and default-App behavior
are unchanged. The [matched spatial experiment](MZ5_SPATIAL_FUSION_RESULTS_20260910.md)
does not supply a stronger practical baseline.

## Fixed method and evidence identity

The [literature note](idea.md) specified this comparison before execution
(Git d3ccf205f049c9ff620559b19cdc5846d11b507b). The diagnostic saved its
[protocol](../../../../artifacts.local/work/mz5-fixed-ensemble-20260910/run-v1/protocol.json)
before loading and combining predictions: exactly
`0.5 * (RGB_ONLY_logits + TOF_ONLY_logits)`, event threshold `>=0`.
No learned weights, threshold search, frame router or oracle branch selection.

[mz5_fixed_ensemble.py](mz5_fixed_ensemble.py) uses the saved final MZ1 unimodal
outputs on all1500 already consumed EVAL_ONLY frames. No model forward or fit
was needed for the scientific diagnostic. The original MZ1 feature/role/truth/
checkpoint hashes and all nine split/arm metric groups were verified; MZ0/MZ3
aggregates and5000 original alerts also match their preserved evidence.

Inputs are the controlled body-query5000 source, frozen RGB representation and
generic45-degree8x8 two-return radial ToF simulation. These are not new natural
observations or device measurements. False event flags do not establish CLEAR.

## Effect and tradeoff

Exact requires all four BODY/HEAD x near/far flags to match. Wrong-far counts
far assertions on600 body/head near-only opportunities. Cross-body includes
BODY->HEAD and HEAD->BODY confusion, with300 opportunities each.

| Readout | Exact /1500 | Wrong-far /600 | BODY->HEAD /300 | HEAD->BODY /300 | HEAD_ONLY near exact /150 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fixed ensemble | 1378 (91.87%) | 4 | 2 | 6 | 137 |
| MZ1 FUSION | 1353 (90.20%) | 12 | 7 | 12 | 133 |
| MZ1 TOF_ONLY | 1327 (88.47%) | 9 | 11 | 8 | 142 |
| MZ1 RGB_ONLY | 1310 (87.33%) | 17 | 9 | 12 | 129 |
| MZ0 retained component | 1310 (87.33%) | 7 | 0 | 15 | 131 |

Against MZ1 fusion, ensemble gains52 exact rows and loses27. It removes8
wrong-far and11 cross-body errors while introducing0 of those two error types.
Negative-control exact improves544->578 of600 (+34/-0 paired); HEAD_ONLY near
improves133->137 of150 (+5/-1). Against MZ0, gains142/losses74 remain substantial
in both directions. Against ToF-only, gains103/losses52 likewise rule out uniform
dominance. All row identities and condition/region strata remain in the result.

The1192 rows where both unimodal readouts were correct are all preserved, versus
1180 preserved by MZ1 fusion. For same-sign event logits, their positive-weight
mean necessarily retains the sign; this agreement property is an algebraic
advantage of this rule, not learned evidence of confidence calibration.

Far recall decreases versus MZ1 fusion: BODY_FAR true positives272->257/300
and HEAD_FAR270->255/300. HEAD_NEAR true positives285->284/300; BODY_NEAR stays
290/300. The all-event exact gain includes fewer false positives and does not
justify claiming a uniformly safer detector or replacing the retained alert path.
The HEAD_ONLY near subset also remains below ToF-only (137 versus142).

## Callable compact implementation

[mz5_ensemble_readout.py](mz5_ensemble_readout.py) exports the same function as
RGB772->128->4 and ToF256->128->4 branches with a fixed logit mean. It removes
only first-layer columns whose corresponding unimodal inputs were always zero;
active weights, biases, ReLUs and output layers are preserved. No retraining.
The frozen RGB encoder/features can be shared once.

| Readout layout | Parameters | Dense MACs/frame |
| --- | ---: | ---: |
| Original stored two-head ensemble | 264456 | 264192 |
| Compact ensemble | 132872 | 132608 |
| Original single fusion head | 132228 | 132096 |

Compact parameters are RGB99460 + ToF33412, only644 more than the single fusion
head. This is a similar weight/MAC budget, not an exactly matched architecture
or proof of equal latency. The diagnostic's original two-head cost record is
preserved; this later engineering export is recorded separately in compact-v1.

```python
from mz5_ensemble_readout import CompactEnsemble

model = CompactEnsemble.from_checkpoint(checkpoint_path)
logits = model(visual, tof)  # [B,772], [B,256], unchanged MZ1 normalization/order
event_flags = logits >= 0  # original B alerts remain a separate output
```

Full5000-row CPU and CUDA checks reproduce every branch and ensemble sign,
including all1500 scientific EVAL predictions. Ensemble maximum absolute logit
error is3.81e-6 on CPU and2.86e-6 on CUDA, versus minimum original threshold
margin6.63e-4. This is numerical agreement with identical decisions, not bitwise
floating-point equality. The two compact branches themselves also have zero
sign mismatches. Exported weights and5000 original alerts match exactly.

Actual warm readout timings, milliseconds per batch:

| Readout | CPU batch1 | CUDA batch1 | CPU batch128 | CUDA batch128 |
| --- | ---: | ---: | ---: | ---: |
| Compact ensemble | 0.04435 | 0.17840 | 0.33405 | 0.18290 |
| Original two-head ensemble | 0.07145 | 0.21980 | 0.66260 | 0.26520 |
| Original single fusion head | 0.02470 | 0.09705 | 0.31475 | 0.09945 |

These are50-sample medians after10 warmups through `tools/research_backend`,
CPU single-thread and RTX5060 Laptop CUDA with synchronization, resident inputs.
They exclude RGB extraction, ToF construction and transfers. CPU is faster for
this isolated batch1 workload and CUDA for batch128; whole-pipeline placement
and Android latency remain unmeasured. Compact is faster than the original
two-head layout, but slower than the single fusion head in these timings.

## Evidence and completion

- [Scientific result](../../../../artifacts.local/work/mz5-fixed-ensemble-20260910/run-v1/result.json),
  [scalar audit](../../../../artifacts.local/work/mz5-fixed-ensemble-20260910/run-v1/independent-audit.json),
  [input/output receipt](../../../../artifacts.local/work/mz5-fixed-ensemble-20260910/run-v1/receipt.json).
- [Compact checkpoint](../../../../artifacts.local/work/mz5-fixed-ensemble-20260910/compact-v1/compact.pt),
  [export receipt](../../../../artifacts.local/work/mz5-fixed-ensemble-20260910/compact-v1/export-receipt.json),
  [full parity](../../../../artifacts.local/work/mz5-fixed-ensemble-20260910/compact-v1/parity-receipt.json),
  [benchmark summary](../../../../artifacts.local/work/mz5-fixed-ensemble-20260910/compact-v1/benchmark-summary.json),
  [engineering receipt](../../../../artifacts.local/work/mz5-fixed-ensemble-20260910/compact-v1/engineering-receipt.json).

The scientific diagnostic took4.67s on CPU. Its saved-array combination timing
is not model latency. A redirected stdout log was finalized after the engineering
receipt was first written; both receipt versions and a separate final-log hash
are preserved. This bookkeeping correction reran no model and changed no result.

Reproduction entry points are the two linked scripts: diagnostic `--root <repo>
--output <fresh-run>`, compact `--root <repo> --output <fresh-compact> --phase
prepare`, then `--phase validate-benchmark` on that compact directory. CLI output
directories must be new children of the canonical MZ5 artifact root. The local
source/cache/checkpoints are required; payloads are not redistributed in Git.

The fixed diagnostic, export and verification are complete. Carry the explicit
far-recall/HEAD_ONLY tradeoffs into the next distinct experiment. No weights or
thresholds were selected from the consumed result, no additional fit was opened,
and no firmware, temporal, device or safety performance is established here.
