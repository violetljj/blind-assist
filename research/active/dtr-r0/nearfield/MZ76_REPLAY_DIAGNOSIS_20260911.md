# Frozen replay repaired; simulated baseline comparison

The user resumed the paused diagnosis, then explicitly chose simulation only.
This engineering replay changes no model, cutoff, training schedule or historical
result. MZ76 remains `INVALID_FOR_REQUESTED_COMPARISON`; neither new MZ76 head
is promoted. No successor training or source collection was started.

## Cause and repair

`fixed_batch_dense` repeats the final image when fewer than16 images remain.
Its `torch.cat` with the expanded padding changes channels-last RGB to contiguous
NCHW. An unpadded16-image call stays channels-last. MZ70's feature cache skipped
TRAIN hits before encoding its remaining rows; every tested HELD frame belonged
to a short historical encoder batch (4–12 images). MZ76 grouped HELD rows into
full16 batches instead. Fixing batch size alone therefore did not fix encoder
memory layout or numerical execution.

The32-frame diagnostic deliberately selected each source's block with the largest
saved raw discrepancy. Changing layout **after** feature extraction did not
repair the error. Restoring historical encoder groups did. Explicitly applying
`rgb.contiguous()` **before** the frozen encoder then reproduced those restored
normalized features bit-exactly, without restoring the grouping. This isolates
encoder input layout from batch companions on this diagnostic subset; no specific
cuDNN kernel is identified. The recorded TF32 settings remain unchanged.

| Diagnostic source | Original raw maximum error | Explicit encoder NCHW |
|---|---:|---:|
| MZ61 selected16 | 0.01165068 | Within original tolerance |
| MZ67 selected16 | 0.01857281 | Within original tolerance |

The separate full verification covers all2,048 exact MZ76 HELD frames under
IDEAL, MERGE_CLOSE and DROP_CLOSE. All42 array comparisons pass the original
`atol=2e-5, rtol=1e-6`: raw maximum error3.8146973e-6, candidate maximum
error1.4305115e-6, zero candidate sign changes. MZ37, support, anchor availability,
anchor vectors and every winner index match exactly. Checkpoint parameter hashes,
packet bytes and frame identities remain fixed. This is original MZ70 replay,
not a rerun or retrospective approval of MZ76's trained comparisons.

The explicit layout is opt-in in [the replay helper](mz76_replay_diagnostic.py)
and [verification runner](mz76_replay_verify.py). Frozen historical runners are
unchanged. Do not globally impose this layout on other frozen caches: those
with unpadded encoder batches may have a different historical contract.

## Same-input simulated baselines

The verified replay also exports the unchanged learned RGB and ToF branches and
their fixed half-logit ensemble MZ5. The scorer binds these to sealed MZ70
MZ37/OLD_NEG/DIVERSE outputs, checks MZ37 compatibility and exact MZ5 arithmetic,
then independently recounts147,456 query bits. No thresholds are selected.

On MZ67 DROP_CLOSE (1,024 frames;1,024 positive and3,072 negative query bits):

| Method | TP | FP | FN | HEAD_NEAR TP / FP |
|---|---:|---:|---:|---:|
| Frozen RGB branch | 90 | 139 | 934 | 22 / 50 |
| Frozen ToF branch | 355 | 155 | 669 | 59 / 15 |
| Fixed half-logit MZ5 | 89 | 78 | 935 | 19 / 19 |
| OLD_NEG union | 562 | 71 | 462 | 158 / 15 |
| Original MZ70 DIVERSE union | 753 | 71 | 271 | 223 / 15 |

Simple averaging is not a sufficient fallback on this source. Keep original
DIVERSE as the existing scoped challenger alongside OLD_NEG for subsequent
simulation work. Its DROP head-near gain is65 true query bits with unchanged15
false bits; this is a comparison to OLD_NEG, not a new gain over MZ70 or evidence
for the MZ76 neighbor features. IDEAL still trades recall for errors: MZ67
667TP/44FP to861/46, MZ61 865/70 to971/78. No universal baseline promotion follows.

Evaluator query UNKNOWN is zero on these selected rows, but ToF coverage is
incomplete. Zero-valid-zone frames for IDEAL/MERGE/DROP are5/5/23 on MZ61 and
9/9/59 on MZ67. All remaining frames have only partial zone coverage. Preserve
these measurement gaps; missing measurements are not clearance. The profiles
are simulated sensitivity inputs, not measured sensor physics. This replay adds
no temporal, natural-scene, hardware or safety evidence; end-to-end real-time
latency is not established.

## Execution and evidence

- Full verification: CUDA, RTX5060 Laptop,56.011s,2,048 RGB decodes shared across
  methods/profiles. Decode17.344s, encoders17.157s, readouts7.588s; these are
  batched diagnostic timings, not per-frame device latency.
- Scalar baseline scorer: CPU (`TASK_NOT_GPU_SUITABLE`),0.475s. Processes exited0,
  image/archive handles closed, no persistent feature cache or worker allocation.
- All checkpoint/source/prediction inputs and output archives are hash-bound.
  Original MZ76 failure and both trained checkpoints remain intact.

[Causal probe](../../../../artifacts.local/work/mz76-replay-diagnostic-20260911/probe-v3/result.json),
[full verification](../../../../artifacts.local/work/mz76-replay-diagnostic-20260911/verify-v1/result.json),
[layout reproducer](../../../../artifacts.local/work/mz76-replay-diagnostic-20260911/layout-contract.json),
[all baseline tables](../../../../artifacts.local/work/mz76-simulated-baselines-20260911/analysis-v1/report.md),
[scalar receipt](../../../../artifacts.local/work/mz76-simulated-baselines-20260911/analysis-v1/receipt.json).

Probe-v1 is retained as exploratory evidence with its original `source.py`
snapshot. Its CLOSEST-versus-MERGE comparison used unequal profile semantics and
is excluded from conclusions; probe-v2/v3 removed that comparison. Probe-v2
tested historical grouping; probe-v3 isolated encoder input layout. None is a
fresh confirmation cohort. The engineering replay introduces no new research
terminal and does not alter historical inheritance roles.
