# MZ81 conditional RGB feature probe

Decision: `PAUSE_SINGLE_FRAME_RGB_RESIDUAL`.

The frozen RGB representation does not provide selectively transferable
collision-query evidence on this source-disjoint Development probe. A tiny head
can fit and partially transfer between the two controlled still-image source
families, but its ranking collapses on the independent MZ77 approach source.
At added FP <=5, even a forbidden posthoc oracle threshold rescues 0 TP in all
three 5-klux profiles. This meets the predeclared stop condition; do not scale
the single-frame RGB residual network.

## Fixed design

- Frozen input: the current normalized 64x45x80 RGB backbone tensor.
- Only learned change: adaptive-average pooling to 3x3 plus a 576-8-4 MLP,
  4,652 parameters. BODY-near/far and HEAD-near/far are separate outputs.
- Train source: all 4,096 MZ61 frames (old geometry families).
- Selection source: all 4,096 MZ67 frames (disjoint topology/material families).
- Test source: 160 MZ77 approach frames, opened for labels only after epoch and
  global cutoff selection.
- Negatives: each train/selection source contains 1,024 visible-nonintruding
  frames, 1,024 fully collision-negative frames and 8,192 query-specific
  wrong BODY/HEAD or near/far negative bits. MZ77 pre-onset frames remain unseen
  transfer tests, rather than being consumed for fitting.

The MZ67 FP budget is 120 bits, scaled from the MZ77 target of 5 FP by the
respective known-negative counts (12,288 versus 510). The chosen epoch is 365;
MZ67 gives 482 TP / 120 FP and PR-AUC 0.519. The backbone and all ToF parameters
remain byte-identical and MZ80's observability gate is unchanged.

## Source-disjoint result

| MZ79 profile | ToF TP/FP/FN | Locked MZ67 cutoff increment | Conditional PR-AUC | Posthoc best at added FP <=5 |
| --- | --- | --- | ---: | --- |
| 5 klux, 88% typical | 42/0/88 | +8 TP / +211 FP | 0.104 | **0 TP / 0 FP** |
| 5 klux, 54% typical | 32/0/98 | +15 / +211 | 0.120 | **0 / 0** |
| 5 klux, 17% typical | 20/0/110 | +24 / +211 | 0.139 | **0 / 0** |

Overall MZ77 query-bit PR-AUC is 0.178. The locked cutoff's 211 added FP shows
a severe source/calibration shift. More importantly, this is not merely a bad
cutoff: on every profile, every MZ77 score threshold that rescues any missed TP
also exceeds five added FP. Therefore the posthoc low-FP ceiling is zero.

The CLEAN invariant remains exact by construction because `U_ToF` is false for
all four queries under the ideal 4 m profile; the new head cannot speak there.

## Interpretation and stop

MZ81 does not prove that RGB, all visual backbones, or temporal vision are
incapable of collision prediction. It rejects this narrower route: a 4.7k head
on fixed 3x3-pooled features from the current single-frame RGB backbone does not
transfer selective rescue to the posed approach source. MZ61/MZ67 contain
controlled query negatives but not a separate training source with natural
approach/pre-onset dynamics, which is part of the observed domain gap.

Per the stopping rule, pause single-frame RGB residual scaling and preserve this
probe as a negative control. A future restart must change the information source
or representation rather than merely increase readout size: temporal RGB,
explicit perspective/corridor geometry, dual or higher-resolution ToF, ToF+IMU,
or radar are admissible new hypotheses. No successor is started here.

## Evidence and verification

Canonical evidence is
`artifacts.local/work/mz81-conditional-feature-probe-20260912/run-v2/`.
It contains frozen pooled features, the selected probe, MZ77 scores, result,
release record, source snapshots and a hash receipt. The run used CUDA on an
NVIDIA GeForce RTX 5060 Laptop GPU, processed 8,352 frames, and took 155.6 s.
Python compilation, two direct helper checks, frozen-backbone identity and
source-role assertions passed. The environment lacks pytest, so the two helper
tests were executed directly. This is consumed controlled-simulation
Development evidence, not natural-scene, measured-hardware or safety evidence.
