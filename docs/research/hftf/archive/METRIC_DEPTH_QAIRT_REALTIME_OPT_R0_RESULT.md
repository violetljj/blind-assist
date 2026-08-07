# Metric depth QAIRT real-time optimization R0 result

Terminals:

- `VITS_392X672_RAFT2_HTP_DEPLOYMENT_PARITY_SUPPORTED_CONSUMED_CANARY_ONLY`
- `VITS_LOW_LATENCY_LIVE_TARGET_NOT_MET`
- `POST_TRAINING_QUANTIZATION_NUMERICALLY_INVALID_OR_SLOWER_STOP`
- `CONVTINY_HTP_DEPLOYMENT_PARITY_SUPPORTED_SOURCE_ACCURACY_NOT_SUPPORTED_CONSUMED_BONN_RGBD_CANARY`

Decision:

`KEEP_VITS_392X672_RAFT2_CACHED_AS_CONTINUITY_CANDIDATE /
STOP_CONVTINY_V1_T_SOURCE_CANDIDATE /
STOP_CURRENT_PTQ_SEARCH /
NO_RESEARCH_MAINLINE_OR_DEFAULT_APP_CHANGE`

## Scope

The device optimization used only the nine already-consumed
`prepared-tokyo-smoke` technical frames, including the existing consecutive
seven-frame window. It did not open fresh data, search an alert operating
point, or evaluate safety utility. All executions used QAIRT Community
`2.47.0.260601` with `libQnnHtp.so` as the only backend on the existing
`SM-S9280 / SM8650 / Android 16` device. Profiles reported four HVX threads,
RPC time, and accelerator time.

The canonical comparison remains Metric3Dv2 ViT-S FP32 ORT at `616x1064`.
Agreement with that source is deployment/continuity evidence, not real metric
truth.

The previously unresolved ConvNeXt-Tiny source disagreement was then tested on
the separate, already-consumed 30-frame Bonn person-tracking canary. Its frozen
manifest supplies registered RGB-D sensor depth that was not model input, the
same torso ROIs and camera intrinsics, and already-produced ViT-S results. No
fresh cohort or tuning was opened.

## ViT-S resolution and early-exit results

The official dynamic-shape ONNX was specialized at `392x672`, with both
dimensions divisible by 28. The full four-iteration graph fell from
`138.605 G` to `56.303 G` MACs and from `1500.794 ms` to `479.115 ms` HTP
execute. HTP versus same-resolution ORT had a `0.0613 m` one-second D44 future
position difference. The resolution change itself was material: the HTP
candidate differed from canonical ORT by `0.4937 m` at that future position.

The four unrolled RAFT updates were then cut after iteration one or two. The
selected iteration's hidden state was passed through the original learned
upsampling mask, and the graph was pruned to depth-only output before QAIRT
conversion.

| Arm | MACs | Mean HTP execute | HTP vs same-arm ORT D44 future | HTP vs canonical ORT D44 future |
| --- | ---: | ---: | ---: | ---: |
| ViT-S 4-iter `616x1064` | `138.605 G` | `1500.794 ms` | `0.0827 m` deployment difference | reference source |
| ViT-S 4-iter `392x672` | `56.303 G` | `479.115 ms` | `0.0613 m` | `0.4937 m` |
| ViT-S RAFT-1 `392x672` | `40.015 G` | `415.645 ms` | `0.0757 m` | `0.5081 m` |
| ViT-S RAFT-2 `392x672` | `45.443 G` | `428.058 ms` | `0.0738 m` | `0.4842 m` |

RAFT-2 is the continuity choice. It costs only `12.413 ms` more than RAFT-1,
but has lower mean person-torso difference versus canonical ORT
(`0.4134 m` versus `0.4724 m`). It is still only about `2.34 fps`; therefore it
does not meet a low-latency per-frame live depth target.

The retained RAFT-2 online DLC is SHA-256
`6DFF1274E094B41A7CA9869C22B0DE82CF3328DA15A8FA9CC31FBFF687CD862D`.
Its target-generated SM8650 cached DLC is SHA-256
`6C6CAF0DBB7C35929F0D3BEA32A262FCF0A5C74B8A82464C8CD0B4F95AC377BA`.
A cached one-frame process reported `73.481 ms` init and `22.195 ms` graph
record load; its output was byte-identical to the online result.

## Quantization stop

Post-training quantization used exactly the same nine consumed inputs as
calibration data. Full INT8 executed at `284.913 ms`, but destroyed the metric
scale: person-torso depth differed from same-resolution ORT by `31.666 m` on
average and the D44 future position differed by `31.573 m`.

W8A16 was both slower (`888.716 ms`) and less faithful than FP16: its D44
future position differed from same-resolution ORT by `0.2459 m` and from
canonical ORT by `0.8009 m`. These two bounded arms are enough to stop the
current PTQ search; calibration/schema tuning is not authorized as a rescue.

## ConvNeXt-Tiny result

The official Metric3D ConvNeXt-Tiny v1 weight was exported to a depth-only
`544x1216` ONNX and converted to HTP FP16. The model has `83.359 G` MACs and
executed in `434.893 ms` mean (`414.150–441.897 ms`). Deployment parity was
strong:

- full-map depth relative MAE versus its own ORT was `2.167%` mean;
- person-torso median depth difference was `0.0030 m` mean and `0.0061 m` max;
- seven-frame D44 future position difference was `0.00793 m`.

However, ConvNeXt-Tiny HTP and canonical ViT-S ORT differed by `2.643 m` mean
person-torso depth and `2.813 m` at the D44 future position. The official tiny
checkpoint is documented as outdoor-only, so model-to-model disagreement alone
did not initially resolve source accuracy.

The consumed Bonn RGB-D canary does resolve that narrow question. Using exactly
the frozen 30 frames, torso ROIs, intrinsics, registered sensor-depth truth,
10%-90% trimmed torso median, and each model's fixed official input geometry:

| Consumed Bonn paired measure | ConvNeXt-Tiny v1-T | ViT-S |
| --- | ---: | ---: |
| Frame torso depth MAE | `1.1025 m` | `0.03816 m` |
| Frame torso mean relative absolute error | `59.13%` | `2.052%` |
| Paired frame wins | `0/30` | `30/30` |
| Seven-frame to one-second future position mean error | `1.0967 m` | `0.4362 m` |
| Paired future-window wins | `0/14` | `14/14` |

The scale failure is not an export artifact. The official PyTorch model and
the exported ONNX produced `0.73282546 m` and `0.73282579 m` respectively on
the first Bonn torso, whose registered truth was `1.85300004 m`; checkpoint
loading had no missing keys. Therefore deployment parity remains supported,
but ConvNeXt-Tiny v1-T source accuracy is not supported on this consumed
indoor person-tracking canary and the candidate is stopped without a tuning
rescue.

The online DLC is SHA-256
`F51B9ABF9F53865FFFA14481BBFE5DC6409AF4ECCB9D18C659BA9B1DCCEE7553`;
the target-generated cached DLC is SHA-256
`D9DF28944DF011D6B6A57530538175762E5D9781903579559D4E977248C0AAEB`.
A cached one-frame process reported `77.250 ms` init and `18.847 ms` graph
record load, with byte-identical output to online execution.

Both cached runs selected a compatible `HTP_V75_SM8650_4MB` record and also
emitted a VTCM/DSP-architecture mismatch warning. The output identity and HTP
profile establish that the cache executed, but the warning remains part of the
deployment evidence and must not be suppressed.

## Interpretation and next admissible step

This R0 retains one deployment-continuity candidate but not a promoted live
source:

- ViT-S RAFT-2 preserves continuity with the existing teacher better but is
  still too slow and inherits a resolution-induced shift.
- ConvNeXt-Tiny provides more pixels at similar latency and excellent
  deployment parity, but its absolute metric scale failed the consumed Bonn
  RGB-D paired diagnostic and the v1-T source candidate is stopped.

Any later source candidate still requires independent meter-level truth from
the final external camera before promotion. The single consumed Bonn sequence
does not establish cross-camera generalization, live utility, or alert safety.
The retained ViT-S RAFT-2 candidate may not feed alerts or replace the research
mainline/default App.

Machine-readable result:
[METRIC_DEPTH_QAIRT_REALTIME_OPT_R0_RESULT.json](METRIC_DEPTH_QAIRT_REALTIME_OPT_R0_RESULT.json).
