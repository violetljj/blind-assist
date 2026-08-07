# Metric depth QAIRT GPU/HTP R0 result

Terminals:

- GPU: `GPU_EXECUTION_ESTABLISHED_NUMERICALLY_INVALID_NONDETERMINISTIC`
- HTP: `HTP_EXECUTION_AND_DEPLOYMENT_PARITY_SUPPORTED_CONSUMED_CANARY_ONLY`

Decision:

`KEEP_HTP_CACHED_METRIC3D_AS_LOW_RATE_OR_OFFLINE_DEPLOYMENT_CANDIDATE /
STOP_CURRENT_GPU_ROUTE /
FULL_616X1064_MODEL_REQUIRES_RESOLUTION_REDUCTION_OR_DISTILLATION_BEFORE_LIVE_D44`

## What was tested

The existing fixed Metric3Dv2-S ONNX was converted with Qualcomm AI Runtime
Community `2.47.0.260601` and executed on the existing Samsung `SM-S9280`,
Android 16, SM8650 device. GPU and HTP were invoked as exclusive QNN backends;
the HTP profile reported four HVX threads plus RPC and accelerator execution
times. This is stronger evidence than NNAPI session creation because neither run
had a CPU fallback backend available.

No fresh RGB or outcome source was opened. Numerical deployment parity used
nine frames from the already-consumed `prepared-tokyo-smoke` technical cohort,
including one consecutive seven-frame window. The comparison authority was the
existing FP32 ONNX Runtime Metric3D source and its frozen preprocessing.

## GPU result

Both FP16 and FP32 GPU DLCs composed and executed, but their final outputs were
not numerically admissible. The same FP32 DLC and input on QNN CPU matched ORT
closely (`0.00203 m` depth MAE), excluding the converter, input layout, and raw
output parser as the primary cause.

Layer probes localized the failure:

- encoder block 11 max absolute error was `0.00188`;
- encoder norm max absolute error was `0.000454`;
- the last checked decoder-front tensor max absolute error was `0.000200`;
- a later decoder tensor was finite in one run but had 27 NaNs and extreme
  values in another diagnostic run;
- disabling GPU queue recording did not fix the result and two runs still had
  up to `120.095 m` depth disagreement.

The GPU route is therefore stopped for this model/SDK/device combination. A
single FP32 execute was about `2.10 s`, but latency of an invalid result is not a
deployment benefit.

## HTP result

After adding the required on-device `libQnnHtpPrepare.so`, the generic HTP FP16
DLC composed, finalized, and executed successfully. Five steady executions had:

| Metric | Result |
| --- | ---: |
| Mean execute | `1500.794 ms` |
| Min execute | `1491.137 ms` |
| Max execute | `1508.819 ms` |
| HVX threads | `4` |

Independent processes produced bit-identical depth, normal, and confidence
outputs. A SM8650 cached-context DLC then reduced graph startup from
`35,777.293 ms` online finalize to `169.908 ms` binary load. Cached and online
outputs were bit-identical for all nine consumed inputs.

On those nine real technical frames, HTP FP16 versus FP32 ORT showed:

- full-map depth relative MAE range `0.950%–1.033%`;
- person-torso median depth difference mean/max `0.0928/0.1179 m`;
- seven-frame relative-position difference mean/max `0.0868/0.1045 m`;
- seven-frame OLS velocity-vector difference `0.00443 m/s`;
- one-second future-position difference `0.0827 m`.

This supports deployment parity for a D44 source experiment. It does not prove
that either FP32 ORT or HTP is correct in real metric space.

## Deployment interpretation

HTP is the first accelerator route here that is both genuine and numerically
usable. It is still only about `0.67 fps` at the canonical `616x1064` input, so
it cannot supply every frame of the current seven-frame live D44 history. The
cached artifact is retained for low-rate shadow/offline experiments and as the
deployment teacher for the next bounded optimization.

The next implementation question is no longer GPU versus NPU. It is whether a
smaller-input or distilled Metric3D source can retain the observed torso-depth
and D44-track parity while reducing HTP execute time materially. No alert,
production, safety, or default-App authority follows from this result.

## Minimal rerun recipe

With QAIRT `2.47.0.260601` activated, the admitted HTP DLC was generated with:

```text
qairt-converter
  --input_network metric3d-vits-616x1064-fixed.onnx
  --target_backend HTP
  --float_bitwidth 16
  --output_path metric3d-vits-616x1064-htp-fp16.dlc
```

Online device composition requires `qnn-net-run`, `libQnnHtp.so`,
`libQnnHtpPrepare.so`, `libQnnModelDlc.so`, `libQnnSystem.so`, the V75 ARM
stub, and the unsigned V75 skeleton in both `LD_LIBRARY_PATH` and
`ADSP_LIBRARY_PATH`. Execute with only `libQnnHtp.so` as the backend.

The cached artifact was created on the target device with:

```text
qnn-context-binary-generator
  --backend ./libQnnHtp.so
  --dlc_path metric3d-vits-616x1064-htp-fp16.dlc
  --output_dlc metric3d-vits-616x1064-htp-fp16-cached.dlc
  --htp_socs sm8650
  --strip_output_dlc
```

The converter's `--target_soc_model SM8650` validator rejected that identifier
in this SDK even though its bundled HTP configuration maps SM8650 to V75. The
generic HTP conversion plus target-device cached-context generation was used;
the exclusive HTP runtime profile is the execution authority.

Machine-readable result:
[METRIC_DEPTH_QAIRT_GPU_HTP_R0_RESULT.json](METRIC_DEPTH_QAIRT_GPU_HTP_R0_RESULT.json).
