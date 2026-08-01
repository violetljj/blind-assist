# RISKSEG-R0 PIDNet-S technical preflight

This package implements the frozen `512x288`, four-class PIDNet-S deployment
surface. It does not read event-eval outcomes and does not authorize training
until both TFLite and strict QNN device gates pass.

The ignored local inputs are:

- official `XuJiacong/PIDNet` source pinned by commit;
- `PIDNet_S_ImageNet.pth.tar`, with source and hashes recorded in the export
  receipt;
- the frozen 520-frame RISKSEG-R0 re-encoded train/dev view.

The deterministic preflight flow is:

1. build a four-class PIDNet-S from the official implementation and load only
   shape-compatible ImageNet tensors;
2. export normalized NCHW RGB to full-resolution NCHW logits as ONNX;
3. select quantization calibration RGB only from the frozen train role;
4. convert to full-integer W8A8 TFLite with signed INT8 input/output;
5. validate NHWC tensor layout, quantization, finite dequantized outputs and
   argmax class range before any device run;
6. run the TFLite artifact with the strict Qualcomm QNN HTP delegate on the
   single locked SM-S9280 and capture latency, fallback and thermal evidence.

The formal device receipt and bounded logcat are independently checked with
`python -m scripts.research.riskseg_r0_pidnet_preflight.validate_device_preflight`.
