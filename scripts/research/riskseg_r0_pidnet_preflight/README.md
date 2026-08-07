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

状态：`development`

## 稳定 Interface

公开入口、输入不变量和失败模式以本目录脚本帮助和专项协议为准；跨域调用不得依赖私有 Implementation。

## 输出

只写入 artifacts.local/ 下的明确证据目录；不写仓库根目录或正式 App 资产。

## 安全边界

本模块不产生默认 App、生产、安全或 unseen confirmation authority；结果按当前协议声明的 Development/diagnostic 角色使用。

## 停止条件

最小判别实验完成、输入权威缺失、预算耗尽或重复失败时停止当前 evidence version，并保持最小 failure scope。

## 产物边界

运行产物必须位于 artifacts.local/，不提交数据集、模型、设备日志或大文件。
