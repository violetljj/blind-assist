# Metric-depth successors R0

This directory implements the two successors frozen after the asynchronous affine R1:

- dense Metric3D residual propagation with bidirectional RAFT consistency and DA new-region fill;
- offline Metric3D teacher distillation into a 770-parameter DA layer-11 CLS calibration head.

Both use the hash-bound consumed TUM cache and remain Development-only. Run focused tests with:

```powershell
E:\codex-tools\bin\blindassist-python.cmd -m unittest discover `
  -s scripts/research/metric_depth_successors_r0 -p "test_*.py" -v
```

The materializer and evaluators refuse to overwrite outputs. Exact commands and evidence paths are
recorded in the result documents after execution.

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
