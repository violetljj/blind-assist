# DEPTHART_ADMISSION_R1 A3：ONNX/QNN deployment preflight

状态：`NOT_EVALUABLE / DEPLOYMENT_PREFLIGHT_BLOCKED`

本轮只检查导出链路和工具可用性，不产生 ONNX parity、QNN 转换、HTP 执行、Android 或生产证据。R0 结论与数据角色不变。

## 观察

- PyTorch 2.11 新 exporter 完成 graph capture，但在 translate 阶段因 `depthart.selective_scan` 没有可用 ONNX function 而停止。
- 强制 legacy exporter 后，官方路径依赖 `depthart_selective_scan_cuda`，本机未安装，因此没有生成有效 ONNX graph。
- 本机 PATH 与 `E:\codex-tools` 检查均未找到 `qnn-onnx-converter`、`qnn-net-run`、`qairt-converter` 或 `qnn-context-binary-generator`。

机器可读 receipt：[`a3-onnx-qnn-preflight.json`](../../../artifacts.local/evidence/hftf/depthart-admission-r1/a3-onnx-qnn-preflight.json)

## 结论与边界

A3 当前是 `NOT_EVALUABLE`，不是模型质量 FAIL，也不是 HTP 不支持。下一步只能在取得官方 Selective Scan CUDA extension 和 QAIRT/QNN SDK 后，按同一冻结合同重跑；不得用 CPU reference、伪造 custom-op graph 或 DA2 的 HTP 结果替代。
