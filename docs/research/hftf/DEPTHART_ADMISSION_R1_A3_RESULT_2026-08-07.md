# DEPTHART_ADMISSION_R1 A3：ONNX/QNN deployment preflight

状态：`ONNX_EXPORT_PASS / QNN_CONVERSION_BLOCKED`

本轮只检查导出链路和工具可用性，不产生 ONNX parity、QNN 转换、HTP 执行、Android 或生产证据。R0 结论与数据角色不变。

## 观察

- 安装 MSVC 2022、CUDA 12.8.93，并为 RTX 5060 / SM 12.0 编译官方 Selective Scan CUDA extension；核心测试 `9/9` 通过。
- 使用 PyTorch legacy custom-symbolic exporter 成功生成 31,985,722-byte metric S448 ONNX：输入 `image,K`、输出 `depth`、3555 nodes、5 个 `com.depthart::SelectiveScan`，SHA-256 `06A0C059...78C`。
- QAIRT `2.47.0.260601` 实际已存在于 `E:\codex-tools\qairt`，此前只是未加入 PATH。原图转换首先在 10 个固定 batched-linear Einsum 上失败。
- 将两种 Einsum 逐项等价改写为 reversed-input MatMul 后，QAIRT 继续前进，但在 metric Camera Embedder 的 ONNX `Acos` 上因没有 translation 而停止；尚未到达 SelectiveScan 转换阶段。

机器可读 receipt：[`a3-onnx-qnn-preflight.json`](../../../artifacts.local/evidence/hftf/depthart-admission-r1/a3-onnx-qnn-preflight.json)

## 结论与边界

A3 已证明 ONNX static graph 可以生成，但 QNN conversion 仍 FAIL。当前直接 blocker 是 Camera Embedder `Acos`，不能提前宣称 SelectiveScan 支持或不支持。下一步必须保持动态 `K` 输入合同，先解决或正式否决 `Acos`，再继续推进到 SelectiveScan；不得缓存固定 K 冒充 metric Camera Adapter。
