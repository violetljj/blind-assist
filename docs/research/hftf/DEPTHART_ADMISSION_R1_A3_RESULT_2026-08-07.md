# DEPTHART_ADMISSION_R1 A3：ONNX/QNN deployment preflight

状态：`G3-A_EXPORT_PASS / G3-B_PARTIAL_PASS / G3-C_BLOCKED_SELECTIVESCAN`

本轮检查导出链路、图改写与 QAIRT converter reachability；不产生完整 ONNX runtime parity、HTP 执行、Android 或生产证据。R0 结论与数据角色不变。

## 观察

- 安装 MSVC 2022、CUDA 12.8.93，并为 RTX 5060 / SM 12.0 编译官方 Selective Scan CUDA extension；核心测试 `9/9` 通过。
- 使用 PyTorch legacy custom-symbolic exporter 成功生成 31,985,722-byte metric S448 ONNX：输入 `image,K`、输出 `depth`、3555 nodes、5 个 `com.depthart::SelectiveScan`，SHA-256 `06A0C059...78C`。
- QAIRT `2.47.0.260601` 实际已存在于 `E:\codex-tools\qairt`，此前只是未加入 PATH。原图转换首先在 10 个固定 batched-linear Einsum 上失败。
- 将两种 Einsum 逐项等价改写为 reversed-input MatMul 后，QAIRT 继续前进，但在 metric Camera Embedder 的 ONNX `Acos` 上因没有 translation 而停止。
- 将 Camera Embedder 外提为 host 计算的四级 `camera_prompt_*` 输入；PyTorch prompt parity `max_abs=0.0`，外提图 2823 nodes、`Acos=0`、`SelectiveScan=5`。再应用 Einsum→MatMul 后，QAIRT 已真正触达 5 个 SelectiveScan。
- 正常转换的首个明确停止点仍是 `onnx_selectivescan`：`No translation registered for op type onnx_selectivescan`。随后 dry-run 还枚举出 `Erf`、`LayerNormalization`、`Resize`、`ConstantOfShape`、`Expand`、`Where`、`Mod` 及若干 unsupported attributes；这些是待清理/复核候选，不等同于已经证明的转换停止点。
- 第一档 Graph Hygiene Pass 已完成：移除 123 个 `BatchNormalization.training_mode=0`、108 个 `Reshape.allowzero=0`，以及零 padding AveragePool 的 4+4 个默认属性；节点数保持 2823。重跑 normal conversion 后 frontier 未漂移，仍首先停在 5 个 `onnx_selectivescan`。

机器可读 receipt：[`a3-onnx-qnn-preflight.json`](../../../artifacts.local/evidence/hftf/depthart-admission-r1/a3-onnx-qnn-preflight.json)

## 结论与边界

A3 已证明 ONNX static graph 可以生成，且通过 host camera-prompt 分区绕过 `Acos` 后 converter 可到达 SelectiveScan。G3-C 仍为 blocked，不是 HTP FAIL：尚未完成 SelectiveScan lowering/custom-op、其他 dry-run 候选的逐层清理、graph/context 生成、partition 或 Snapdragon 实机执行。reference 合同继续保持 `image,K→depth`；mobile graph 的 prompt 输入只代表硬件感知分区，不是固定 K 冒充动态 metric conditioning。
