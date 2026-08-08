# DepthART deployment module

状态：`current / deployment-only / adapter-first`

## 稳定 Interface

本目录是 DepthART QAIRT/QNN/HTP/ONNX 与 SelectiveScan 部署工作的规范入口。
DepthART/DA2 QAIRT 部署实现和定向回归测试已物理迁入本目录；仓库内部旧 Python
shim 已退役，统一从本目录导入。

## 输出

- 候选 ONNX/QAIRT 图
- lowering、operator、parity 和 backend 诊断记录
- `build_depthart_selective_scan_htp_op_package.ps1` 生成的 v73/aarch64 本机 package 与 build receipt
- `prepare_depthart_selective_scan_canary.py` 生成冻结单算子图、三组输入与 float32 oracle
- `evaluate_depthart_selective_scan_canary.py` 计算设备输出的绝对、相对、分位与逐 step 误差
- `build_depthart_converter_op_package.ps1` 将 SelectiveScan/LayerNorm shape/type inference DLL 可复现地构建到 `artifacts.local/`
- `rewrite_depthart_layernorm_rank4_onnx.py` 与 `lower_depthart_layernorm_onnx.py` 提供 G4-C 的等价 LayerNorm rank/formula 诊断路径；它们不自带 runtime 或 parity authority
- `rewrite_depthart_layernorm_custom_onnx.py`、`depthart_layernorm_htp_reference.cpp`、`prepare/evaluate_depthart_layernorm_canary.py` 提供最后一轴 float32 LayerNorm 的映射、HTP reference 与单算子 parity；当前已用于 `SM8650 / Snapdragon 8 Gen 3 / HTP v75` 完整 context 闭合
- `prepare/evaluate_depthart_full_graph_canary.py` 冻结程序化 RGB、camera prompts 与 PyTorch oracle，并计算 PyTorch、exact-primitive ONNX、QNN HTP direct/context 及首个 custom-op frontier 的完整图差异；只具 synthetic numerical authority
- `bisect_depthart_pre_scan_parity.py` 从同一 canonical ONNX 反向裁剪首个 SelectiveScan 第一个输入的纯标准算子依赖图，以单终点 probe 和冻结 RGB input 做 ORT/HTP 数值二分；固定使用 ORT `1.27.0` 并沿用 G4-D 的 `rtol=3e-5 / atol=3e-6`
- `localize_depthart_pytorch_onnx_parity.py` 在同一冻结 canary 上采集 patch embed、四级 DAA/backbone、depth head、scale head 和最终 depth，分别比较原生 PyTorch、导出语义 replay 与 exact-primitive ONNX；用于定位导出侧首个漂移段，不改变样本或容差
- `rewrite_depthart_first_patch_conv_custom_onnx.py`、`rewrite_depthart_batchnorm_custom_onnx.py` 与 `rewrite_depthart_gelu_custom_onnx.py` 只改写已定位的节点族；对应 float32 HTP reference kernels 用于 correctness-first 诊断，不是性能实现
- `evaluate_depthart_g4d_repair.py` 固定 `rtol=3e-5 / atol=3e-6`，同时签署 PyTorch↔canonical ONNX、canonical ONNX↔SM8650 HTP、DLC direct↔saved context 三项门；任一失败即保持 G4-D FAIL
- 只写入 `artifacts.local/` 的 receipt 与日志

## 安全边界

部署脚本只能证明导出、lowering、数值 parity 或设备可行性，不能单独证明算法准入、
默认 App、产品安全或生产授权。

## 停止条件

- G4-D parity 不通过时停止进入 G4-E partition purity 与 G4-F performance
- 缺少冻结输入、receipt 或调用方清单时停止物理迁移
- 不移动并行任务产生的 SelectiveScan `.cpp/.xml/.exp/.lib`

产物目录：`artifacts.local/`
