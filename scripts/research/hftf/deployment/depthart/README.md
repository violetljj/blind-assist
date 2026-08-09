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
- `DEPTHART_TASK_PRESERVING_D0_PRECISION_SCREEN_PROTOCOL_2026-08-09` 先在 Development 数据比较 FP16/W8A16/INT8；独立 R2 cohort 禁止参与 calibration、调参和选模
- `validate_depthart_task_preserving_d0_preflight.py` 静态核验三臂 recipe、QAIRT 工具、公共 source/control、FP32 custom-island package 与 strict G4-D/R2 数据边界；它不转换模型、不读取 outcome
- `prepare_depthart_task_preserving_d0_arm.py` 在 fresh `artifacts.local/` evidence root 中按冻结 recipe 物化单个 FP16/W8A16/INT8 DLC；量化臂没有冻结 calibration list 会 fail closed
- `plan_depthart_task_preserving_d0_tum_calibration_roster.py` 从本地 TUM RGB index 先排除既有 consumed R0 rows，再按每 sequence 固定 SHA-256 顺序冻结 W8A16/INT8 共用的 outcome-free calibration roster
- `materialize_depthart_task_preserving_d0_calibration_inputs.py` 只读取锁定 RGB/intrinsics，生成 image 与四级 camera prompt float32 raws 和单一绝对路径 calibration list；不运行 depth model outcome
- `plan_depthart_task_preserving_d1_arkit_roster.py` 读取 Apple split CSV 的冻结 Git blob，同时排除 HFTF 与 Assistive Geometry 冻结快照中的全部官方 identity，再按固定哈希锁定 8 primary + 8 reserve Training visit/session；不读取媒体或 outcome
- `validate_depthart_task_preserving_d1_contract.py` 核验 D1 产品 portrait/K、三 band × 三 horizon task postprocess、R2 等值质量门、metadata roster 独立性与未激活状态；它不下载媒体、不重建图、不授权 outcome access
- `validate_depthart_task_preserving_r2_activation.py` 只检查 R2 pre-outcome activation manifest 的 cohort 角色、候选/reference 身份、固定任务门与旧 G4-D 排除项；它不读取模型输出，不激活执行，也不签署质量或部署结论
- `plan_depthart_task_preserving_r2_arkit_roster.py` 在 Apple 官方 split CSV 上，以冻结 Git snapshot 排除全部既有 HFTF ARKit identity，再按固定哈希顺序锁定唯一 visit/session；只读元数据
- `evaluate_depthart_task_preserving_r2_quality.py` 计算 reference/candidate 对独立 truth 的 pooled、parent-macro、session-macro 与 worst-parent 任务门；CLI 没有显式 activation receipt 会拒绝读取 outcome
- 只写入 `artifacts.local/` 的 receipt 与日志

## 安全边界

部署脚本只能证明导出、lowering、数值 parity 或设备可行性，不能单独证明算法准入、
默认 App、产品安全或生产授权。

## 停止条件

- strict G4-D 保持负终态，不继续 custom 化标准算子；task-preserving R2 只有任务质量 PASS 后才进入该候选自己的 partition/performance
- 缺少冻结输入、receipt 或调用方清单时停止物理迁移
- 不移动并行任务产生的 SelectiveScan `.cpp/.xml/.exp/.lib`

产物目录：`artifacts.local/`
