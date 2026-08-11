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
- `preflight_depthart_task_preserving_d1_arkit_assets.py` 只对冻结的 D1 primary/reserve roster 与五类 ARKitScenes 资产执行 HEAD 可用性检查；它不读取媒体 body、truth/model outcome，也不在 header 阶段替换 primary
- `materialize_depthart_task_preserving_d1_arkit_preflight.py` 按冻结顺序下载 D1 的 16 个身份，机械审计 300 帧连续 portrait、pose bracket、RGB-D-confidence/K 完整性，并只按 label-blind 规则形成 8 身份 Development roster；它不计算任务 truth 或模型结果
- `DEPTHART_TASK_PRESERVING_D1_ARKIT_BODY_PREFLIGHT_RESULT_2026-08-10` 已把 4 primary + 4 frozen-order reserve replacement 锁成最终 8-session Development roster；逐资产/逐帧 receipt 留在 `artifacts.local/`，没有 task/model outcome
- `DEPTHART_TASK_PRESERVING_D1_PRODUCT_ASPECT_TECHNICAL_PREFLIGHT_RESULT_2026-08-10` 已锁唯一 `608×448` candidate ONNX/DLC、reference checkpoint、postprocess 与 roster SHA；host conversion PASS 不等于 SM8650 context、HTP execution 或 parity，设备缺席时必须停止
- `prepare_depthart_full_graph_canary.py` 同时支持 square `--resolution` 与固定 `--height/--width`，D1 使用 deterministic `608×448` synthetic input 与 PyTorch oracle；它不读取 ARKitScenes task outcome
- `validate/evaluate_depthart_task_preserving_d1_device_*` 分别在设备输出前验证 exact protocol/runtime/canary SHA，并在设备执行后重算 context、shape/finite、direct/context bit-exact 与 raw-depth diagnostic；后者不是 task-quality 或性能 evaluator
- `DEPTHART_TASK_PRESERVING_D1_SM8650_HTP_CONTEXT_AND_OUTCOME_ACTIVATION_PREFLIGHT_RESULT_2026-08-10` 已在 fresh `SM-S9280 / SM8650 / HTP v75 / DZG1` 上关闭 context 与 execute 前门；raw-depth parity 仍 FAIL，Development task outcome 仍未启动
- `depthart_task_preserving_d2_task_head_canary.py` 只验证小型 task-evidence head 的 shape、硬 UNKNOWN、horizon monotonicity 和 bounded residual mechanics；不训练、不读取真实 outcome，也不产生 accuracy 或 candidate authority
- `plan_depthart_task_preserving_d2_support_pool.py` 从 Apple Training metadata 中排除所有已冻结官方 identity，并以固定哈希锁定 32 个 D2 source-support 候选；不读取媒体或模型结果
- `preflight/materialize_depthart_task_preserving_d2_phase_a_*` 先做 intrinsics/trajectory HEAD，再按冻结顺序机械锁定 16 个具有 300-frame portrait/pose continuity 的 identity；不读取 depth truth、RGB 或模型
- `preflight/materialize_depthart_task_preserving_d2_phase_b_*` 对 Phase-A 首窗口的 depth/confidence 做 source-truth support 审计；当前终态仅 2/8 support-qualified，少于 8 时生成器强制输出空 role list，不授权训练或 Development outcome
- `DEPTHART_TASK_PRESERVING_D2R1_TARGET_SUPPORT_WINDOW_RECOVERY_PROTOCOL_2026-08-11` 仅冻结同一 16 个 identity 的 full-run 300-frame support-window recovery；在新 source scope 获得显式授权前不得执行 HEAD、GET 或 scan
- `preflight_depthart_task_preserving_d2r1_assets.py` 对同一 16 个 identity 的 intrinsics/trajectory/depth/confidence 执行 64 个 HEAD，并强制总 body 不超过授权的 2.90 GB
- `materialize_depthart_task_preserving_d2r1.py` 逐身份下载并以未序列化 prefix sums 选择第一个全门通过的 300-frame portrait window；每身份有可恢复 checkpoint，不读取 RGB 或模型
- `reseal/validate_depthart_task_preserving_d2r1_*` 在不改写 receipt、不重算 truth 的前提下修复 v1 Windows CRLF byte seal，并复验 16/16 合格、4 TRAIN + 4 sealed DEVELOPMENT 的唯一角色锁
- `preflight_depthart_task_preserving_d2_phase_c_rgb_assets.py` 只对 exact 8 个 D2 role identity 的 `lowres_wide.zip` 执行 HEAD；当前 8/8 可用、body 总量 3,718,339,716 bytes，未读取 body
- `materialize/validate_depthart_task_preserving_d2_phase_c_sources.py` 下载精确 32 个 ZIP、验证全成员 CRC、按冻结 stems 提取 9,600 个源文件并全量复验 bytes/SHA；没有图像 decode、truth、模型或训练，Development 路径保持 sealed
- `run/validate_depthart_task_preserving_d2_train_only.py` 在 4 个 TRAIN identity 上以 24 个可恢复 chunk 生成 1,200 个 saved-context base outputs，冻结 3,600-band dataset 并按唯一固定 recipe 训练/确定性复验 step-500、277 参数 head；Development 和 R2 均未打开
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
