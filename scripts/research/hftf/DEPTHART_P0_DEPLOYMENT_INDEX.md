# DepthART P0 部署主题索引

状态：`current / deployment-only / G4-A_PACKAGE_REGISTRATION_PASS / G4-B_OPERATOR_PARITY_PASS_SM8650_V75 / G4-C_FULL_CONTEXT_PASS_SM8650_V75 / G4-D_FAIL_CURRENT_QAIRT_HTP_STANDARD_FLOAT_PATH_NOT_SUPPORTED`

本页是 DepthART 部署可行性主题簇的短入口。它只回答“先看什么、当前证明到哪、下一步是什么”；完整实验过程、原始日志和生成物留在本目录或 `artifacts.local/`，不在这里展开。

## 三十秒读取路径

1. [DepthART 路线 README](README.md)：算法主张、准入边界和 successor。
2. [部署当前入口](../../../docs/research/SYSTEM_RESEARCH_CURRENT.md)：系统/部署总状态。
3. 下表中与本次任务直接相关的一个脚本或结果；不要扫描整个 `hftf/`。

迁移前的机器清单见
[DEPTHART_P0_MIGRATION_MANIFEST.json](DEPTHART_P0_MIGRATION_MANIFEST.json)。当前状态是
`inventory_frozen_no_physical_move`，它不是移动授权。

## 当前分层

| 层 | 唯一入口 | 当前含义 | 禁止推断 |
|---|---|---|---|
| 算法主张 | [README.md](README.md) | DepthART-S 是当前算法主线 | 部署成功不等于算法 admission |
| 图改写与静态形状 | [deployment/depthart/rewrite_depthart_qairt_onnx.py](deployment/depthart/rewrite_depthart_qairt_onnx.py)、[deployment/depthart/rewrite_depthart_qairt_static_shape.py](deployment/depthart/rewrite_depthart_qairt_static_shape.py)、[deployment/depthart/lower_depthart_selective_scan_onnx.py](deployment/depthart/lower_depthart_selective_scan_onnx.py)；测试同目录 | 生成 QAIRT 可消费图；unrolled primitive 图仅作 parity/上界 reference | 不证明完整数值 parity 或 HTP 效率 |
| hygiene / admission | [deployment/depthart/rewrite_depthart_qairt_hygiene.py](deployment/depthart/rewrite_depthart_qairt_hygiene.py)、[deployment/depthart/depthart_admission_r1.py](deployment/depthart/depthart_admission_r1.py)；测试同目录 | 检查输入合同、准入前置条件 | 不改变默认 App |
| SelectiveScan 自定义算子 | [depthart_selective_scan_converter_op.cpp](depthart_selective_scan_converter_op.cpp)、[depthart_selective_scan_op_package.xml](depthart_selective_scan_op_package.xml)、[HTP scalar reference kernel](deployment/depthart/depthart_selective_scan_htp_reference.cpp)、[可复现构建脚本](deployment/depthart/build_depthart_selective_scan_htp_op_package.ps1)、[canary/oracle](deployment/depthart/prepare_depthart_selective_scan_canary.py)、[parity evaluator](deployment/depthart/evaluate_depthart_selective_scan_canary.py)；[合同测试](deployment/depthart/test_depthart_selective_scan_op_package.py) | v73 保留为 compile-only 工件；当前 SM8650/v75 已完成 package load、compose/finalize、HTP execute 与 3/3 oracle parity | 单算子 PASS 不代表完整图、partition purity 或性能 |
| LayerNorm 自定义算子 | [HTP scalar reference kernel](deployment/depthart/depthart_layernorm_htp_reference.cpp)、[全图映射](deployment/depthart/rewrite_depthart_layernorm_custom_onnx.py)、[canary/oracle](deployment/depthart/prepare_depthart_layernorm_canary.py)、[parity evaluator](deployment/depthart/evaluate_depthart_layernorm_canary.py) | SM8650/v75 单算子 3/3 PASS；23 个 LayerNorm 与 5 个 SelectiveScan 的完整图 context finalize/save PASS | context PASS 不代表 full-model parity、partition purity 或性能 |
| 首个 SelectiveScan 前数值二分 | [纯标准 prefix probe 与 evaluator](deployment/depthart/bisect_depthart_pre_scan_parity.py) | 固定 canonical ONNX、程序化 RGB canary 与 G4-D 容差，按单输出子图定位 ORT/HTP 首个漂移点 | 只具 synthetic numerical diagnostic authority；不得启动 G4-E/F 或替换 DA2 |
| PyTorch→ONNX 分段定位 | [stage anchor localizer](deployment/depthart/localize_depthart_pytorch_onnx_parity.py) | 同时保留原生 PyTorch、导出语义 replay 与 exact-primitive ONNX，覆盖 backbone/DAA/depth head/scale head | 只允许修复首个已证明的漂移段；不得换 canary 或放宽容差 |
| G4-D 节点族修复与总门 | [PatchConv rewrite](deployment/depthart/rewrite_depthart_first_patch_conv_custom_onnx.py)、[BatchNorm rewrite](deployment/depthart/rewrite_depthart_batchnorm_custom_onnx.py)、[GELU rewrite](deployment/depthart/rewrite_depthart_gelu_custom_onnx.py)、[三项 evaluator](deployment/depthart/evaluate_depthart_g4d_repair.py) | TF32-off 后 PyTorch↔ONNX PASS；局部 custom PatchConv/BN/GELU 前缀 PASS，但下一标准 Conv 与整图 HTP 仍 FAIL；direct↔context bit-exact | 当前 QAIRT 2.47/SM8650 HTP 标准 float 路径的 strict G4-D 为负终态；不扩写为全部 HTP 或全部 custom-op 不可行 |
| Task-preserving D0 | [三臂冻结协议](../../../docs/research/hftf/DEPTHART_TASK_PRESERVING_D0_PRECISION_SCREEN_PROTOCOL_2026-08-09.md)、[source/control lock](../../../docs/research/hftf/DEPTHART_TASK_PRESERVING_D0_SOURCE_CONTROL_LOCK_2026-08-09.json)、[terminal result](../../../docs/research/hftf/DEPTHART_TASK_PRESERVING_D0_PRECISION_SCREEN_RESULT_2026-08-09.md) | FP16/W8A16/INT8 使用同一 source；量化臂共享 16-frame calibration；只执行 outcome 前技术前门 | 三臂均技术淘汰，未访问任务质量/性能；`D0_NO_TASK_PRESERVING_CANDIDATE_R2_NOT_ACTIVATED` |
| Task-preserving R2 | [冻结协议](../../../docs/research/hftf/DEPTHART_TASK_PRESERVING_DEPLOYMENT_R2_PROTOCOL_2026-08-09.md)、[ARKit roster lock](../../../docs/research/hftf/DEPTHART_TASK_PRESERVING_R2_ARKIT_ROSTER_LOCK_2026-08-09.json)、[pre-outcome validator](deployment/depthart/validate_depthart_task_preserving_r2_activation.py)、[quality evaluator](deployment/depthart/evaluate_depthart_task_preserving_r2_quality.py) | 新 Development screen receipt 后只允许一个冻结候选；8 个新 Validation visit/session 与旧 HFTF identity 零重叠，payload 零读取 | `CANDIDATE_NOT_SELECTED / MEDIA_DOWNLOAD_AUTHORIZATION_REQUIRED / EXECUTION_NOT_ACTIVATED`；R2 只做独立确认，不选模 |
| QNN/HTP 诊断 | [diagnostics/depthart/](diagnostics/depthart/) | operator/profile/lint 诊断 | 诊断结果不是性能或安全授权 |
| 已有结果 | [DepthART A3 result](../../../docs/research/hftf/DEPTHART_ADMISSION_R1_A3_RESULT_2026-08-07.md)、[旧 QAIRT/HTP result](../../../docs/research/hftf/archive/DEPTH_ANYTHING_V2_QAIRT_HTP_R0_RESULT.md) | 当前证据与历史阻塞点 | 旧结果不能自动生成 successor |

## 唯一 successor

`DEPTHART_TASK_PRESERVING_D1_FIXED_MIXED_DEVELOPMENT_SCREEN`：D0 已关闭且无合格 arm。
下一步先冻结产品纵横比/FOV/resize、intrinsics/truth 对齐与 task postprocess，重建对应的
fixed-mixed 图，并建立一份不复用 R0/calibration/R2 rows 的 Development roster，再做单候选
task screen。它是新 successor，不得回写 D0；
strict G4-D、旧 G4-E/F 与 DA2 replacement 状态均不变。

## 禁止动作

- 不把 QAIRT/QNN/HTP 导出成功写成算法准确率、产品安全或默认 App 准入。
- 不修改已 consumed 数据、旧 protocol、receipt 或历史路径来“修”结果。
- 不把本页的部署 successor 复制到算法或数据 current 页面。
- 不把未登记的新 converter、`.exp/.lib` 或设备产物移动到其他目录；先登记 manifest 和调用方。
