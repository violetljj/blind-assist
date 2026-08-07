# DepthART P0 部署主题索引

状态：`current / deployment-only / A3_BLOCKED_SELECTIVESCAN`

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
| 图改写与静态形状 | [deployment/depthart/legacy_adapter.py](deployment/depthart/legacy_adapter.py) → 旧实现 [rewrite_depthart_qairt_onnx.py](rewrite_depthart_qairt_onnx.py)、[rewrite_depthart_qairt_static_shape.py](rewrite_depthart_qairt_static_shape.py) | 生成 QAIRT 可消费的候选图 | 不证明数值 parity |
| hygiene / admission | [deployment/depthart/legacy_adapter.py](deployment/depthart/legacy_adapter.py) → 旧实现 [rewrite_depthart_qairt_hygiene.py](rewrite_depthart_qairt_hygiene.py)、[depthart_admission_r1.py](depthart_admission_r1.py) | 检查输入合同、准入前置条件 | 不改变默认 App |
| SelectiveScan 自定义算子 | [depthart_selective_scan_converter_op.cpp](depthart_selective_scan_converter_op.cpp)、[depthart_selective_scan_op_package.xml](depthart_selective_scan_op_package.xml) | 当前 A3 的 lowering / converter 入口 | 不代表 HTP 已可运行 |
| QNN/HTP 诊断 | [diagnostics/depthart/legacy_adapter.py](diagnostics/depthart/legacy_adapter.py) → [run_qnn_detailed_operator_profile_r0.ps1](run_qnn_detailed_operator_profile_r0.ps1)、[run_qnn_htp_linting_profile_r0.ps1](run_qnn_htp_linting_profile_r0.ps1) | operator/profile/lint 诊断 | 诊断结果不是性能或安全授权 |
| 已有结果 | [DEPTHART_ADMISSION_R1_A3_RESULT_2026-08-07.md](DEPTHART_ADMISSION_R1_A3_RESULT_2026-08-07.md)、[DEPTH_ANYTHING_V2_QAIRT_HTP_R0_RESULT.md](DEPTH_ANYTHING_V2_QAIRT_HTP_R0_RESULT.md) | 当前证据与阻塞点 | 旧结果不能自动生成 successor |

## 唯一 successor

`DEPTHART_PARENT_DISJOINT_ADMISSION_SUCCESSOR`：

1. 冻结 parent-disjoint 输入和数值比较口径。
2. 完成 SelectiveScan lowering 与 numerical parity。
3. 只有 parity/lowering 通过后，才进入设备 backend/HTP 评估。

## 禁止动作

- 不把 QAIRT/QNN/HTP 导出成功写成算法准确率、产品安全或默认 App 准入。
- 不修改已 consumed 数据、旧 protocol、receipt 或历史路径来“修”结果。
- 不把本页的部署 successor 复制到算法或数据 current 页面。
- 不把未登记的新 converter、`.exp/.lib` 或设备产物移动到其他目录；先登记 manifest 和调用方。
