# DepthART P0 部署主题索引

状态：`current / deployment-only / cluster-physically-staged / HTP_V73_AND_AARCH64_PACKAGE_COMPILED / RUNTIME_NOT_EVALUATED`

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
| SelectiveScan 自定义算子 | [depthart_selective_scan_converter_op.cpp](depthart_selective_scan_converter_op.cpp)、[depthart_selective_scan_op_package.xml](depthart_selective_scan_op_package.xml)、[HTP scalar reference kernel](deployment/depthart/depthart_selective_scan_htp_reference.cpp)、[可复现构建脚本](deployment/depthart/build_depthart_selective_scan_htp_op_package.ps1)；[合同测试](deployment/depthart/test_depthart_selective_scan_op_package.py) | converter mapping 与 exact oracle 已完成；v73 HTP 和 aarch64 prepare-side package 已编译，待算子 parity 与 runtime load/execute | 编译成功不代表 QNN context 或 HTP 已执行 |
| QNN/HTP 诊断 | [diagnostics/depthart/](diagnostics/depthart/) | operator/profile/lint 诊断 | 诊断结果不是性能或安全授权 |
| 已有结果 | [DepthART A3 result](../../../docs/research/hftf/DEPTHART_ADMISSION_R1_A3_RESULT_2026-08-07.md)、[旧 QAIRT/HTP result](../../../docs/research/hftf/archive/DEPTH_ANYTHING_V2_QAIRT_HTP_R0_RESULT.md) | 当前证据与历史阻塞点 | 旧结果不能自动生成 successor |

## 唯一 successor

`DEPTHART_PARENT_DISJOINT_ADMISSION_SUCCESSOR`：

1. 冻结 parent-disjoint 输入和数值比较口径。
2. 保留 exact unrolled primitive 图作为算子 parity oracle，不将其作为移动端实现候选。
3. 保留已编译的 v73/aarch64 Op Package 及 receipt；不得把 binary presence 写成 runtime evidence。
4. 将已编译 kernel 与 primitive oracle 做算子级 parity，并完成 canonical ONNX numerical parity；只有 parity 和真实 runtime load/execute 通过后，才进入 QNN context、设备 backend/HTP 与性能评估。

## 禁止动作

- 不把 QAIRT/QNN/HTP 导出成功写成算法准确率、产品安全或默认 App 准入。
- 不修改已 consumed 数据、旧 protocol、receipt 或历史路径来“修”结果。
- 不把本页的部署 successor 复制到算法或数据 current 页面。
- 不把未登记的新 converter、`.exp/.lib` 或设备产物移动到其他目录；先登记 manifest 和调用方。
