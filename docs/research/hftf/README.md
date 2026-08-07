# DepthART 算法路线与并行项目研究 workstreams

状态：`current / DEVELOPMENT_STANDARD / INNOVATION_NOT_EVALUABLE / DEFAULT_APP_UNCHANGED`

本页只维护当前摘要、权限和唯一 successor。完整历史已保留在 [archive/README_FULL_HISTORY_2026-08-07.md](archive/README_FULL_HISTORY_2026-08-07.md)，日期化协议、receipt 和结果仍是 snapshot/机器证据。

## 当前主张

当前算法研究主要围绕项目自有的 DepthART-S。项目整体还包含数据集/数据治理、通信链路与
端到端延迟、CPU/GPU/HTP 性能、内存/热/稳定性和部署可行性等并行 workstream；它们可以独立
研究，也可以在接口和证据边界明确后与 DepthART 或其他路线耦合。不等于正式 App 能力。

## 当前状态

- DepthART 算法路线与双环论文次线隔离，默认 App 和正式 YOLO 模型不变。
- DA2 保持冻结的 metric teacher、baseline、regression reference 和 fallback，不因新候选结果删除或降级。
- DepthART-S 是当前研发主力候选：R0 为 `QUALITY_NOT_ADMITTED`，R1 保持 `RESEARCH_MAINLINE`；A3 ONNX/QNN 部署预检为 `BLOCKED_SELECTIVESCAN`，HTP/Android/生产 authority 关闭。
- 既有 DA V2、FRESH-TF、Metric3D、ToF 和 temporal 结果保留为 Development、diagnostic 或 paused 证据，不能互相拼接成晋级结论。

## Workstreams

- 算法：DepthART-S admission、clearance/false-clear、时序与几何候选。
- 数据：parent/session 隔离、truth、coverage、质量和数据角色合同。
- 链路：外设、网络、拷贝、排队、推理、反馈的端到端延迟分解。
- 平台：模型导出、operator/lowering、CPU/GPU/HTP、内存、热、稳定性和部署可行性。

## 稳定入口

- [DepthART R0 protocol/result](DEPTHART_ADMISSION_R0_PROTOCOL_2026-08-07.md) · [R0 result](DEPTHART_ADMISSION_R0_RESULT_2026-08-07.md)
- [DepthART R1 A3 result](DEPTHART_ADMISSION_R1_A3_RESULT_2026-08-07.md)
- [DA2 P1/P2 closure](DAV2_P1_P2_EXECUTION_CLOSURE_2026-08-05.md)
- [HFTF candidate charter](HFTF_CANDIDATE_LANE_CHARTER_R0_2026-08-01.md)
- [项目算法路线总表](../ALGORITHM_ROUTE_REGISTRY.md)

## 唯一 successor

`HFTF_FRESH_PARENT_DISJOINT_CANDIDATE_SUCCESSOR`：只有在明确 causal difference、独立 parent/session 数据和最小判别实验被冻结后，才能开启下一候选；部署预检修复本身不产生科学晋级。

当前两条明确的路线 successor：DA2 只作为冻结 reference 使用；DepthART-S 先完成 numerical parity/SelectiveScan 可行性，再另行激活 parent-disjoint admission。两者都不能自动产生默认 App 权限。

## 禁止与权限边界

禁止在 consumed 数据上调参回救、把 teacher/合成/单设备性能当作 accuracy 或 safety、把 HTP/Android canary 接入默认 App、或将任何 HFTF 结果写成生产权限。没有满足 successor 条件时，路线保持 `diagnostic` 或 `paused`。
