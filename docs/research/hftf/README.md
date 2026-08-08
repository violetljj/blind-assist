# DepthART 算法路线

状态：`current / DEVELOPMENT_STANDARD / INNOVATION_NOT_EVALUABLE / DEFAULT_APP_UNCHANGED`

本页只维护当前摘要、权限和唯一 successor。完整历史已保留在 [archive/README_FULL_HISTORY_2026-08-07.md](archive/README_FULL_HISTORY_2026-08-07.md)，日期化协议、receipt 和结果仍是 snapshot/机器证据。

## 当前主张

当前算法研究主要围绕项目自有的 DepthART-S。本页只维护 DepthART 算法路线；项目级数据、
通信链路、端到端延迟、性能和部署研究分别从 [研究总入口](../README.md) 进入，只有显式绑定时
才成为本路线的输入或证据。不等于正式 App 能力。

## 当前状态

- DepthART 算法路线与双环论文次线隔离，默认 App 和正式 YOLO 模型不变。
- DA2 保持冻结的 metric teacher、baseline、regression reference 和 fallback，不因新候选结果删除或降级。
- DepthART-S 是当前研发主力候选：R0 为 `QUALITY_NOT_ADMITTED`，R1 保持 `RESEARCH_MAINLINE`；G4-A/B/C PASS。两段式定位已完成：关闭 CUDA TF32 后，PyTorch↔canonical ONNX 最终 depth 在冻结容差下 PASS（`max_abs=1.40667e-5`）；SM8650/v75 侧以 custom float32 PatchConv、123×BatchNorm 与 27×GELU 逐族修复后，整图误差从约 `1.45` 降至 `0.0272727`，但仍 FAIL。修复前缀的下一首因是第二个标准 Conv（`max_abs=0.00568485`），而 direct DLC↔saved context bit-exact。结合 QAIRT 对 HTP FP32 模型底层 16-bit math 的明确边界，当前 QAIRT 2.47/SM8650 HTP 标准 float 路径的 strict G4-D 记为负终态；不推断全部 HTP 或近完整 custom-float32 engine 不可行。真实场景任务质量、G4-E partition purity、G4-F 性能、Android/生产 authority 仍未评价，DA2 保持冻结 baseline/fallback。
- 既有 DA V2、FRESH-TF、Metric3D、ToF 和 temporal 结果保留为 Development、diagnostic 或 paused 证据，不能互相拼接成晋级结论。

## 稳定入口

- [DepthART R0 protocol/result](DEPTHART_ADMISSION_R0_PROTOCOL_2026-08-07.md) · [R0 result](DEPTHART_ADMISSION_R0_RESULT_2026-08-07.md)
- [DepthART R1 A3 result](DEPTHART_ADMISSION_R1_A3_RESULT_2026-08-07.md)
- [DepthART task-preserving deployment R2 protocol](DEPTHART_TASK_PRESERVING_DEPLOYMENT_R2_PROTOCOL_2026-08-09.md) · [machine contract](DEPTHART_TASK_PRESERVING_DEPLOYMENT_R2_PROTOCOL_2026-08-09.json)
- [DA2 P1/P2 closure](DAV2_P1_P2_EXECUTION_CLOSURE_2026-08-05.md)
- [HFTF candidate charter](HFTF_CANDIDATE_LANE_CHARTER_R0_2026-08-01.md)
- [算法研究入口](../ALGORITHM_RESEARCH_CURRENT.md)

## 唯一 successor

`DEPTHART_TASK_PRESERVING_DEPLOYMENT_R2`：strict G4-D 保持不可变负终态，不再继续
PatchConv→Conv→Norm→activation 的隐式 custom-engine 扩张。R2 已冻结任务等价合同，但
`EXECUTION_NOT_ACTIVATED`：下一步只允许准备新独立 parent/session-disjoint cohort、冻结
一个 HTP-friendly candidate 及其身份，并用 pre-outcome validator 检查 activation
manifest。只有用户显式激活且任务质量门 PASS，才可评价该候选自己的 partition、latency、
RAM 与 thermal。部署结果仍不产生 scientific admission、DA2 替换或默认 App 权限。

近完整 custom-float32 engine 与新 runtime/hardware 保留为未激活的新立项候选，不能从
R2 或旧 G4-D 自动获得执行权限。

## 禁止与权限边界

禁止在 consumed 数据上调参回救、把 teacher/合成/单设备性能当作 accuracy 或 safety、
把 HTP/Android canary 接入默认 App、或将任何 DepthART 结果写成生产权限。没有满足
successor 条件时，路线保持 `diagnostic` 或 `paused`。
