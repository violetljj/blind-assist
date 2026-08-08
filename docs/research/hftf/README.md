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
- DepthART-S 是当前研发主力候选：R0 为 `QUALITY_NOT_ADMITTED`，R1 保持 `RESEARCH_MAINLINE`；A3 的 converter mapping 与 exact primitive reference 均可完成 QAIRT conversion，但 primitive 图膨胀到 21,440 QNN IR ops，未选为移动实现。无堆 float32 scalar reference 已用 Hexagon Tools 8.7.06 编译为 v73 HTP package，并同时生成 Android aarch64 prepare-side package；尚未完成算子 parity、QNN context、HTP/设备执行或性能评价，Android/生产 authority 关闭。
- 既有 DA V2、FRESH-TF、Metric3D、ToF 和 temporal 结果保留为 Development、diagnostic 或 paused 证据，不能互相拼接成晋级结论。

## 稳定入口

- [DepthART R0 protocol/result](DEPTHART_ADMISSION_R0_PROTOCOL_2026-08-07.md) · [R0 result](DEPTHART_ADMISSION_R0_RESULT_2026-08-07.md)
- [DepthART R1 A3 result](DEPTHART_ADMISSION_R1_A3_RESULT_2026-08-07.md)
- [DA2 P1/P2 closure](DAV2_P1_P2_EXECUTION_CLOSURE_2026-08-05.md)
- [HFTF candidate charter](HFTF_CANDIDATE_LANE_CHARTER_R0_2026-08-01.md)
- [算法研究入口](../ALGORITHM_RESEARCH_CURRENT.md)

## 唯一 successor

`DEPTHART_PARENT_DISJOINT_ADMISSION_SUCCESSOR`：只有在明确 causal difference、独立
parent/session 数据和最小判别实验被冻结后，才能开启新的 scientific admission；
部署预检修复本身不产生科学晋级。

当前两条明确的路线 successor：DA2 只作为冻结 reference 使用；DepthART-S 先完成 canonical numerical parity，并以 exact primitive 图作 oracle 验证已编译的最小 HTP runtime kernel，之后才进入 QNN context/device backend，再另行激活 parent-disjoint admission。两者都不能自动产生默认 App 权限。

## 禁止与权限边界

禁止在 consumed 数据上调参回救、把 teacher/合成/单设备性能当作 accuracy 或 safety、
把 HTP/Android canary 接入默认 App、或将任何 DepthART 结果写成生产权限。没有满足
successor 条件时，路线保持 `diagnostic` 或 `paused`。
