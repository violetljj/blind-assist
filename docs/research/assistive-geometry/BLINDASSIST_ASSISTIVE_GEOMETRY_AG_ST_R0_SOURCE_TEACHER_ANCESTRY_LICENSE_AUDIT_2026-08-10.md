# AG-ST R0 source / Teacher / ancestry / license audit

状态：`PARTIAL_PASS_FOR_PROTOCOL_LOCK_ONLY / EXECUTION_BLOCKED`

本审计只回答“哪些候选值得写入选择性 labelability 协议，以及执行前还缺什么”。没有下载代码或
权重，没有运行 Teacher，没有新开 source payload，也没有物化伪标签或训练 student。

## 结论

| 候选 | R0 角色 | 已确认能力 | 关键依赖或重叠 | 当前处置 |
|---|---|---|---|---|
| MapAnything | Stage 0A 主候选 | 可组合 RGB、K/ray、depth、pose，输出 metric geometry、confidence 与 mask | 官方训练管线列出 ScanNet++ v2，并使用 DINOv2 初始化；default / Apache 权重训练组成仍须随 exact model card 锁定 | 只允许下一步 artifact lock |
| DA3 | Stage 0B 增量候选 | 任意视图、可输入已知 pose；Nested `-1.1` 支持 metric depth | 论文明确使用 ARKitScenes，并以对齐 sparse/noisy depth 的 teacher pseudo-depth 训练；不能视为对 ARKitScenes source truth 的独立票 | 0A 可评价后才允许 artifact lock |
| UniDepthV2 | 延后 depth diagnostic | 单帧 metric depth + confidence | 训练集包含 ARKitScenes、ScanNet、ScanNet++；confidence 仅是单输入内相对误差排序 | 不进入初始 R0 |
| Metric3Dv2 | 延后 normal/depth diagnostic | 单帧 metric depth + surface normal | README 只明确 code 为 BSD-2；exact weight 条款与训练祖先仍未锁定 | 不进入初始 R0 |

官方依据：[MapAnything](https://github.com/facebookresearch/map-anything)、
[MapAnything training](https://github.com/facebookresearch/map-anything/blob/main/train.md)、
[Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3)、
[DA3 paper](https://arxiv.org/html/2511.10647)、
[UniDepthV2 paper](https://arxiv.org/html/2502.20110v2)、
[UniDepthV2 confidence contract](https://github.com/lpiccinelli-eth/UniDepth/blob/main/assets/docs/V2_README.md)、
[Metric3Dv2](https://github.com/YvanYin/Metric3D)。

## Source 现状

现有 B1 TRAIN pool 提供 metric depth `4,767 frames / 16 parents`，但 support 只有
`320 / 11`，joint factor bundle 只有 `310 / 10`，continuous boundary 与 complete R2 factor
schema 都是 `0 / 0`。更重要的是，DCA
target atlas 的 explicit timestamp 与 pose transform 仍是 `0 / 0`；B0 raw receipt 中存在 trajectory
和 K，不等于 pose-conditioned Teacher 已有可执行输入合同。

因此，下一步必须先冻结 RGB/K/pose/depth adapter、坐标约定、anchor 与 hidden truth 的不相交
receipt。source-native depth 也只在其注册、有效性和误差条件满足处提供 evidence，不能自动覆盖所有像素。

## 不可跨越的解释边界

- 多 Teacher agreement 是相关证据，不是独立投票。
- confidence、disagreement、reprojection residual 只能进入 `ACCEPT / UNKNOWN` gate，不能充当 sigma truth。
- dense normal 只能做 gate/派生诊断；当前 F1 schema 的 normal 是 support-plane normal。
- semantic mask 不产生 metric physical boundary，也不产生 UNKNOWN truth；SAM 类模型不进入 Stage 0A/0B。
- 即使未来 R0 PASS，也只能支持提出新的 source-label contract amendment；不能签署当前 F1 frontdoor PASS。
- Stage 0B 若在 Stage 0A outcome 后才加入，必须另立 pre-outcome 协议并使用未消费 canary；不能在已开真的 0A canary 上调双 Teacher gate。

机器审计见
[JSON](BLINDASSIST_ASSISTIVE_GEOMETRY_AG_ST_R0_SOURCE_TEACHER_ANCESTRY_LICENSE_AUDIT_2026-08-10.json)。
