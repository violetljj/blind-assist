# SANPO 当前状态

状态：current
最后核验：2026-07-20
适用范围：SANPO 数据、候选模型、公开银标和设备评测工作。

## 结论先行

- 正式 BlindAssist App 保持 `yolo11n_fp16_320.tflite` 默认检测路径；SANPO 候选不替换默认模型。
- SANPO 的数据、离线质量、INT8 和设备事件门必须按当前协议逐段通过；任一门未通过，不导出、不接入 App、不表述为生产能力。
- 公开视频银标、反事实和生命周期 r7.* 工作属于受限研究证据。它们不能仅凭单次实验授权训练、校准、blind 评测、Android runtime 或默认模型替换。
- Corridor-Causal Student 仅完成 benchmark-only 特征与性能可行性检查；缺少完整双人复核的连续事件真值，`96 episode / 48 matched pair` 门仍阻塞训练和事件效果评测。详见 [CORRIDOR_CAUSAL_PROGRESS_2026-07-20.md](CORRIDOR_CAUSAL_PROGRESS_2026-07-20.md)。
- USTRF-SC 已把 route-conditioned、object-agnostic risk field 设为优先研究主线，但新合同的 `120 episode / 60 matched pair` 真实事件真值当前为 0，设备米制几何 admission 仍为 false；不训练 student、不接入 App、不替换默认模型。详见 [ROUTE_CONDITIONED_OBJECT_AGNOSTIC_RISK_FIELD_PLAN_2026-07-20.md](ROUTE_CONDITIONED_OBJECT_AGNOSTIC_RISK_FIELD_PLAN_2026-07-20.md)。

## 当前操作入口

| 问题 | 当前真源 |
| --- | --- |
| 数据集、训练隔离与 blind 规则 | [SANPO_TRAINING_PROTOCOL.md](SANPO_TRAINING_PROTOCOL.md) |
| 候选晋级与设备事件门 | [SANPO_CANDIDATE_PROMOTION_GATES.md](SANPO_CANDIDATE_PROMOTION_GATES.md) |
| 连续序列评测和 baseline | [SANPO_SEQUENCE_EVALSET.md](SANPO_SEQUENCE_EVALSET.md)、[SANPO_TRAVERSABILITY_BASELINE.md](SANPO_TRAVERSABILITY_BASELINE.md) |
| 反事实采集与生命周期目标 | [SANPO_COUNTERFACTUAL_EPISODE_COLLECTION.md](SANPO_COUNTERFACTUAL_EPISODE_COLLECTION.md) |
| Corridor-Causal 候选的本轮结论 | [CORRIDOR_CAUSAL_PROGRESS_2026-07-20.md](CORRIDOR_CAUSAL_PROGRESS_2026-07-20.md)：仅工程可行性，未获得事件效果或晋级授权。 |
| Route-conditioned USTRF-SC 主线 | [ROUTE_CONDITIONED_OBJECT_AGNOSTIC_RISK_FIELD_PLAN_2026-07-20.md](ROUTE_CONDITIONED_OBJECT_AGNOSTIC_RISK_FIELD_PLAN_2026-07-20.md)：研究主线已激活，两道硬门仍阻塞，生产路径不变。 |
| 公开银标与来源研究 | 仅在对应协议已登记为 `current` 后按其执行；未提交的本地草稿不能作为仓库规则或授权依据。 |
| 最近研究证据 | 已提交的日期化 snapshot；未提交的研究记录保持任务本地状态，待其所属任务完成后再登记。 |

## 硬边界

- 不以 benchmark-only、oracle、机器生成标签、单一来源或事后压力样本冒充可部署结论。
- 不绕过数据集根门、blind 隔离、哈希/许可/隐私证据或既定晋级门。
- 若本文件与可复现门禁报告或 current 协议不一致，停止升级操作，先修正本文件并记录证据链接。

## 更新规则

只有以下事实改变时更新本页：默认模型/产品路线、门禁状态、授权边界、当前阻塞点或下一道可执行门。逐轮实验数字、失败细节和素材发现进入日期化 snapshot 与 `DEVELOPMENT_LOG.md`，本页只保留链接和操作结论。
