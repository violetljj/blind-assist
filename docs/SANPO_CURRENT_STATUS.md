# SANPO 当前状态

状态：current
最后核验：2026-08-01
适用范围：SANPO 数据、候选模型、公开银标和设备评测工作。

```text
FORWARD_GOVERNANCE: THESIS_FIRST_RESEARCH_GOVERNANCE_R4
DEFAULT_NEW_WORK_LANE: THESIS_DEVELOPMENT
DEVELOPMENT_REQUIRES_PRODUCTION_PROMOTION_GATES: false
PRODUCTION_PROMOTION_REQUIRES_EXPLICIT_SCOPE: true
HISTORICAL_TERMINALS_IMMUTABLE: true
```

## 结论先行

- SANPO 新研究默认采用 `WILD_LAB / THESIS_DEVELOPMENT` 风格：优先追求新的表征、
  Teacher 数据升级、跨数据集训练、合成/伪标签和可证伪算法突破；缺少产品安全、
  真实用户或设备晋级证据，只限制相应高等级 claim，不阻止论文级机制研究。
- 正式 BlindAssist App 保持 `yolo11n_fp16_320.tflite` 默认检测路径；SANPO 候选不替换默认模型。
- SANPO 新工作默认进入论文 `THESIS_DEVELOPMENT`：可使用声明的 Development、
  consumed 或 synthetic 数据做训练、候选 utility、映射/decoder canary，以及算法选模
  或平台工程 benchmark；不要求依次通过 blind、INT8、设备事件和发布审查。
- 只有任务明确以默认模型替换、发布或生产晋级为目标时，才进入
  `PRODUCTION_PROMOTION`，并按当前协议逐段通过数据、离线质量、INT8、设备事件和发布
  审查；任一门未通过都只阻止晋级，不追溯否定论文 Development 结果。
- 公开视频银标、反事实和生命周期 r7.* 工作属于受限研究证据。它们可以在披露来源、
  标签类型和限制后用于 `THESIS_DEVELOPMENT`，但不能仅凭单次实验授权 production
  training、blind 评测、Android runtime 或默认模型替换。
- Corridor-Causal Student 仅完成 benchmark-only 特征与性能可行性检查；缺少完整、隔离的 GPT/Codex 连续事件共识 receipt，`96 episode / 48 matched pair` 门仍阻塞训练和事件效果评测。旧日期化快照中的“双人人工复核”要求由 [GPT / Codex 自主复核治理](AI_REVIEW_GOVERNANCE.md) 取代。
- 其他研究线不在本页复制动态阶段、终态或下一步：[RCLE current 入口](research/rcle/README.md)拥有 RCLE 当前真相，[USTRF route-conditioned closure R1](research/ustrf-sc/USTRF_ROUTE_CONDITIONED_PROGRAM_CLOSURE_R1_2026-07-25.md)保留历史收口边界。两者的研究证据均不自动改变 SANPO 门禁、正式 App、默认模型或产品权限。

## 当前操作入口

| 问题 | 当前真源 |
| --- | --- |
| 全项目实验模式与证据等级 | [渐进式研究治理](RESEARCH_GOVERNANCE.md)：新工作默认 `THESIS_DEVELOPMENT`；最终 Confirmation 或生产晋级需显式激活 |
| GPT/Codex 自主复核与仲裁 | [AI_REVIEW_GOVERNANCE.md](AI_REVIEW_GOVERNANCE.md) |
| Development 训练与生产训练隔离 | [SANPO_TRAINING_PROTOCOL.md](SANPO_TRAINING_PROTOCOL.md)：先选 `THESIS_DEVELOPMENT` 或 `PRODUCTION_PROMOTION` lane |
| 候选晋级与设备事件门 | [SANPO_CANDIDATE_PROMOTION_GATES.md](SANPO_CANDIDATE_PROMOTION_GATES.md)：只约束 `PRODUCTION_PROMOTION` |
| 连续序列评测和 baseline | [SANPO_SEQUENCE_EVALSET.md](SANPO_SEQUENCE_EVALSET.md)、[SANPO_TRAVERSABILITY_BASELINE.md](SANPO_TRAVERSABILITY_BASELINE.md) |
| 反事实采集与生命周期目标 | [SANPO_COUNTERFACTUAL_EPISODE_COLLECTION.md](SANPO_COUNTERFACTUAL_EPISODE_COLLECTION.md) |
| Corridor-Causal 候选的本轮结论 | [CORRIDOR_CAUSAL_PROGRESS_2026-07-20.md](CORRIDOR_CAUSAL_PROGRESS_2026-07-20.md)：仅工程可行性，未获得事件效果或晋级授权。 |
| RCLE 暂停与历史终态 | [RCLE current 入口](research/rcle/README.md)：暂停、历史终态和未来恢复边界只在该入口维护，本页不复述 |
| 历史 USTRF 边界 | [USTRF route-conditioned program 收口 R1](research/ustrf-sc/USTRF_ROUTE_CONDITIONED_PROGRAM_CLOSURE_R1_2026-07-25.md)：仅作为历史收口入口，不形成 SANPO 或产品权限。 |
| 公开银标与来源研究 | 仅在对应协议已登记为 `current` 后按其执行；未提交的本地草稿不能作为仓库规则或授权依据。 |
| 最近研究证据 | 已提交的日期化 snapshot；未提交的研究记录保持任务本地状态，待其所属任务完成后再登记。 |

## 硬边界

- 不以 benchmark-only、oracle、未绑定的单次模型标签、单一来源或事后压力样本冒充可部署结论；只有满足独立多模型 receipt 的 workflow 才获得其明确 authority。
- `WILD_LAB` 允许候选超出现有 Android 延迟、模型大小和 YOLO 抽象；这些约束在探索阶段
  作为记录的工程属性，不是算法假设的前置禁令。若进入模型替换或产品路径，再切换到
  `EVIDENCE_TRACK / PRODUCTION_PROMOTION`。
- `THESIS_DEVELOPMENT` 不得声称默认模型替换、真实用户安全或生产能力，但也不得因缺少
  INT8、blind、设备事件或发布 receipt 被判为实验不可运行。
- `PRODUCTION_PROMOTION` 不绕过数据集根门、blind 隔离、哈希/许可/隐私证据或既定
  晋级门；这些门不倒灌为 Development 的前置条件。
- 若本文件与可复现门禁报告或 current 协议不一致，停止升级操作，先修正本文件并记录证据链接。

## 更新规则

只有以下事实改变时更新本页：默认模型/产品路线、门禁状态、授权边界、当前阻塞点或下一道可执行门。逐轮实验数字、失败细节和素材发现进入日期化 snapshot 与 `DEVELOPMENT_LOG.md`，本页只保留链接和操作结论。
