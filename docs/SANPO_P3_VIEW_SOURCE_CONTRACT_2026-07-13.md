# SANPO P3 视角与新增来源契约（2026-07-13）

## 决策

P3 的目标域改为**前向自我视角 RGB 导航**，不再把 SANPO `camera_chest/left` 当成唯一可用视角。官方 SANPO 候选允许 `camera_chest/left`（优先）和 `camera_head/left`（无合格 chest 标注时回退）；同一个 native session 即使有两套相机也只能选择一个视角，且绝不能跨 train/dev。

公开街景/车载资料只能进入 B 层辅助预训练或受控对照，**不得**进入 P3 session 覆盖、P3 dev、blind 或最终离线晋级指标。真正补足 P3 每场景 `4–6 train + 2–3 dev` 独立 session 的新增路径是官方 SANPO head-left 与经同意的胸前/手持手机前向序列。

机器可读合同见 [sanpo_p3_view_source_contract_20260713.json](../configs/sanpo_p3_view_source_contract_20260713.json)。

## SANPO 视角门

- 只接受官方 `left` segmentation view；`right` 一律拒绝。
- 以对齐 mask inventory 选择视角，不能从 session description 推断。
- 首选 `camera_chest/left`，不足时回退 `camera_head/left`。
- head-left 先作为独立 cross-view candidate；在 chest-only 与 head-only 分层 dev 门均预注册通过前，不得静默混入 chest canonical。
- split 主键固定为 `source_id:native_session_id`，而不是 camera；同一 session 不得将 chest/head 拆入不同 split。
- sequence 内 camera、lens、原始宽高、source split、native session 和 annotation quality 必须恒定。接受 SANPO 的 `2208×1242` 或 `1920×1080`；训练 resize 不得覆盖原始尺寸证据。
- canonical 必须保存 camera、lens、原始宽高、view ID 与 annotation quality，并分层报告 chest/head 的像素分布、dev 指标及 camera×scene 覆盖。

SANPO 论文说明：237 个带 panoptic segmentation 的 session 中 146 个为 chest、91 个为 head；继续锁死 chest 会理论上丢失约 38.4% 的有标注 session。发现器现默认 chest 优先、head 回退。

## 新来源分层

| 来源 | 层级 | 当前结论 |
|---|---|---|
| SANPO-Real v0 head-left | A：P3 canonical 候选 | 已批准，遵循同一官方 train、CC BY 4.0、隐私和 split 合同。 |
| 经同意的胸前/手持手机前向序列 | A：P3 canonical 候选 | 必须有真实主体的采集同意凭证，以及自动残余 PII 检查、模型共识像素标注、会话隔离与多模型 scene 复核。 |
| Mapillary Vistas | B：辅助预训练候选 | 下载前必须保存数据集专用许可快照并完成商业/再分发核验；单帧不能补 P3 session。 |
| BDD100K / IDD / ACDC | 未批准候选 | 当前许可未核验，禁止下载或纳入训练。 |
| Cityscapes | 研究对照 | 官方许可限非商业；禁止进入产品训练路径或 P3。 |
| GuideTWSI procedural | C：程序化增强 | 不能替代真实 session。 |

Mapillary 的通用公开影像采用 CC BY-SA，但其条款说明数据集可能有优先适用的独立许可，因此不得以通用条款代替 Vistas 下载页的专用许可。[Mapillary 许可说明](https://help.mapillary.com/hc/en-us/articles/115001770409-CC-BY-SA-license-for-open-data)；[Mapillary 条款](https://www.mapillary.com/legal/terms)。Cityscapes 官方许可明确限非商业用途。[Cityscapes 许可](https://www.cityscapes-dataset.com/license/)。

## 执行顺序

1. 用更新后的 chest-priority/head-fallback 发现器重新扫描 SANPO official-train；候选仍须依次通过 16-frame、50-frame、RGB、PII、人工 scene 和 P3 planner。
2. 建立经同意手机序列的 intake receipt 与像素级标注队列；只有 A 层可满足 P3 session 数量与最终门。
3. 固化 Mapillary Vistas 专用许可、类别映射、隐私和分发审计后，才可作为 B 层预训练候选；B 层不得改变 P3 dev/benchmark 结论。

未读取 blind 标签，未下载任何外部新数据，未启动训练或改变 App。

官方 train 的完整候选发现现已完成；其完整性、候选分布和仍未解除的门见 [full discovery record](SANPO_P3_OFFICIAL_TRAIN_FULL_DISCOVERY_2026-07-13.md)。
