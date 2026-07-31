# YOLO + 语义分割图像空间互补性 Development 设计 R0

状态：`DEVELOPMENT_STANDARD / DESIGN_ONLY / NOT_EXECUTED / NO_EFFECT_AUTHORITY`

日期：2026-07-31（Asia/Hong_Kong）
执行者：`violjjet`

## 1. 研究问题与允许主张

研究问题：在同一 RGB 帧和同一 YOLO 输出上，语义分割是否产生了**不被 YOLO
检测框覆盖的稳定图像空间候选区域**？

本设计只回答 mechanism-level 的 image-space complementarity，不回答：

- 区域是否现实不可通行；
- 区域是否是风险事件或需要提醒的障碍；
- segmentation 是否改善事件召回、误提醒、反馈或安全；
- C 是否应该进入 Android、默认模型或生产。

中央阻塞 Agent 标签、D0-A1/D0-A successor observation、risk、feedback 和任何
人工/Agent 语义真值均不在输入面。

## 2. 设计状态、数据角色和候选身份

这是一个新的、一次性的 `DEVELOPMENT_STANDARD` 设计，不是 D0-A readiness successor，
也不增加 D0-A2/A3/A4 或第三 Agent 裁决层。当前仅冻结设计，未执行效果对照，因为
current entry 仍把 D0-B 模型执行、融合和事件增量评价标为 `NOT_AUTHORIZED`。

若获得独立执行授权，第一份机制 diagnostic 使用一套已存在的 matched Development
输入：

| 项目 | 值 |
| --- | --- |
| RGB manifest | `artifacts.local/evidence/dual-loop-r1-unseen-natural-event-r0/rank2-shiraz/input-10hz-r1/manifest.jsonl` |
| RGB manifest SHA256 | `af0ab3c735d96737f451a6e64d1784681966345c7849131ad51bd46c9d7e6571` |
| YOLO reference trace | `artifacts.local/evidence/dual-loop-r1-unseen-natural-event-r0/rank2-shiraz/device-r1/baseline-output/trace.jsonl` |
| YOLO trace SHA256 | `b9b1b55890e08fd268cb7d650954651a923fc75c9d537bb1e24721deb5753e9b` |
| 输入规模 | 4,891 个 10 Hz RGB frame / 一个 Shiraz source session |
| segmentation reference | `sanpo-v3-pretrained-weighted-best-int8-20260713.tflite` |
| segmentation SHA256 | `88f0184d2671230c1f1f43192758689d286b530d7490e1d1ca0671f83b50b50c` |
| segmentation 角色 | 已存在 benchmark-only reference；不是生产替换或正式模型选择结果 |

选择该 reference 的理由仅是它已有固定 INT8 合同和非退化的 SANPO dev mask 报告；
这不是多个模型的排名，也不把 dev mIoU 外推到 Shiraz 或新双环效果。
初始 `gpu-smoke` artifact 的 `walkable=100%` 塌缩结果保留为单独负诊断，不覆盖。

同一 24-slot technical diagnostic 的本地产物为
`artifacts.local/evidence/dual-loop-segmentation-technical-smoke-r0/pretrained-reference-diagnostic.json`
（SHA256：`1ad6833f7b7b8feffcef243f9961d9e8d841c37c8f287f67a92feaba22e59124`）。其 argmax
像素分布为 `walkable=39.50%`、`boundary_step_curb=0.97%`、`obstacle=6.10%`、
`unknown_nonwalkable=53.43%`；这只证明输出没有像初始 artifact 那样完全塌缩，仍不构成
目标来源上的分割质量、类别选择或互补性结论。

## 3. 配对、观测单位与缺失处理

- 基本观测单位是一个 `source_id + frame_id + image_sha256` 配对 frame；不是独立
  样本，不能把 4,891 帧当成 4,891 个独立重复。
- 每个 frame 必须同时存在 RGB、YOLO trace 和 segmentation 输出；RGB SHA、frame id、
  source timestamp 不一致时整行 `NOT_EVALUABLE`，不通过插值、滑窗或最近帧补齐。
- A/B/C 在同一 frame 上配对：A 只用 YOLO box union，B 只用 segmentation mask，C
  是二者的几何 union。三组不重新选择 detector 阈值、NMS、风险阈值或类别映射。
- 主要汇总按 session 先聚合再等权汇总；frame-level 数字只作为描述，避免长序列
  通过帧数支配结果。
- 该 Shiraz session 曾用于既有 Development route，因此只可作为 burned mechanism
  diagnostic，不能作为新来源 Confirmation 或泛化效果证据。

## 4. 冻结的图像空间 estimand

在归一化到固定分析栅格 `W×H` 后，定义：

```text
D_t       = union of all YOLO detection rectangles at frame t
S_t,k     = argmax segmentation mask for class k at frame t
U_t,k     = S_t,k \ D_t
C_t       = D_t union (union over k of S_t,k)
```

其中 `k` 分别保留四类 `walkable`、`boundary_step_curb`、`obstacle`、
`unknown_nonwalkable`；不把后三类合并成“风险”。主要互补量为：

```text
uncovered_fraction_t,k = |U_t,k| / |Ω|
union_increment_t      = (|C_t| - |D_t|) / |Ω|
```

`Ω` 是完整分析图像，不使用中央 ROI，不把下半视野包装成路线真值。下半视野或
连通区域只能作为预注册的次要描述，不能替换 primary estimand。

每个 session 报告：

1. A 的 detector-covered fraction；
2. B 的各类 segmentation fraction；
3. C 相对 A 的 `union_increment`；
4. 每类 `uncovered_fraction` 的 frame 分布和 session median；
5. 相邻时间戳的 `IoU(U_t,k, U_{t-1,k})`、非空比例和连通组件数量，描述时间稳定性；
6. 缺失、非有限、空 mask 和运行耗时分母。

不预注册 p-value 或“通过/失败”风险门。若以后需要不确定性，按 session/source
cluster 做 bootstrap 或混合效应汇总，不能按 frame 独立抽样；当前无事件真值，不能
选择事件级检验、显著性门或因果模型。

## 5. A/B/C 输出和停止规则

| 臂 | 输入 | 允许输出 | 禁止解释 |
| --- | --- | --- | --- |
| A | YOLO boxes | box coverage、box count、时序描述 | 已发现全部障碍 |
| B | segmentation mask | per-class mask coverage、component、host/device cost | 可通行性或风险真值 |
| C | A 与 B 的几何 union | `U_t,k`、union increment、稳定性 | 融合后风险改善、提醒改善 |

立即停止当前 candidate 的条件：接口/有限值失败、输入 pairing 破坏，或输出在
整个可评价输入上退化为单一类别而无法产生可解释的 mask 分层。停止只关闭该
candidate/evidence version，不关闭语义分割问题本身。

即使 `U_t,k` 非零且稳定，也只能进入下一次明确授权的融合设计；不能把非零增量
直接写成“分割发现了有效障碍”。如果 held-out source 上无法复现同一 image-space
增量，则关闭该 candidate 的互补性主张。

## 6. 当前终点

当前为 `DESIGN_ONLY / NOT_EXECUTED`：

- technical smoke 已证明一个 reference 接口可运行，但初始 artifact 在 smoke input
  上发生 `walkable` 塌缩；
- pretrained reference 只完成了非持久的类别可用性诊断，尚未形成 A/B/C 结果；
- matched YOLO trace 的 risk/feedback/event 字段不在本设计输入面；
- D0-B 效果、融合、Android 和生产权限保持关闭。

下一条唯一合理动作是：在 current entry 另行批准后，运行这一份固定设计的 image-space
mechanism diagnostic；不再增加 Agent 标签、中央阻塞 prompt、第三 Agent 或 readiness
子阶段。
