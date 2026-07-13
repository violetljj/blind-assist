# 最终报告证据与推理质量复审

## 最终结论

**PASS（有两项非阻塞文字修正建议）。**

上一轮 Q1–Q6 已实质修复：报告已纳入 P1 完成及失败结果，OS4/OS16 不再作为待执行路线，Mobile-PID-lite 已降为结构重入门后的条件候选；近期执行顺序与仓库最新合同统一为 P2、D0、T0、U0；论文总页数、最新 gate SHA、HUMAN/MACHINE 派生审计边界和 evidence-map citation slots 均已修正。

当前版本能够作为“研究机制与执行顺序报告”交付，但仍不能被解读为新模型已晋级或安全有效性已得到证明。

## 拒绝性检查

| 检查 | 结果 | 复审依据 |
|---|---|---|
| 每个核心论点至少有一条强证据 | **PASS** | P1/P2 状态由 P0、P1 本地审计直接支撑；边界、uncertainty、事件层和 BLV 分布论点均有正式论文与本地合同交叉证据。 |
| 研究空白有交叉证据 | **PASS** | seed 稳定性、标签质量、时序事件、calibration 和真实 BLV capture 均同时列出本地失败与论文机制，不再以单篇论文构造空白。 |
| 未用平均 mIoU 或 GPU FPS 外推安全/手机性能 | **PASS** | 报告明确要求 worst-seed/session/scene、INT8、设备 P95 和事件门；PIDNet、Mobile-Seed、DTERN、BOFP、STEPP、VisAssist 的平台结果均未被当作手机证据。 |
| 未把模拟参与者当成真实 BLV 用户 | **PASS** | AI Guide Dog 被明确标注为 8 名 sighted 模拟参与者，只支持低频 intent prior；VisAssist/CLIP-BLV 也只支持真实 BLV 分布审计，不被外推成导航干预安全。 |

## 上轮必须修复项复核

| 编号 | 状态 | 复核结果 |
|---|---|---|
| Q1 P1 状态冲突 | **已修复** | 报告 §2、§5、§8、§9 和 LOCAL-06 均写明 P1 已完成；最佳值 `0.4642/0.5235`、range `0.2951`、OS4/OS16 失败与 P1 审计一致。 |
| Q2 路线违反进入条件 | **已修复** | Mobile-PID-lite 已降为条件候选；未通过结构重入门时先 P2，仍高方差则进入 I0，不再直接实施 E2。 |
| Q3 页数错误 | **已修复** | 报告改为 14 篇、162 页，与 inventory 的 14/14、162 页及三份笔记 `19+77+66` 一致。 |
| Q4 evidence map 不可追踪 | **已修复** | slot 已重写到现有 §2、§3.1–§3.6 和 §5 的 P1/P2/I0/E2/R/U/T/H 阶段；抽查均有对应标题或表格行。 |
| Q5 最新来源闭环锚点不足 | **已修复** | 报告和 LOCAL-01 已绑定 real-only r3、400/200/120、14 session、10/10 green、training gate `4c68e434...` 和 build/assembly report `f7f7b11e...`；旧 `32968...` 仅作历史对照。 |
| Q6 HUMAN/MACHINE 被写成正式合同 | **已修复** | 已明确为一次性本地派生审计，写出 reviewed manifest `image_sha256` 到 14 个 draft manifest `source.sha256/source_annotation_quality` 的回连，注明不适用于 evidence-v4，且尚非 schema/sidecar/gate。 |

## 强支撑与证据边界

### 强支撑

- P0 固定 sampler 的 model-seed range `0.2685`、固定 model seed 的 sampler range `0.0112`、约 `24.1×` 和五组最弱场景均为 `step_curb`，与本地审计一致。
- P1-A 最佳 mIoU/boundary `0.4642/0.5235`、model-seed range `0.2951`、OS4 两个 boundary IoU `0.0271/0.0130`、OS4/OS16 与 OS8/OS16 最佳 selection `0.0968/0.1549`，与 P1 审计一致。
- 三段晋级门和阈值由仓库合同及脚本支撑，报告没有把跨后端 green 或单 seed 最佳值误写为 INT8/设备晋级。
- 90 帧 `88.9%/25.9%` 已明确标注为 SANPO oracle v2 规则实验、非训练模型/blind 证据，并保留“小规模、未晋级”边界。
- PIDNet/Mobile-Seed、ValUES/Kandinsky、DTERN/BOFP、VisAssist/CLIP-BLV 分别形成结构、不确定性、时序和用户分布的交叉证据链。

### 仍属待证伪假设

- P2 deterministic quota sampler 是否能提高 worst-scene、worst-seed 或缩小 range 尚无运行结果；它只是关闭可控覆盖变量的当前主线。
- Mobile-PID-lite、HRFP、UPC-lite、SWSEG、causal event association、capture-quality head 和 Kandinsky-style abstain 均未在 BlindAssist 上证明有效。
- P0 只能把高方差定位到初始化及与 model seed 绑定的 Torch 随机状态，不能声称已证明“纯初始化因果”；2×2 交互仍未完成。
- 30% range 降幅、16 通道 D-lite、固定空间 cluster 等是工程预注册起点，不是论文通用阈值。
- 四个新增 calibration session 只足够经验校准；连续帧不能冒充独立样本，也不能据此声称 session-level 95% conformal guarantee。

## 错误外推与不安全建议复核

未发现阻塞性交叉外推或不安全建议：

- 没有以 Cityscapes/GTA/VSPW 平均 mIoU 推断 `step_curb`、真实 BLV 或事件安全收益。
- 没有以 GPU、Jetson、机器人或 iPhone 2 Hz 指标替代目标 Android 同机 P95。
- 没有把 geometric boundary、uniformity、VEC、mTC、coverage 或 entropy 单独当作生产晋级依据。
- 双向 BOFP 仅为离线上界；VLM、CLIP、STEPP anomaly 和低频路径 prior 均不得直接触发近场告警。
- simulated sighted participant、真实 BLV capture benchmark 和真实导航干预研究三者已经分开。
- HUMAN donor 只有 38 张、donor 垄断、硬 CutMix 伪边界和高置信 pseudo error 均已作为 UPC 风险保留。

## 数字、链接与映射抽查

| 项目 | 结果 |
|---|---|
| 论文 inventory | 14 篇、162 页，报告一致 |
| P0 数字 | `0.4424-0.1739=0.2685`；`0.4424-0.4312=0.0112`；约 `24.1×`，一致 |
| HUMAN/MACHINE | train `38+362=400`，dev `39+161=200`，blind `30+90=120`；train/dev MACHINE `523/600=87.2%`，一致 |
| 最新数据闭环 | training gate `4c68e434...`、build report `f7f7b11e...` 可在 DEVELOPMENT_LOG 中交叉定位 |
| INT8 阈值 | argmax `0.995`、逐类 prediction IoU `0.97`、逐类 GT IoU drop `0.02`、mean IoU drop `0.01`，一致 |
| device 阈值 | event recall `0.90`、critical miss `0.05`、false alerts/min `0.50`、clearance `0.90`、repeat `0.10`、P95 `100 ms`，一致 |
| 相对链接 | inventory、三份 notes、evidence map 和新增 P1 审计链接均存在 |
| evidence slots | 所有抽查的 §/阶段 ID 均存在；不存在上一版的 §4.4、§4.5、§6 E0/E1 等失效定位 |

## 非阻塞剩余问题

1. 报告 §2 的“这说明 quota sampler 仍值得保留”容易让读者误以为 P0 已使用 deterministic quota sampler。P0 实际审计的是当前 sampler RNG，P2 才是待实现的确定性 quota sampler。建议改成“现有 session-balanced/rare-class 采样逻辑仍值得保留，但下一步将其改造成确定性 quota”。后续段落已正确说明 P2 尚未执行，因此该措辞不改变当前路线判定。
2. LOCAL-03 的“P1 必须优先验证”是 P0 时点的历史结论，而 LOCAL-06 已记录 P1 完成。建议在 LOCAL-03 加“P0 当时结论”标签，避免单独摘录时被误读为当前待办。

## 交付边界

本次 PASS 仅表示报告的事实、证据映射、推理边界和执行优先级已达到可交付水平。它不授权：

- 导出 INT8 或运行设备晋级门；
- 替换 App 默认模型；
- 启动未满足前置条件的 UPC/SWSEG、Mobile-PID 或 conformal calibration；
- 将 benchmark-only、oracle、模拟参与者或离线论文结果写成真实用户安全结论。

后续只有 P2/D0/T0/U0 产生新证据后，才应更新此报告的状态性结论。
