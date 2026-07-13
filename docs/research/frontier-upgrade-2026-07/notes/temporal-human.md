# 时序、事件层与真实用户证据精读笔记

## 0. 阅读范围、证据等级与结论先行

本任务包逐页核对了 7 篇本地全文，共 `66/66` 页：STEPP 7 页、DTERN 11 页、BOFP 11 页、AI Guide Dog 8 页、Escalator Problem 9 页、VisAssist 9 页、CLIP-BLV 11 页。方法、实验、局限页逐页精读；纯参考文献页用于来源和论文谱系核对。论文 PDF、逐页文本、页数和 SHA256 以 `refs/paper-inventory.json` 为准。

先给工程结论：下一次跨越不应是“让分割 mask 更平滑”或“让 VLM 直接判断是否报警”，而应把系统拆成三个不同时间尺度：

1. **10–15 FPS 的安全感知层**：语义/边界、光流或低成本运动证据、质量与未知状态，只输出证据，不直接播报。
2. **事件生命周期层**：用因果历史维护 `APPROACHING -> ALERTED -> PASSED_OR_RECEDING -> CLEARED`，同一事件最多接受一次用户提醒；遮挡、标签抖动和短时丢失不能创建新事件。
3. **低频交互层**：VLM 只用于解释、拍摄调整建议、离线标注和非紧急问答，不得直接触发安全告警。

这一判断来自两组相互补强的证据：DTERN/BOFP 证明“时序一致性必须同时约束语义有效性与遮挡失真”；VisAssist/CLIP-BLV 证明“真实 BLV 相机分布、缺失信息和质量退化会使通用视觉语言模型产生显著性能差距和错误自信”。本地 90 帧连续序列则进一步证明：当前主要失败不是没有检测到风险，而是登阶后重复提醒和边界误升级，亦即**事件身份、阶段和清除**问题。

### 0.1 会议级别与可外推边界

| ID | 发表状态 | 直接证据对象 | 可以支持 | 不能支持 |
|---|---|---|---|---|
| STEPP | ICRA 2025 accepted manuscript；本地为 arXiv 作者稿 | 人类行走轨迹训练、ANYmal 机器人真实室内外实验 | 轨迹投影正样本、异常分数、语义与几何联合 | BLV 用户安全、手机端实时性；机器人可通行性不等于人类助盲风险 |
| DTERN | ICCV 2025 主会 | VSPW、Cityscapes 视频分割 | 局部/全局 exemplar、有效时序一致性 VEC | Android 端性能、助盲事件指标、低算力可部署性 |
| BOFP | WACV 2024 主会 | Cityscapes、CamVid、VSPW | 遮挡感知的特征传播、三目标权衡 | 在线手机部署；双向未来帧不能用于实时告警 |
| AIGD | AAAI Spring Symposium 2025 full paper，非 AAAI 主会 | 8 名以研究生/实习生为主的参与者，戴黑眼镜模拟目标情境 | 手机路径方向分类、意图融合、端侧工程参考 | 真实 BLV 用户有效性、临床/安全有效性、近场障碍告警 |
| ESCALATOR | ICCV 2025 workshop **position paper**，非主会算法论文 | 示例性质的 MLLM 失败 | 低信号连续运动的问题定义、以信任/安全为目标的评测倡议 | 算法增益、失败发生率、用户研究结论 |
| VISASSIST | AAAI 2026 主会 | 真实视障志愿者采集的 13,413 段视频 | BLV 第一视角分布、缺失信息/质量退化、视频 QA 失败 | 实时导航安全；论文未报告真实行走干预效果与参与者数量分布 |
| CLIPBLV | CVPR 2024 主会 | ORBIT 67 名 BLV 用户、VizWiz 超过 11,000 名 BLV 用户 | 通用 CLIP 在真实 BLV 数据上的 QoS 差距及成因 | 直接证明 BlindAssist 分割失败；研究任务是分类/下游示例，不是连续导航 |

## 1. BlindAssist 当前事件证据的关键缺口

本地合同已经把 `eventAlertRecall` 定义为：同一 `risk_event_id` 的期待提醒窗口中任一帧成功提醒即事件命中；这是正确的主门口径，逐帧 recall 继续作为诊断。当前还存在四个需要先修正的度量或状态语义：

1. `repeatedAlertCount` 当前统计的是 `FeedbackReason.EVENT_ALREADY_ALERTED`，即**重复提醒尝试被成功抑制的帧数**，并不是用户实际收到的重复提醒数。晋级合同中的 `repeated_alert_rate` 应拆成：
   - `delivered_repeated_alert_count = Σ_event max(0, delivered_alerts(event)-1)`；
   - `suppressed_duplicate_attempt_count`，保留现有计数用于诊断上游抖动。
2. `alertFalsePositiveRate` 是非提醒帧上的比例，设备晋级合同却要求 `false_alerts_per_minute`。连续序列应同时报告两个口径，后者按有效视频时长归一化，不能互相替代。
3. 当前 `RiskEventTracker` 仅用 `label + center-x delta` 关联事件，并在 3 帧 receding/missing 后清除。标签翻转、短遮挡或中心漂移可能把同一物理障碍拆成新事件；相同标签的相邻障碍也可能被合并。
4. `passedWindowFalseAlertCount` 已能观察事件后仍提醒，但还缺少**清除时延分布**和**错误再生事件数**。90 帧证据中“登阶后 receding 段重复提醒”说明必须直接测量这两个量。

因此，论文中的时序一致性不能只作为 mask 指标移植；必须最终映射到事件身份、首次提醒、清除和再生。

## 2. 逐篇证据卡

### 2.1 STEPP：从“走过的地方”学习可通行性与异常

- **问题与机制**：用人类真实行走和 UE 合成轨迹作为正样本，将未来 40 个位姿投影到图像；SLIC 生成 400 个超像素，DINOv2-small 产生 `50×50×384` 特征，区域平均后由 7 层 MLP 自编码器重建。低重建误差代表熟悉的可走区域，高误差代表陌生或危险区域（pp.3–5）。
- **主要证据**：500 张未见森林图像上用人工区域设定阈值；Table I（p.5）报告 real forest `0.803`、simulated `0.684`、forest+indoor `0.769`、全部数据 `0.835`。混入不相关室内实验室数据反而使森林准确率下降，说明“更多数据”不等于更好域泛化。ANYmal 室内迷宫和森林实机案例显示语义代价能补充纯高度规划（pp.5–6）。
- **局限**：阈值 `0.35` 在同一批 500 张标注区域上调到最优，未展示独立阈值校准集；准确率不是事件安全指标。系统仅 `2.5 Hz`，依赖 ZED2 深度、LiDAR/SLAM、NUC 和 Jetson Orin；论文还明确报告相似图像代价波动、未见地形误判、SLIC 分组错误、下中部位置偏置和深度标定依赖（p.6）。
- **证据边界**：这是机器人实机证据，训练中的“人类行走”不等于真实 BLV 用户研究。ANYmal 能越过的草地、坡地和人类手持手机的安全通行条件不同。
- **可迁移机制**：只迁移“正样本轨迹投影 + 异常辅助分数”。在 BlindAssist 中，已确认的 `walkable corridor` 可作为正样本；高异常只能转成 `unknown_motion_or_surface`，不能自动转成 obstacle/alert。
- **最小实验**：离线冻结 DINO/轻量特征 teacher，仅对 canonical train 的走廊区域拟合一个重建或距离分数，在 dev 上测试：是否能区分 `walkable`、`unknown_nonwalkable` 与 `step_curb`，并对 worst-session 提供增益。端侧第一轮不部署 teacher，只蒸馏一个标量 anomaly head。
- **停止条件**：若异常分数只能区分“训练域/新域”而不能区分安全/危险，或明显提高 unknown 覆盖但 unknown precision/recall 任一退化，则停止；若蒸馏后端侧增量超过既有 P95 预算，也只保留为离线数据筛查器。

### 2.2 DTERN：有效一致性比“看起来平滑”更重要

- **问题与机制**：LTEM 从四帧 `{t-9,t-6,t-3,t}` 构造局部 cluster exemplars，GTEM 用可学习的 frame/video cluster centers 聚合全局时序信息；目标是缓解光照、遮挡和形态变化造成的帧间同类特征偏差（pp.3–5）。
- **主要证据**：VSPW 上 MiT-B0 从 SegFormer 的 `33.1 mIoU / 30.1 VEC8 / 31.3 VEC16 / 96.1 FPS` 提升到 `35.9 / 33.4 / 34.5 / 31.3 FPS`；即有效一致性提高，但吞吐约降至三分之一，且参数从 3.8M 增至 8.2M（Table 1, p.6，RTX 4090）。Cityscapes MiT-B0 从 71.9 提到 74.5 mIoU（Table 2, p.7）。
- **VEC 的价值**：旧 VC 对“全零但始终不变”的预测可给出满分；VEC 把连续交集与类别 IoU结合。Table 5（p.8）中，VEC 与人工 ground-truth temporal consistency 的 Pearson/Spearman 约 `88`，旧 VC 约 `59–61`；这直接支持“不以平滑掩盖错误”。
- **局限**：GTEM 的消融只提高 VEC8/VEC16 各 `0.2`，同时 mIoU 从 40.4 降到 40.3，作者解释为全局上下文稀释细节（p.7）。论文无助盲、移动端或因果在线事件评测；RTX 4090 FPS 不能外推手机。
- **可迁移机制**：优先迁移 VEC 指标和“原型数量应适中”的结论，而不是整网。Table 4（p.8）显示 cluster 数过少、等于类别数或远大于类别数都会退化；轻量事件层可维护 4–8 个风险原型，而不是高维全局 memory。
- **最小实验**：在现有 10 FPS 连续集上增加 `VEC8/VEC16` 作为诊断，并实现仅使用历史帧的 causal exemplar：对 `walkable/boundary/obstacle/unknown` 的 logits 或低维 features 做置信度加权原型更新；当前帧与原型不一致时降低晋级置信度，不直接覆盖当前证据。
- **停止条件**：若 VEC 上升但 `eventAlertRecall`、critical event miss、passed-window clearance 或实际重复提醒无改善，判定为“只做了视觉平滑”并停止；若细 `boundary_step_curb` 被原型平均掉，也立即回退。

### 2.3 BOFP：遮挡区域需要特殊处理，但双向只能作离线上界

- **问题与机制**：深网络只跑 keyframe，FlowNet 将高层特征向中间帧前后传播；OANet 读取四个双向光流，估计遮挡/反遮挡注意图，对不可信区域更多采用当前帧浅层特征（pp.3–5）。
- **主要证据**：Cityscapes 上 HRNetV2-W18 基线 `75.9 mIoU / 81.0 mTC / 18.9 FPS`，BOFP 为 `75.7 / 86.5 / 19.7 FPS`；Xception-71 版本 mTC 从 76.6 提到 84.8，mIoU 仅降 0.1（Table 1, p.7）。消融中 forward-only `72.0 mIoU / 171 ms`，加入 backward 为 `75.9 / 196 ms`，完整 OANet+current 为 `76.5 / 265 ms`（Table 3, p.8）。
- **局限**：完整方法需要未来 keyframe 和反向传播，在线告警时不可用；论文测试为驾驶/通用视频分割，不是 egocentric BLV 行走。其 mTC 也可能奖励稳定错误，不能替代 VEC 或事件指标。
- **可迁移机制**：把双向 BOFP 作为**离线上界/teacher**；线上只做 forward-only：上一帧风险区域经轻量 flow/box motion 前向投影，使用 forward-backward consistency 的离线 teacher 或当前帧重检测置信度估计 occlusion。遮挡区优先“保留事件身份但降低语义信心”，而不是清除后重建事件。
- **最小实验**：先在录制序列离线比较三组：无传播、forward-only、bidirectional upper bound。只评价事件身份切分数、passed 后再生事件数、event recall、false alerts/min 与 P95；mask mIoU/mTC 仅诊断。
- **停止条件**：若双向上界相对无传播仍不能降低事件切分/重复提醒，说明瓶颈不在遮挡传播，停止移植；若 forward-only 引入 critical miss、平均首次提醒变晚，或端侧增量超过预算，回退为仅用于离线标签修复。

### 2.4 AI Guide Dog：方向预测可作低频意图，不是安全告警器

- **问题与机制**：将未来 1 秒动作抽象为 FRONT/LEFT/RIGHT 多标签分类，室内无目的地允许多个方向，室外把 Google Maps 指令和 GPS 作为 intent embedding；用前 5 秒（10 帧）上下文（pp.2–4）。
- **数据与结果**：57 小时、392,580 个样本、8 名参与者，场景跨 Pittsburgh/Seattle/Bay Area，并按场景隔离 train/validation/test（p.3, Table 1）。最佳 CNN+LSTM+Intent 在混合测试集 LEFT/RIGHT recall 仅 `0.559/0.583`，F1 `0.6324/0.6621`；FRONT 明显更高（Table 3, p.6）。iPhone 13 使用 FP16、2 FPS，推理频率 2 Hz（p.7）。
- **核心证据边界**：参与者主要是研究生和技术实习生，戴黑眼镜、慢走模拟目标场景；这不是 BLV 用户证据。论文没有真实 BLV 行走试验、碰撞/近失事件或信任测量。黑眼镜模拟不能重现长期的 cane 技能、残余视力、听觉策略和真实相机持握分布。
- **可迁移机制**：路径方向是**低频意图先验**，可帮助事件层判断“用户是否正朝风险区域移动”，但不能取代近场障碍感知。GPS/Maps 只用于高层意图，作者自己指出约 4.9 m GPS 精度不够局部导航（p.4）。
- **最小实验**：从现有稳定器导出 1 秒走廊方向/光流趋势，作为风险融合的弱先验；只允许它降低侧向非侵入风险或选择提示方向，不允许单独产生安全提醒。
- **停止条件**：若方向先验对 center-risk/event recall 无增益，或导致侧向真实切入漏报，则停止；2 Hz 路径模型不得进入近场 critical alert 路径。

### 2.5 Escalator Problem：有价值的问题定义，弱算法证据

- **论文定位**：作者明确称其为 position paper；没有新模型、数据集或标准化大样本实验（pp.1–2）。示例展示若干领先 MLLM 对两台相反方向扶梯的回答错误，但没有报告模型版本、样本量、重复次数、提示模板和置信区间。
- **有价值的论点**：稀疏帧抽样会在输入阶段不可逆地丢失低信号连续运动；扶梯、旋转门、自动门、传送带、 crowd flow 等任务的关键信息存在于帧间物理变化，不在单帧语义（Table 1, p.6）。Table 2（p.8）把目标从第三人称动作分类转向第一人称连续流上的信任、可靠性和安全。
- **不能外推**：论文不能证明所有 MLLM 都“100% 失败”，也不能证明光流或 event camera 已解决问题；它提供研究假设和 benchmark 设计，不是算法有效性证据。
- **可迁移机制**：构建“隐式运动小套件”：扶梯方向/静止、旋转门状态、自动门起动、横向行人/电动车、登阶后的 receding。每类必须包含静态外观几乎相同、运动方向相反的 hard pair，防止模型靠场景外观猜测。
- **最小实验**：比较单帧、0.5 FPS、1 FPS、10 FPS flow、10 FPS causal feature state；报告方向正确率、错误自信率、abstain quality、event lead time 和端侧延迟。
- **停止条件**：任何依赖 VLM 文本回答的路径若不能在信息缺失时稳定 abstain，或其 95% 上界仍超过安全允许错误率，就只能作为解释层；不得直接触发告警。

### 2.6 VisAssist：真实 BLV 视频首先要求“知道画面里没有答案”

- **数据贡献**：13,413 段真实视障志愿者视频，共 137,554.64 秒、5,465,939 帧，平均 10.26 秒；17.25% 视频含拍摄调整建议，约 10% 不包含可回答关键信息（pp.2–3）。标注至少由两人独立完成，冲突由第三人仲裁；问题覆盖物体、文本、位置、深度和拍摄调整（pp.2–3）。
- **主要证据**：零样本模型在 depth 和 direction 上普遍最弱。Gemini-Pro 在 0–5 评分下 Avg `3.30`，Qwen2.5-VL `2.37`，ChatGPT-4o `2.76`；即最佳结果也远非可靠（Table 2, p.5）。反射、远距离、缺失/遮挡画面易诱发上下文驱动的合理但错误回答（Table 3, p.5）。
- **时序/成本证据**：1 FPS 通常比单帧/0.5 FPS 更准，但 3090 上开放模型延迟约 2.15–5.26 秒；FSM 最多选 3 帧、平均 2 帧，可接近 0.5 FPS，但弱模型可能因选错关键帧而更差（Table 5, p.7）。448 分辨率普遍更准但成本更高（Table 6, p.7）。
- **局限**：论文是开放式 VideoQA，不是连续导航；评分由 LLM 语义指标 COR/DO/SU 给出，不是碰撞、漏报或用户任务成功。参与者数量和 BLV 人群分层未在正文中报告；地理和文字以中文场景为主。没有手机端实时部署或前瞻性用户试验。
- **可迁移机制**：增加独立的 `capture_quality` 门：`ANSWER_VISIBLE / RECOVERABLE_QUALITY / INFORMATION_MISSING`。后两者不得生成“前方安全”或具体距离；应触发短促的相机调整提示，如“镜头向下/左移、放稳”，并限制提示频率。
- **最小实验**：对连续集人为生成 blur、glare、crop-out、rotation、低照度和关键区域遮挡；比较当前置信度与 quality head。主指标是 missing-information recall、错误安全声明率、调整后恢复率，而不是单纯分类准确率。
- **停止条件**：若 quality head 不能把“低质量但仍有信息”与“关键信息完全缺失”分开，不能让它解除或确认风险；只允许输出 generic reacquisition 提示。

### 2.7 CLIP-BLV：通用预训练规模不能自动覆盖 BLV 分布

- **研究设计**：25 个 CLIP 变体，覆盖多种 ViT、大规模预训练集和 80M–3.8B 数据量；在 ORBIT、VizWiz-Classification 与 MSCOCO/Open Images 上用统一 episodic zero-shot 分类，并用 logistic regression 分离质量因素（pp.2–4）。
- **真实 BLV 证据**：ORBIT 包含 67 名 BLV 用户、3,822 段视频、2.68M 帧、486 个物体；VizWiz-Classification 来自超过 11,000 名 BLV 用户的真实辅助应用（p.3）。这是本任务包中最强的真实 BLV 相机分布证据，但仍是分类审计，不是导航干预。
- **主要结果**：BLV 数据平均准确率 `51.5%`，web 数据 `66.5%`，差 15 个百分点（Fig.1, p.1）。ORBIT Clean 中 disability objects 比 non-disability objects 低 17.1 点，Clutter 中低 25.1 点（Table 1, p.4）；大模型/更多 web 数据不能消除差距。预训练 caption 中 disability objects 比普通对象少约 16–17 倍（Table 2, p.5）。
- **质量退化**：blur、viewpoint、occlusion、lighting 的平均边际影响在 ORBIT 分别约 `-11/-9/-9/-23` 个百分点；影响可叠加，且更大预训练集不保证鲁棒（p.6）。这要求 BlindAssist 按质量和用户分层报告，而不是只给总平均。
- **个性化边界**：5-shot ProtoNet 在 ORBIT Clean 把 disability/non-disability gap 降至 2.1 点，但 Clutter 仍有 14.6 点；更多于 5 shots 很快饱和（Table 3, p.5）。因此少样本个性化可做，但不能被宣传为复杂真实场景的充分修复。
- **可迁移机制**：建立 `BLV capture audit`：按 blur、viewpoint、occlusion、lighting、framing、设备、用户、场景和 disability-specific object 分组；所有 teacher/VLM/foundation features 都必须单独给出这些 worst-group 指标。
- **最小实验**：从真实或经授权 BLV 风格视频建立不进训练的审计切片；对 segmentation/anomaly/VLM explanation 分别报告 worst-group event recall、false alerts/min、unknown recall 与 calibration。可探索 5-shot 用户/设备适配，但必须 session 隔离。
- **停止条件**：若个性化只改善 Clean、在 Clutter/worst-user 无改善，或需要访问 blind holdout 调参，立即停止晋级；任何 CLIP/VLM teacher 的置信度都不能绕过 event gate。

## 3. 建议的事件层：Causal Evidence-to-Event Gate

### 3.1 数据流

```text
RGB 10–15 FPS
  -> mobile segmentation / detector
  -> quality head + unknown evidence
  -> causal motion evidence (flow/track/area-bottom trend)
  -> event association + phase state
  -> one-event-one-alert gate
  -> speech/haptic feedback

VLM / large teacher
  -> offline annotation, failure mining, optional low-frequency explanation
  -X-> direct safety alert
```

### 3.2 事件身份

事件关联键不能只靠 label 和 center-x。建议的物理事件相似度包含：

`category family + mask/box warped IoU + center/bottom displacement + approach trend + corridor overlap + short-gap age`。

- 标签在 `boundary_step_curb <-> obstacle <-> unknown` 间翻转时，只要几何轨迹连续，就保持同一事件 ID，并更新 evidence distribution。
- 短时遮挡/丢失进入 `OCCLUDED_HOLD`，保留已提醒状态但禁止新提醒；超过预注册 gap 才清除。
- `PASSED_OR_RECEDING` 必须由连续 receding、走廊退出或深度远离共同确认；清除后若同一轨迹重新出现，应记录 `event_regeneration`，供诊断是否误拆分。
- 同一事件第一次反馈被实际接受后锁定。后续高风险帧可更新 UI/震动保持状态，但不再播报第二条同类提醒；只有明确新事件 ID 才重新提醒。

### 3.3 证据与未知状态分离

当前 `unknown_nonwalkable` 是像素语义 abstain，事件层还需要区分：

- `SEMANTIC_UNKNOWN`：模型不知道是什么表面/障碍；
- `MOTION_UNKNOWN`：光流、遮挡或趋势不足，不能判定接近/远离；
- `CAPTURE_MISSING`：目标区域不在画面或关键视觉信息不存在；
- `CAPTURE_DEGRADED`：模糊/逆光但仍可能恢复。

这些状态不能统统映射为 obstacle。安全策略应为：高置信危险证据可提醒；不确定时不得宣称安全；`CAPTURE_MISSING/DEGRADED` 走相机调整提示，且与障碍告警使用不同语音/震动编码。

## 4. 人因与事件级评测指标

### 4.1 必须晋级的机器指标

| 维度 | 建议指标 | 说明 |
|---|---|---|
| 事件发现 | `eventAlertRecall`, `criticalEventMissCount` | 同一事件任一有效窗口提醒一次即命中；保持当前主门 |
| 及时性 | `lead_time_ms`, `first_alert_frame`, `late_alert_rate` | 仅“最终提醒”不够，需保证可反应时间 |
| 打扰 | `false_alerts_per_minute` | 与视频有效时长绑定，保留 frame FP 作诊断 |
| 重复 | `delivered_repeated_alert_rate` | 实际交付的第 2 次及以后提醒；与 suppressed attempts 分开 |
| 上游抖动 | `suppressed_duplicate_attempts_per_event` | 反映事件层挡住了多少重复尝试，数值高仍说明感知不稳 |
| 事件清除 | `post_event_clearance_rate`, `clearance_latency_ms`, `event_regeneration_rate` | 直接覆盖当前登阶后 receding 失败 |
| 有效时序 | `VEC8/VEC16` | 只作 mask/语义诊断，不可替代事件指标 |
| 未知与质量 | unknown precision/recall、missing-info recall、错误安全声明率 | 不允许全 unknown 或从不 unknown 刷分 |
| 分组稳健性 | worst-user/session/quality/device/scene | 对应真实 BLV 分布与当前 worst-session 合同 |
| 端侧 | P50/P95、增量 P95、热稳定、电量 | 双向/大模型结果不能外推手机 |

### 4.2 真实 BLV 参与评测

进入用户阶段前，必须先由真实 BLV 用户和无障碍专家共同定义任务、提示语言与停止协议；戴眼罩的 sighted participant 只能做早期可用性 smoke，不能替代目标用户证据。建议分三阶段：

1. **桌面/坐姿录像回放**：参与者判断提示是否清楚、是否过载，不涉及行走风险。
2. **受控室内路线**：有安全员、白杖/导盲犬照常使用，评估 one-event-one-alert、相机调整提示和恢复。
3. **真实路线 shadow mode**：系统先记录不播报，与人工事件标注对照；安全门全绿后才讨论提示试验。

除机器指标外记录：任务完成率、碰撞/近失事件、停顿与绕行次数、错误依赖、主观信任校准、理解时间、认知负荷、提示可区分性和相机调整成功率。不能只问“喜欢不喜欢”；应把“高信任但错误依赖增加”视为失败。

## 5. 最小实验矩阵与停止规则

| 阶段 | 变更 | 数据 | 成功判据 | 立即停止条件 |
|---|---|---|---|---|
| T0 | 修正重复提醒指标语义；新增 actual repeat、suppressed attempts、clearance latency、regeneration | 现有 90 帧 + v3 连续集 | 指标可由逐帧日志确定性重算；同事件一次提醒合同可测试 | 指标仍混淆“尝试”和“交付”，不得进入模型比较 |
| T1 | DTERN-inspired causal logit/exemplar，历史帧 only | canonical dev 连续 session | worst-session/VEC 提升，event recall 不降，boundary 不被平均 | 只提高 VEC，事件指标不变；step/curb 召回下降 |
| T2 | BOFP bidirectional offline upper bound | 固定录制序列 | 事件切分/再生明显下降 | 上界都无收益，停止 flow 路线 |
| T3 | forward-only occlusion hold | 同 T2 + 真机 | 接近 T2 事件收益且满足 P95 门 | critical miss/late alert 增加，或 P95 越界 |
| T4 | capture-quality/missing-info head | 合成退化 + 真实 BLV 审计切片 | missing recall、错误安全声明、恢复率改善 | 无法区分 missing 与 degraded；不得解锁安全判断 |
| T5 | 低频 intent/path prior | 室内/室外连续路线 | 降低侧向/平行误报且 center event recall 不降 | 侧向切入漏报或 2 Hz 先验进入 critical path |
| T6 | VLM explanation shadow | VisAssist 风格视频、非实时 | 能正确 abstain，解释不改变安全决策 | 任何 VLM 输出直接触发安全告警 |

所有实验都必须遵守现有 `offline_training_quality -> int8_fidelity -> device_event` 三段门，不读取 blind holdout 调参。BOFP 双向结果、DTERN 4090 FPS、STEPP Jetson/机器人结果只能作为上界或机制证据，不能作为端侧通过证据。

## 6. Evidence–Claim Map

| Source ID | 全文证据 | 可写入报告的论点 | 建议引用位置 | 风险/限定 |
|---|---|---|---|---|
| LOCAL-05 + EventAlertMetrics | 90 帧 88.9% 危险提醒召回但 25.9% 错误提醒；事件命中按任一帧计算 | BlindAssist 当前瓶颈是事件生命周期与错误提醒，不是单纯逐帧召回 | 当前瓶颈、事件层设计 | 数据仍小；需 v3/blind 扩展 |
| DTERN pp.6–8 | VEC 避免全零平滑作弊；与人工 TC 相关性高于 VC | 时序指标必须同时考虑语义正确与连续性 | 评价方法 | 通用 VSS，非助盲 |
| BOFP pp.5–8 | 遮挡注意提高 mTC、保持近似 mIoU；双向需未来帧 | 遮挡应保留事件身份；双向只能离线上界 | 时序模块 | 双向非因果、非手机 |
| STEPP pp.3–6 | 轨迹正样本、异常重建；mixed 0.835，2.5 Hz 与多项限制 | 走过轨迹可训练异常辅助，但异常不等于危险 | 长期路线/teacher | 机器人证据，阈值校准不独立 |
| AIGD pp.3,6–7 | 模拟参与者数据；最佳转向 recall 约 0.56–0.58；iPhone 2 Hz | 路径方向只适合低频意图先验，不能承担近场安全 | 交互/意图路线 | 非真实 BLV，无安全用户试验 |
| ESCALATOR pp.5–8 | 稀疏采样遗漏低信号连续运动；提出人本 benchmark | 静态语义充分不代表动态安全充分 | 研究空白/benchmark | position paper，无定量算法证据 |
| VISASSIST pp.2–7 | 13,413 真实 BLV 视频；depth/direction 弱；缺失信息导致错误回答；延迟秒级 | VLM 不能直接告警；必须有 missing-information 与拍摄恢复门 | 人因、质量门 | VideoQA、LLM 评分、非导航试验 |
| CLIPBLV pp.1,3–8 | BLV-web 差 15pp；真实用户数据；质量因素与 disability content 差距 | 所有 foundation/VLM 路线必须做 BLV capture 分组审计 | 数据与公平性 | 分类审计，不是连续导航因果证据 |

## 7. 自检与未决问题

- [x] 7/7 PDF 元数据与 inventory 匹配；逐页文本 `66/66` 页可读。
- [x] 明确区分 ICCV/CVPR/WACV/AAAI 主会、AAAI Spring Symposium、ICCV Workshop position paper 和 ICRA accepted manuscript。
- [x] 明确区分真实 BLV 数据、戴黑眼镜模拟、普通视频数据和机器人实机证据。
- [x] 没有把 VLM 放入直接安全告警路径；双向 BOFP 仅作为离线上界。
- [x] 每篇均记录机制、数值证据、局限、迁移点、最小实验和停止条件。
- [x] 事件层指标以 `eventAlertRecall` 为主，逐帧/VEC/mTC 仅诊断；识别出 `repeatedAlertCount` 当前命名与实际语义不一致。
- [x] 建立 evidence–claim map，关键“VLM 不可直接告警”由 VisAssist 与 CLIP-BLV 两条真实用户分布证据共同支撑。
- [ ] VisAssist 正文未给出志愿者数量、视力分层和设备分布，报告不可自行补写。
- [ ] 本地仍缺真实 BLV 连续行走的事件级数据；在获得目标用户证据前，任何用户安全结论必须保持 `not_evaluated`。
- [ ] T0 指标修正、T1/T2/T3 实验尚未实施；本文件是可执行研究设计，不是模型晋级证明。
