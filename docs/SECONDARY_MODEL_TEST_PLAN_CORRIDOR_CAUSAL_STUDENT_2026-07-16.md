# 第二模型测试方案：Corridor-Causal Student

## 文档状态

- 方案编号：`secondary_corridor_causal_v1`
- 状态：`proposal_only / not_started`
- 定位：独立于当前 SANPO 分割、P3 数据闭环及公开视频银标主线的第二测试方案
- 权限：仅允许进入独立的 benchmark-only 实验通道
- 当前结论：`do_not_replace_default_model`

本方案不替换、不暂停、也不修改当前正在推进的模型优化方案。它只保存一个可独立启动、独立归因和独立停止的第二候选方向；后续若执行，代码、数据、权重、报告和真机资产必须与主线隔离。

## 一、核心判断

当前单帧 `MobileNetV3Small + LR-ASPP` 路线主要回答“每个像素是什么”，但实际提醒决策还需要回答：

1. 用户当前准备向哪里移动；
2. 对象位于目标走廊内、边缘还是走廊外；
3. 对象正在接近、保持、通过还是远离；
4. 这是新事件、被遮挡的旧事件，还是已经提醒并应清除的事件；
5. 信息不足时应提醒、拒判还是提示调整相机。

第二方案因此把主要学习目标改为：

> **相机运动补偿后的走廊相对关系 + 因果风险生命周期。**

像素分割、对象检测、边界和离线深度只作为输入证据或辅助监督，不再单独决定是否提醒。

## 二、与当前主线的隔离合同

| 项目 | 第二方案约束 |
|---|---|
| 默认 App 模型 | 保持现有 `yolo11n_fp16_320.tflite`，不得替换 |
| SANPO 主线 | 不修改其 canonical、P3 split、训练授权、阈值或结论 |
| blind holdout | 禁止训练、选特征、选阈值和失败挖掘访问 |
| 模型资产 | 不写入 `app/src/main/assets` |
| 本地产物 | 统一写入 `artifacts.local/experiments/secondary-corridor-causal/<run-id>/` |
| 评价 | 使用独立报告；不得把主线结果拼接成第二方案通过证据 |
| 晋级 | 仍须依次通过离线事件门、INT8 保真门、同机事件门和隔离 GPT/Codex 自动发布准入 |

如果后续实现需要修改共享 runtime 或 benchmark，只允许增加默认关闭的实验入口，并保持生产行为和主线结果不变。

## 三、候选架构

```text
RGB 10 FPS + 可选 IMU/相机运动
  -> 现有 YOLO 检测证据 + MobileNet 多尺度空间特征
  -> heading / vanishing-point / optical-flow 运动补偿
  -> 走廊坐标空间网格
  -> 4–8 帧 history-only causal TCN
  -> corridor relation + lifecycle + risk curve + abstain
  -> 现有 RiskEventTracker / FeedbackPlanner 安全门
```

### 3.1 第一版输入

- MobileNet OS8 空间特征；不得先做全局平均池化；
- YOLO box/mask 类别、置信度、中心点、底边位置和尺度；
- 当前分割 logits，仅作为辅助空间证据；
- 相邻帧 warped IoU、中心/底边位移、面积变化和 flow consistency；
- 行进方向或消失点；本地采集允许加入同步 IMU，公开视频无 IMU 时必须显式标记；
- capture quality、semantic unknown 和 model uncertainty 必须保持不同字段。

### 3.2 第一版输出

- `corridor_relation`：`outside / edge / intruding`；
- `lifecycle`：`non_alert / approach / alertable / post_event`；
- `hazard_family`：平行边界、台阶/路沿、中心障碍、侧向行人或电动车；
- 连续 `risk_score` 与 episode 级 `should_alert`；
- `abstain_reason`：至少区分 semantic unknown、motion unknown、capture missing 和 capture degraded。

### 3.3 规模目标

- 第一版复用现有 backbone，只训练小型事件头；
- 事件头目标参数量不高于约 `0.3M`，最终以实际图和模型检查为准；
- 只使用 LiteRT/TFLite 友好的 Conv、Conv1D、pool、concat 和基础激活；
- 目标为 full INT8；同机总 P95 不高于 `70 ms`，相对当前 YOLO 增量不高于 `15 ms`。

该规模是工程预算，不是已验证性能结论。

## 四、训练目标

### 4.1 主监督

主监督单位是连续 episode 和物理风险事件，不是独立帧：

- episode 级 hazard、corridor relation 和 `should_alert`；
- 逐时刻 `approach / alertable / post_event`；
- `first_visible`、`alertable_start`、`passed_or_cleared`；
- matched positive/negative 共享 `matched_pair_id`；
- 同一物理事件在短时遮挡和语义翻转后保持同一 ID。

### 4.2 损失候选

第一轮只允许一次增加一个损失：

1. corridor relation 与 lifecycle 分类损失；
2. matched-pair ranking：同上下文正例风险高于负例；
3. approach 窗口内的 risk future-regularization / 单调趋势约束；
4. post-event 风险衰减和及时清除约束；
5. 分割或几何边界辅助损失。

禁止在第一轮同时加入 SAM/ASAM、复杂半监督正则、多个蒸馏头和新 backbone，否则无法判断提升来自哪里。

## 五、离线教师与前沿项目的角色

| 项目/论文 | 本方案允许用途 | 禁止用途 |
|---|---|---|
| [VPSeg](https://openaccess.thecvf.com/content/CVPR2024/html/Guo_Vanishing-Point-Guided_Video_Semantic_Segmentation_of_Driving_Scenes_CVPR_2024_paper.html) | 消失点、径向运动和走廊坐标设计参考 | 把驾驶域结果当作步行域通过证据 |
| [RiskProp](https://openaccess.thecvf.com/content/CVPR2026/html/Zou_RiskProp_Collision-Anchored_Self-Supervised_Risk_Propagation_For_Early_Accident_Anticipation_CVPR_2026_paper.html) | 事件前风险传播和趋势约束参考 | 直接照搬驾驶事故模型或跨过 post-event 强制单调 |
| [Video Depth Anything](https://github.com/DepthAnything/Video-Depth-Anything) | 离线生成时序深度梯度、表面变化和 TTC-like 诊断特征 | Android 在线并行深度主链、把相对深度称为米制距离 |
| [EdgeTAM](https://github.com/facebookresearch/EdgeTAM) | 离线 mask 传播、短时身份和遮挡上界 | 直接作为无提示的生产语义模型 |
| [DINOv3](https://github.com/facebookresearch/dinov3) | 离线 dense matching、unknown 和 failure mining | 直接触发提醒；未完成许可证审核前进入生产资产 |
| [PIDNet](https://github.com/XuJiacong/PIDNet) / MobileNetV4 / RepViT | 第二阶段 backbone A/B 候选 | 在走廊时序表征未证明有效前直接替换主干 |

大模型、VLM、深度和开放词汇输出可以作为披露 provenance 的离线 teacher、model-consensus reference 或 provisional evidence，不能被表述为客观传感器事实、真人用户效果，也不能直接触发安全提醒。

## 六、分阶段可证伪实验

### S0：确定性表征 probe

在不训练 backbone 的前提下，同数据、同 split、同特征抽样比较：

1. `static_pooled_baseline`；
2. `motion_geometry_only`：运动补偿 + 显式几何趋势；
3. `corridor_grid_temporal`：空间网格 + 运动几何 + 小型 causal TCN。

进入 S1 的最低条件：

- source-grouped balanced accuracy `>= 0.70`；
- positive/negative recall 均 `>= 0.50`；
- matched-pair 增量方向一致性达到预注册门；
- 同一 source 的全部 episode 保持在同一折；
- 不读取 blind，不通过同视频切窗增加虚假独立样本数。

如果第三组不优于第一组，立即停止第二方案，不转向超参数大扫或直接换 backbone。

### S1：事件头短跑

- 使用完整反事实 episode 矩阵和 leave-one-session-out；
- 冻结 backbone，只训练事件头；
- 固定 5 个预注册 seed；
- 报告均值、标准差、最差 seed、最差 session、最差 scene；
- 像素 mIoU、boundary IoU 和逐帧 recall 只作诊断。

### S2：单因素结构 A/B

只有 S1 证明事件表征有效后，才允许单独比较：

- 当前 MobileNetV3 特征；
- MobileNetV4 或 RepViT；
- Mobile-PID-lite；
- PIDNet-S 端侧基线。

候选必须先过冻结特征 probe，再进入训练。GPU FPS、iPhone/CoreML 或厂商 NPU 数据不得代替目标 Android 设备实测。

### S3：INT8 与同机事件门

最终候选须绑定模型 SHA256，并依次验证：

- FP32/Torch 与导出图等价；
- full-INT8 输出保真；
- 同一 SM-S9280、同一脚本、同一连续 evalset A/B；
- 生产 APK 不含候选模型资产；
- 最终结果保持 `replace_default_model_now=false`，直到隔离 GPT/Codex 发布收据与所有自动门禁通过。

## 七、晋级指标

首轮事件硬门：

| 指标 | 阈值 |
|---|---:|
| `event_recall` | `>= 0.90` |
| `critical_miss_rate` | `<= 0.05` |
| `false_alerts_per_minute` | `<= 0.50` |
| `delivered_repeated_alert_rate` | `<= 0.10` |
| `post_event_clearance_rate` | `>= 0.90` |
| 同机总 P95 | `<= 70 ms` |
| 相对当前 YOLO 增量 P95 | `<= 15 ms` |

同时必须报告：

- `late_alert_rate`、`lead_time_ms`；
- `clearance_latency_ms`；
- `event_regeneration_rate`；
- `suppressed_duplicate_attempts_per_event`；
- parallel boundary、step/curb、center obstacle、lateral cut-in 分项；
- capture missing/degraded 与 unknown 的错误安全声明率。

## 八、停止条件

出现任一情况即停止本方案并保留负向报告：

1. 运动/走廊表征不能提高 source-grouped 可分性；
2. VEC、mask 平滑或 mIoU 提升，但事件召回、误提醒和清除不改善；
3. 转弯、俯仰或相机抖动导致走廊坐标系统性失效；
4. matched negative 压制真实 lateral cut-in 或关键台阶提醒；
5. 依靠频繁 unknown/abstain 降低误报，同时关键事件召回下降；
6. teacher 输出泄漏到 blind、标定或冻结评价答案字段；
7. INT8 或端侧 P95 超出预算；
8. 需要修改默认 App 行为才能证明离线收益。

## 九、首个执行包

若未来启动第二方案，第一批工作只包含：

1. 冻结输入/输出 schema 和实验目录；
2. 实现无训练的 heading/flow corridor probe；
3. 在现有非 blind、source-grouped episode 上生成三组 S0 报告；
4. 根据 S0 的 go/no-go 决定是否实现 causal TCN trainer。

第一执行包不训练新 backbone、不导出模型、不连接真机、不修改 App、不复制任何资产。它的唯一目标是证明“走廊条件化时序表征”是否比静态池化更可分。

## 十、决策摘要

本方案保留当前主线的来源、blind、量化和真机安全门，但改变候选模型的主要学习目标：

> 从“识别像素类别”升级为“判断某个物理事件正在如何进入用户的目标走廊，以及何时提醒、何时清除”。

只有 S0 先证明这种表征在独立 source 上可分，第二方案才获得训练资格。在此之前，它只是一个独立、可证伪、不会干扰当前主线的测试提案。
