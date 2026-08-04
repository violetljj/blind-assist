# HFTF fresh metric snapshot 三层侵入 R0 协议

日期：2026-08-05

状态：`FROZEN_BEFORE_SOURCE_COLLECTION_OR_QNN_OUTCOME`

## 唯一问题

在手机 QNN 米制深度刚完成的同一 CameraX 帧上，继承的 foot/body/head 人体扫掠包络，
能否在独立物理真值下比 ground-only 二维净空更可靠地表达碰撞侵入，同时保持
`UNKNOWN != CLEAR`？

本轮不传播深度，不使用光流、历史米制状态、Track、语义、ToF、ARCore depth、NPU
调度或提醒。三层表示是本项目已经建立的原创 HFTF 基础贡献；本 successor 只评价
fresh 手机米制深度与真实受控障碍真值的新增结合，不在论文内部重复计算 HFTF 创新增量。

## 新来源

现有 R4 只提供合成障碍与解析地形 mechanics；既有 75 帧手机数据镜头高度约 0.15 m，
且 Samsung Quick Measure 不是独立真值；D45 的 ARCore raw depth 时间戳也未通过。
这些来源全部不得回救本轮。

新 cohort 使用 `SM-S9280`、现有 CameraX 同帧 QNN 深度支线和刚性 tripod/头高支架。
每个 parent session 必须完全重置支架与场景，镜头高度实测在 1.35–1.75 m，误差不超过
1 cm，pitch/roll/yaw 对齐误差不超过 2°。相机内参必须在 formal cohort 前单独标定并
hash 绑定，标定重投影误差不得超过 0.50 px。

障碍物使用不透明哑光表面；其距离、横向范围和垂直范围分别由卷尺/刚尺、激光测距、
水平仪和 fiducial pose board 建立，距离误差不超过 2 cm，高度与横向误差不超过 1 cm。
支撑结构必须位于全部被评价人体包络之外。物理真值在第一份 QNN output 前封存，不得
使用 QNN、Samsung Quick Measure 或 ARCore depth 放置、筛选或重标障碍。

## 固定场与 cohort

方向中心固定为 `-25/0/+25°`，距离中心固定为 `1.0/1.5/2.0 m`。继承：

- foot `[0.05,0.35) m`，有效半宽 0.30 m；
- body `[0.35,1.35) m`，有效半宽 0.40 m；
- head `[1.35,2.05] m`，有效半宽 0.28 m。

六类场景为完全畅通、仅足部、仅躯干、仅头部、多层同时侵入、左右不同高度竞争。
每类精确 3 个独立 session，每 session 10 个重复 snapshot，共 18 sessions / 180
snapshots。frame 只是 session 内重复量测，不得当作独立样本。18 个 session ID 已在相邻
JSON 中冻结；任何 QNN output 打开后不得替换失败 session。

## 三臂

- `B0_GROUND_ONLY_2D`：只使用足部地面净空与连续性；
- `B1_HEIGHT_COLLAPSED_3D`：相同 fresh 3D 点和最大人体包络，只输出任意高度侵入；
- `C1_HEIGHT_STRATIFIED`：相同 fresh 3D 点，输出 foot/body/head 分层侵入。

B1 是公平性 guard：C1 的总体 any-intrusion 不应因为分层而改变。C1 真正新增的是高度
归属和层级化风险表达；B0 比较地面净空对悬空/躯干危险的 false-clear。

## 评价与门

主单位是 parent session，同时报告 frame count。必须报告 false-clear frame/event、各层
precision/recall/F1、层混淆、UNKNOWN 原因、macro/worst-session known coverage、左右
相反错误、最近侵入 clearance MAE/P90、B1-C1 any-risk agreement 和 B0-C1 false-clear
reduction。

支持要求全部同时满足：各层 precision/recall 与 macro F1 均不低于 0.90；层混淆不高于
0.10；false-clear session event、UNKNOWN→CLEAR 和左右相反错误均为 0；macro known
coverage 不低于 0.80、worst-session 不低于 0.60；clearance MAE/P90 不高于
0.25/0.50 m；B1-C1 any-intrusion agreement 不低于 0.98；C1 相对 B0 false-clear rate
至少下降 0.30。

缺 session、类别、物理真值、同帧 QNN depth 或必要机会时返回 `NOT_EVALUABLE`。正式
结果打开后不得更改 roster、层高、人体宽度、真值、阈值、门或聚合。即使通过，也只
允许另行冻结 periodic metric anchor utility 协议，不直接授权 PMAF Track、调度、App、
提醒、导航、生产或安全主张。
