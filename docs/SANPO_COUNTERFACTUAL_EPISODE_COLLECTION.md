# SANPO 交叉反事实事件采集协议 v1

## 目的与边界

本协议把数据单位定义为连续 `episode` 和物理 `risk_event`，而不是相互高度相关的独立帧。首轮冻结为 **6 个 session × 4 个 scene × 每格 2 个 positive + 2 个 matched negative**，合计 96 个 episode、48 个正负 matched pair。

本轮只定义可离线执行的采集、标注和评价合同，不下载数据，不训练模型，不读取既有 blind holdout，也不授权导出、设备接入或替换 App 默认模型。像素 mask 可作为少量辅助证据，但不再是事件晋级的唯一真值。

机器可读合同位于 `configs/sanpo_counterfactual_episode_collection_v1.json`。配置中的 session 是待采集槽位，不代表数据已经存在或通过复核。

## 采集矩阵

六个 session 均须在同一设备、相机配置和连续采集上下文内覆盖以下四类 scene：

1. `parallel_boundary`：平行路沿/边界不应提醒；匹配的正例为边界进入或横断中心走廊。
2. `step_curb`：接近台阶或路沿为正例；已经通过、正在远离或仅位于远场为匹配负例。
3. `center_obstacle`：障碍侵入中心近场走廊为正例；外观相似但未侵入或仍处远场为匹配负例。
4. `lateral_pedestrian_or_ebike`：真实切入路径为正例；仅侧向经过、不进入路径为匹配负例。

每个 session-scene 单元必须包含两个 matched pair。每对由一个 positive 和一个 negative 组成，尽量保持地点、光照、相机姿态、物体类别和拍摄时段一致，只改变与提醒相关的运动或空间关系。不得用不同地点、昼夜或设备来制造容易识别的“伪反事实”。

每个 episode 建议持续 10–20 秒，保留采集前后上下文，不把同一连续事件裁成多个训练样本。原始视频只归属于一个 session；衍生帧、光流、mask 或 clip 不得跨 session/fold 复用。

## 事件标注

positive episode 至少包含：

- `risk_event_id`：同一物理事件稳定不变；
- `first_visible_ms`：风险对象首次可审计地出现；
- `alertable_start_ms`：第一次允许系统提醒的时刻；
- `passed_or_cleared_ms`：事件已通过、远离或清除的时刻；
- `expected_should_alert=true`。

三个时间点必须满足 `first_visible <= alertable_start < passed_or_cleared`。建议双人独立复核至少 20% 的 positive；`alertable_start` 或 `passed_or_cleared` 分歧超过 500 ms 时须仲裁。

negative episode 使用 `expected_should_alert=false`，三个正事件时间点必须为 `null`，并提供 `negative_reason`。matched pair 必须共享 `matched_pair_id`，同时记录允许保持不同的风险几何字段，避免把负例伪装成与正例完全相同的事件。

人工标注的是“是否应提醒和何时应提醒”，不是证明系统已经安全。模型输出、伪标签或已有提醒日志不得直接充当事件真值。

## 来源、许可、隐私与哈希

每个 session 必须具有本地 `source_receipt`，至少记录：

- 数据拥有者/来源、采集日期和许可状态；
- 隐私复核状态、人物可识别信息处理方式和复核者；
- 原始视频相对路径、字节数和 SHA256；
- 设备、相机参数和任何重编码/裁剪派生关系；
- episode manifest 与标注文件 SHA256。

配置和 manifest 不依赖网络 URL 才能验证。远程出处可以作为说明字段，但本地证据、许可文本、隐私记录与哈希 inventory 必须足以独立完成审计。缺许可、隐私状态不是 green、SHA256 缺失或哈希不匹配时，该 session 整体不得进入训练或评价。

## 切分与随机审计

采用固定 leave-one-session-out 六折。每折完整留出一个 session；其原始视频、episode、衍生资产、标签和 matched pair 均不得出现在该折训练、校准、阈值选择或主动选样中。不得按帧随机切分，也不得把一个 matched pair 拆到不同数据权限区域。

主动失败队列必须保留至少 20% 的固定随机审计样本。随机样本从尚未进入候选队列的有效长序列中抽取，并与模型分歧、高置信误提醒和事件再生候选分开报告。holdout session 不参与主动选样。

## 评价与门槛

每折以及六折汇总必须报告：

- `event_recall` 与 `critical_miss_rate`；
- `false_alerts_per_minute`；
- `delivered_alerts_per_event` 与 `delivered_repeated_alert_rate`；
- `post_event_clearance_rate`、`clearance_latency_ms`；
- `event_regeneration_rate`；
- worst-session、worst-scene 和 matched-negative false-alert breakdown。

首轮成功门冻结为：每个 fold 的 event recall ≥0.90、critical miss rate ≤0.05、false alerts/min ≤0.50、delivered repeated alert rate ≤0.10、post-event clearance rate ≥0.90。逐帧 recall、mIoU、boundary IoU 和 session identity probe 只作诊断；它们不能单独授权晋级。

若关键漏报增加、真实切入事件被反事实约束压制、事件指标不随像素指标改善、session identity 显著可预测、随机审计持续发现主动队列漏掉的系统性失败，或事件时间点无法稳定复核，则停止扩训并回到采集/标注合同修订。

## 采集安全

采集应在静态或受控路线中进行，由健视安全员负责环境控制和停止权。不得要求盲人参与者依赖未验证候选完成风险路线，不得把采集结果描述为临床、人因或真实助盲安全验证。本项目始终是辅助提醒原型，不替代盲杖、导盲犬、人工判断或专业辅助设备。
