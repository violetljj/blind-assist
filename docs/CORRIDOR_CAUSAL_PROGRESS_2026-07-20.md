# Corridor-Causal Student 进度快照（2026-07-20）

状态：snapshot
范围：独立的 `secondary-corridor-causal` benchmark-only 研究通道。
关联方案：[SECONDARY_MODEL_TEST_PLAN_CORRIDOR_CAUSAL_STUDENT_2026-07-16.md](SECONDARY_MODEL_TEST_PLAN_CORRIDOR_CAUSAL_STUDENT_2026-07-16.md)。

## 结论先行

- 已排除两个工程可行性风险：候选可从既有 YOLO 检测和同步 IMU 形成 8 帧因果特征；未训练的 INT8 TCN 夹具在 SM-S9280 的真实 CameraX + YOLO 热路径中通过了 `<= 70 ms` 总 P95 性能门。
- 没有获得模型效果结论：特征、TCN 和公开 ADVIO 序列尚未提供按冻结合同审计的 `should_alert`、生命周期或 matched-negative 自动多模型事件参考。因此没有校准、blind 评测、Android shadow、提醒路由或默认模型替换授权。
- 当前阻塞点是自动多模型事件证据覆盖而不是端侧算力。96 个 episode / 48 个 matched pair 的槽位仍全部为 `awaiting_autonomous_acquisition`；在来源 Agent/自动设备/仿真生成的数据完成隔离双模型复核与必要的第三模型裁决前，S0 可分性和后续事件头训练保持关闭。

## 本轮完成的独立证据

| 事项 | 结果 | 可说明的范围 | 不可说明的范围 |
| --- | --- | --- | --- |
| YOLO + IMU 因果特征 | Android 特征提取器输出 `[4,4,8]` 空间网格、`[20]` 运动向量和 `[8,...]` history-only 窗口；7 项定向测试通过。纯特征构造最坏 P95 `78.958 us`。 | 特征契约可在目标设备运行。 | 固定图像走廊不是路线、风险真值或告警规则。 |
| 公开同步 RGB + IMU 机制夹具 | ADVIO 60 帧 / 10 Hz 形成 53 个完整窗口；每帧都有 YOLO 检测，IMU 旋转值非零。 | 输入传输不只依赖合成检测框。 | ADVIO 是 CC-BY-NC 机制来源，不含风险事件标签，不能进入训练或事件门。 |
| 特征 + INT8 TCN 组件 | 62,689 参数未训练 TCN 的特征、窗口、量化和推理组合 P95 `0.3155 ms`。 | 小事件头本身未耗尽组件预算。 | 不包含相机/YOLO，且未训练输出没有风险语义。 |
| 完整实时性能夹具 | 三次成功运行的总 P95 为 `69.567 / 68.829 / 68.799 ms`，最坏值距 `70 ms` 门余量 `0.433 ms`；资源清理竞态修复后均为 0 失败。随后复用 TCN 输入/输出缓冲区的复核运行 P95 为 `69.016 ms`、0 失败。 | 当前计算形态没有立即超过目标设备总延迟预算。 | 余量很窄；训练权重、真实 IMU 对齐、归一化或任何运行时接入变化均须重新全链路测量。 |

详细本地产物位于：

- `artifacts.local/experiments/secondary-corridor-causal/yolo-imu-causal-feature-probe-v0-20260720/`
- `artifacts.local/experiments/secondary-corridor-causal/advio-yolo-imu-mechanism-probe-v0-20260720/`
- `artifacts.local/experiments/secondary-corridor-causal/yolo-imu-event-head-increment-benchmark-v0-20260720/`
- `artifacts.local/experiments/secondary-corridor-causal/live-camera-yolo-event-head-fixture-device-result-20260720/`

## 已收紧的判断

ADVIO 的三个独立 IMU route/turn probe 未达可分门：原始、旋转不变和 current-only 转向确认的 AUROC 分别为 `.4770`、`.4746`、`.3465`。因此 IMU 只能作为未来待验证的姿态/运动补偿辅助输入，不能充当行进方向、路线选择或转向确认器；本轮没有新增生产传感器接口。

此前在线视觉 sidecar 使总 P95 超过预算，已停止作为该候选的在线主路。现存轻量路径只复用 YOLO 框和外部对齐运动输入，不读取像素、不运行光流，也不调用 `RiskEventTracker`、语音、振动或生产告警路由。

## 当前门禁与下一步

默认 App 仍使用 `yolo11n_fp16_320.tflite`。本快照不改变 [SANPO_CANDIDATE_PROMOTION_GATES.md](SANPO_CANDIDATE_PROMOTION_GATES.md) 的数据、离线、INT8 和同机事件门。

下一道有效门不是继续调参，而是按 [SANPO_COUNTERFACTUAL_EPISODE_COLLECTION.md](SANPO_COUNTERFACTUAL_EPISODE_COLLECTION.md) 准入带隔离 GPT/Codex 共识、许可/隐私收据和哈希的连续 episode。矩阵完整后可自动在六折 leave-one-session-out 上执行预注册 S0：比较静态 pooled、运动几何和空间网格 + causal TCN；只有满足 `balanced accuracy >= .70`、正负召回均 `>= .50` 且 matched-pair 方向一致，才允许训练冻结 backbone 的事件头。

在此之前，不以单次未审计的公开视频、合成数据、模型输出、未训练 TCN 或性能夹具充当风险事件共识，也不把本快照表述为助盲能力验证。
