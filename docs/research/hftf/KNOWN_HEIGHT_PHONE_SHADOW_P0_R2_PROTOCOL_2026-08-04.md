# Known-height phone shadow P0 / R2 frozen protocol

日期：2026-08-04

状态：`FROZEN_BEFORE_PHONE_OUTCOMES / SHADOW_ONLY / NO_APP_AUTHORIZATION`

## 固定对象

模型固定为 `CAMERA_CONDITIONED_SCALE_STUDENT_R0_FINAL_5P`，深度骨干固定为
Depth Anything V2 Metric Hypersim ViT-S 的同一 PyTorch checkpoint。三个评价臂同时产出：
原始 DA metric、已冻结 known-height R0、已封存 student R0。不得在看到手机结果后重训、
改阈值、换 checkpoint、删困难 session 或搜索 operating point。

旧的合成 external confirmation 已被消费；它用于封存参数，不作为这里的 fresh phone
证据。P0/R2 的原始 RGB、量高收据和独立参考值此前均未参与模型选择。

## P0：来源与几何预检

- 3 个固定支架 session，每个目标 120 帧，少于 90 个合格帧则该 session 不入场；
- 相机光心高度必须现场测量，范围 0.8–2.2 m，不确定度不得超过 0.02 m；
- 每个 session 固定 device、camera id、mount profile、intrinsics、裁剪与旋转约定；
- capture timestamp 必须严格递增，RGB 文件及清单逐项 SHA-256 校验；
- 独立卷尺/激光参考只进入离线 evaluator，严禁进入三臂 inference；
- P0 只回答 source/geometry 是否可评价，不产生效果晋级。

## R2：固定手机复核

仅在 3/3 P0 session 入场后采集 8 个新 session，每个 session 按预先固定的等间隔规则
选 25 个 anchor，共 200 个。独立单位是 session，不是 frame。固定门槛为：macro median
absolute-relative error <= 0.20、macro P90 <= 0.50、每个 session false-clear rate <= 0.05、
temporal instability <= 0.15。四项必须同时通过；否则 `STOP`，不得用 frame-level 汇总掩盖
失败 session。

## 当前执行边界

Android 隔离包只采集 RGB、CameraX timestamp、相机/内参/旋转与 mount/量高收据，不发
提示、不写 App 状态。精确 PyTorch depth checkpoint 尚无经金标证明等价的 Android 导出，
所以三臂在 host 离线运行；设备端完整模型 latency 在导出等价性成立前必须记为
`NOT_EVALUABLE_NO_EQUIVALENT_ANDROID_EXPORT`。

如果没有真实量高、独立参考物或固定支架，当前轮次必须停在
`READY_FOR_PHYSICAL_CAPTURE / NOT_EVALUATED`，不能填入估计值替代。
