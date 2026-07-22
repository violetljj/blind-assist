# USTRF-SC 手机到眼镜的设备阶段策略

决策日期：2026-07-20。
当前选择：不要求为手持手机阶段制作物理标定物；手机路线保持 reference-free shadow。未来眼镜设备定型后建立独立的 frame、时间、标定和安全证据链。

## 阶段 A - 当前手持手机

- 设备：SM-S9280 后置摄像头，`arcore-camera-v1`。
- 可做：时间戳、rotation、tracking、raw-depth freshness、内参 API observation、非米制语义/事件 typed shadow、日志与回放合同。
- 不做：metric geometry、地面/高度投影、跨帧空间记忆、camera-to-body 生产外参、生产导航动作。
- 已有可打印棋盘格靶仅作为以后可选实验物料；不打印、不使用不会阻塞阶段 A。

## 阶段 B - 眼镜设备定型后

眼镜接入仍遵循既有 `GlassesFrameSource`、`GlassesConnectionRepository` 和 `GlassesControlChannel` seam。具体硬件、相机模组、安装位置、传输协议和供电方案定型前，不创建虚假的 `glasses-*` calibration receipt。

当设备到位时，必须新建：

1. 新 `cameraFrame` 和 `bodyFrame`，例如 `glasses-camera-<revision>` / `glasses-body-<revision>`。
2. 独立 capture clock、帧背压、连接失效与离线降级 receipt。
3. 独立内参、depth-to-camera registration、完整 SE(3) mount calibration manifest 与复核记录。
4. 该设备上的时延、热、动态事件和连续自动多模型事件 shadow。

手机产生的 ARCore observation、手持刚体、内参数值、时间戳统计和任何未来 calibration manifest 都不得迁移或复制到眼镜设备。

## 迁移门

| 结论 | 允许依据 | 不允许依据 |
| --- | --- | --- |
| 手机上的 reference-free shadow | SM-S9280 本机候选观测和 JVM 合同 | ARCore tracking 作为米制几何证明 |
| 眼镜帧源实验 | 新设备的明确授权、FrameSource/连接 seam 和隔离 benchmark | 手机相机/外参/时间数据 |
| 眼镜几何 shadow | 眼镜自身的独立 calibration manifest + device gate | 手机 manifest、不可验证的口头确认或单帧深度 |
| 生产扩展 | 独立事件、性能、隐私和默认链路 gate 全部完成 | 任一单项 benchmark 通过 |

本策略降低当前试验门槛，但不降低未来设备的安全证据标准。
