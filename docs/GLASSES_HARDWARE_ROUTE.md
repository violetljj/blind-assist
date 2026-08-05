# 眼镜硬件接入路线

## 当前边界

Android App 中的“眼镜外界硬件连接”是统一外设入口。首个真实适配器为
AtomS3R-M12 + Unit ToF4M：手机与设备加入同一局域网后，App 读取设备状态与单区
ToF 距离，并验证 MJPEG 流端点可达。该入口不是蓝牙扫描器，也不表示其他眼镜硬件
已经接入。

扫描/配对设备、网络发现、发送真实控制命令、采集外设数据或刷写固件都属于外部状态变更，必须先取得用户明确授权。

## 可信资料

- 旧眼镜工程：`E:\linnan\glassses`
- 参考资料：`E:\linnan\esp32参考资料`
- 优先阅读：`services/audio_service.py`、`services/camera_service.py`、`services/microphone_service.py`、`stm32code/esp32_firmware_mic.ino`、`stm32code/speaker.cpp`

不要将截断的 `glassses-main.zip`、失败克隆残留目录或临时恢复目录当作可信源码。

## 迁移前的设计门槛

Android 侧按以下边界继续扩展，并保持无外设降级行为：

- `GlassesConnectionRepository`：已实现 AtomS3R HTTP 状态、ToF 与 MJPEG 端点探测；
  后续补设备发现、持续心跳与重连。
- `GlassesControlChannel`：控制命令的协议抽象、超时和幂等性。
- `GlassesFrameSource`：尚未实现；下一里程碑是 MJPEG 解码、设备时间戳、latest-only
  背压和失效帧处理。

旧工程中的 TCP PCM / MJPEG 协议仅作实验参考，不能直接视为 Android 产品接入方案或可靠性结论。先完成威胁建模、权限设计、离线降级和可测试接口，再决定是否复用协议。
