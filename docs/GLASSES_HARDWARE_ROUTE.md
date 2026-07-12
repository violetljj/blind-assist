# 眼镜硬件接入路线

## 当前边界

Android App 中的“连接眼镜设备”仍是占位入口：不会扫描蓝牙、不会联网、不会申请额外权限，也不代表已接入真实 ESP32 或眼镜硬件。任何产品说明都必须维持这一边界。

扫描/配对设备、网络发现、发送真实控制命令、采集外设数据或刷写固件都属于外部状态变更，必须先取得用户明确授权。

## 可信资料

- 旧眼镜工程：`E:\linnan\glassses`
- 参考资料：`E:\linnan\esp32参考资料`
- 优先阅读：`services/audio_service.py`、`services/camera_service.py`、`services/microphone_service.py`、`stm32code/esp32_firmware_mic.ino`、`stm32code/speaker.cpp`

不要将截断的 `glassses-main.zip`、失败克隆残留目录或临时恢复目录当作可信源码。

## 迁移前的设计门槛

在 Android 侧接入前，先设计并评审以下边界及无外设降级行为：

- `GlassesConnectionRepository`：连接状态、重连、权限和错误边界。
- `GlassesControlChannel`：控制命令的协议抽象、超时和幂等性。
- `GlassesFrameSource`：帧输入、时间戳、背压和失效帧处理。

旧工程中的 TCP PCM / MJPEG 协议仅作实验参考，不能直接视为 Android 产品接入方案或可靠性结论。先完成威胁建模、权限设计、离线降级和可测试接口，再决定是否复用协议。
