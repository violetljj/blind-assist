# AtomS3R-M12 + Unit ToF4M 实时画面与测距

状态：development

这是 AtomS3R-M12 与 M5Stack Unit ToF4M (`VL53L1X`) 的可逆设备联调入口。
它只产生单区 ToF 开发采集，不是现有 `VL53L8CX` 多区输入，不授权 RGB-ToF
同步、米制整帧融合、Android 集成、主动提醒或产品/安全结论。

## 接线

使用 ToF4M 附带的 HY2.0-4P Grove 线直接连接：

| 线色 | AtomS3R-M12 | Unit ToF4M |
| --- | --- | --- |
| 黑 | GND | GND |
| 红 | 5V | 5V |
| 黄 | GPIO2 | SDA |
| 白 | GPIO1 | SCL |

固件固定外部 I2C 为 `SDA=GPIO2`、`SCL=GPIO1`、`400 kHz`，使用 7 位地址
`0x29`。VL53L1X 数据手册中的 `0x52/0x53` 是包含 R/W 位的 8 位总线表示。

## Arduino 依赖与编译

- Arduino CLI `1.5.1` 或 Arduino IDE 2.x；
- M5Stack board package `m5stack:esp32@3.3.8`；
- Pololu `VL53L1X@1.3.1`；
- 板卡 FQBN：`m5stack:esp32:m5stack_atoms3r`。

```powershell
arduino-cli config set board_manager.additional_urls https://m5stack.oss-cn-shenzhen.aliyuncs.com/resource/arduino/package_m5stack_index.json
arduino-cli core update-index
arduino-cli core install m5stack:esp32@3.3.8
arduino-cli lib install VL53L1X@1.3.1
arduino-cli compile --fqbn m5stack:esp32:m5stack_atoms3r scripts/research/hftf/atoms3r_m12_tof4m
arduino-cli upload --fqbn m5stack:esp32:m5stack_atoms3r -p COMx scripts/research/hftf/atoms3r_m12_tof4m
```

进入下载模式：长按侧面复位键约两秒，看到内部绿色 LED 后松开，再选择新出现的
COM 端口。烧录后串口为 `115200 baud`。

## Wi-Fi 与实时网页

固件会优先连接已保存的 2.4 GHz Wi-Fi。首次使用或连接失败时，它会启动临时热点：

- SSID：`AtomS3R-ToF-Setup`
- 密码：`blindassist`
- 配置页：`http://192.168.4.1/`

保存后设备重启并加入用户网络；凭据只存入设备 NVS，不写入串口或仓库。电脑或手机
切回同一个局域网后，访问 `http://atoms3r-tof.local/`。若当前系统的 mDNS 不可用，
请从路由器设备列表或串口 `wifi_station` 事件取得 DHCP 地址后直接访问，例如
`http://192.168.x.x/`。DHCP 地址可能随重连变化。

控制页在端口 80，MJPEG 流在端口 81。默认相机档为 XGA `1024×768`、JPEG quality
`10`、双 PSRAM framebuffer 和 latest-frame grab；这是清晰度与延迟的平衡档，并非
OV3660 的最高静态分辨率。页面绿色框只标示 ToF4M 的中央单区窄视场，画面其余区域
没有对应距离，网页也不构成避障或安全系统。

### 网页功能

- 相机分辨率可在 `VGA/SVGA/XGA/SXGA/UXGA` 间切换，JPEG quality 限制为
  `6..30`，亮度限制为 `-2..2`；
- 曝光支持自动模式下 `-2..2` 补偿，或关闭自动曝光后使用 `0..1200` 手动值；
- 参数经过设备端白名单与范围校验，应用后丢弃三帧过渡缓冲；设置只在当前开机
  session 生效，重启恢复稳定的 XGA/quality 10 默认值；
- `下载截图＋JSON` 会下载一张实际 JPEG 和浏览器生成的配套 JSON。JSON 记录
  boot sequence、frame sequence、相机首 DMA、JPEG ready、最近时刻 ToF 样本、
  有符号 ToF—frame 差、实际 JPEG 宽高和 quality；这是同一设备时钟中的
  nearest-sample binding，不代表完成了 RGB-ToF 硬件触发同步或外参标定；
- `/status` 每秒显示运行时间、空闲内存、Wi-Fi/IP/RSSI/重连次数、相机配置、近期
  帧率、累计帧、流客户端和 ToF 状态；
- 设备启用 ESP32 自动重连并在断线时每 5 秒主动调用 reconnect；网页对距离 API
  使用超时和退避重试，对 MJPEG 使用指数退避并在状态 API 报告停帧时重新连接；
- `/?static=1` 使用一次有限抓拍代替无限 MJPEG，仅用于局域网页面诊断。

主要 HTTP 接口：

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| GET | `/api/range` | 最近一次 fail-closed 单区距离 |
| GET | `/api/status` | 系统、Wi-Fi、相机、帧率与 ToF 状态 |
| GET | `/api/time` | HTTP 对时/身份诊断；正式基线使用 3333/UDP |
| GET/POST | `/api/camera` | 读取或应用当前 session 相机参数 |
| GET | `/api/snapshot` | JPEG；`X-Capture-Metadata` 响应头携带配套 JSON |
| GET | `/status` | 人可读设备状态页 |

浏览器可能要求用户允许同一站点连续下载两个文件；若只出现一个文件，请在浏览器
下载提示中允许多个下载后重试。

## 输出合同

串口只输出 JSONL。事件行使用 `blindassist_atoms3r_tof4m_event_r0`；测量行使用
`blindassist_atoms3r_tof4m_sample_r0`。测量时间是当前 ESP32 boot monotonic
时钟中的 `sensor_read_complete`，不能直接与手机或 PC monotonic 时间比较。

MJPEG 每个 part 额外携带 `X-Frame-Sequence`、`X-Capture-Timestamp-Us`、
`X-Jpeg-Ready-Timestamp-Us`、`X-Device-Send-Start-Timestamp-Us`、
`X-Tof-Timestamp-Us` 和 `X-Tof-Minus-Capture-Us`。相机 capture 时间的精确定义是
`esp32-camera` 的“首个 DMA buffer 自开机时间”，不是曝光起始。固件在 3333/UDP
提供独立高优先级二进制对时服务；主机端以最小 RTT midpoint 映射两个 monotonic
时钟，并逐帧保留 RTT 与误差上界。

有效测量同时要求：无超时、VL53L1X `RangeValid`、距离在 `40..4000 mm`。
无效测量保留原始 `range_mm`、状态码、signal/ambient rate，但 `range_m=null`，
防止无效原始值进入米制融合。驱动不暴露 per-sample sigma，因此固件不会伪造
`sigma_m`。

## 采集与校验

从仓库根目录执行；输出被限制在 `artifacts.local/`，且不会覆盖已有采集：

```powershell
pwsh -NoProfile -File scripts/research/hftf/atoms3r_m12_tof4m/capture_tof4m_serial.ps1 -Port COMx -DurationSeconds 60
E:\codex-tools\bin\blindassist-python.cmd scripts/research/hftf/atoms3r_m12_tof4m/validate_tof4m_jsonl.py artifacts.local/evidence/tof4m/<capture>.jsonl
```

采集脚本同时生成 SHA-256 receipt，初始角色固定为
`DEVELOPMENT_DEVICE_CAPTURE_UNVALIDATED`。validator 通过只表示 JSON、序号、时钟、
状态和单位合同一致，不证明传感器精度、同步、标定或安全效用。

## 端到端时间基线

完整主机链路从仓库根目录运行：

```powershell
E:\codex-tools\bin\blindassist-python.cmd scripts/research/hftf/atoms3r_m12_tof4m/measure_e2e_latency.py `
  --duration-seconds 300 `
  --pipeline-module scripts/research/hftf/atoms3r_m12_tof4m/host_reference_pipeline.py `
  --pipeline-model app/src/main/assets/yolo11n_fp16_320.tflite
```

每次运行在 `artifacts.local/evidence/atoms3r-e2e/<UTC>/` 生成逐帧、状态、对时
JSONL 和 summary。参考 pipeline 只用于测量 JPEG 解码、YOLO11n host CPU 推理、
简单 ToF 风险计算及反馈记录调度；它不发出物理语音/震动，也不是产品风险算法。
日常性能回归默认 300 秒；30–60 分钟仅在明确要求时作为压力测试。主机使用独立
MJPEG reader 和容量 1 的 latest-frame 队列，推理短时落后时显式覆盖旧帧并计数，
避免把过时画面堆在 TCP 缓冲中。

2026-08-05 的 XGA/quality 10、同一局域网 30 分钟实测共 43,230 帧：0 次流重连、
0 个错误、0 个 frame-sequence 缺口；capture→完整 JPEG P50/P95/P99 为
`99.1/225.5/259.2 ms`，capture→decode 为 `101.9/228.4/262.2 ms`，
capture→反馈记录完成为 `137.2/265.8/300.7 ms`。绝对 ToF—frame skew
P50/P95/max 为 `23.3/51.5/59.7 ms`。空闲堆首尾同为 `153,288 B`；ESP32 内部
温度首尾 `67.1→71.1 °C`、最高 `72.1 °C`；RSSI P50 `-37 dBm`。UDP 对时误差
上界 P50/P95 为 `1.45/2.20 ms`。这些数字是当前主机/网络/配置的 Development
基线，不代表手机端、物理输出、人体使用或安全性能。

latest-frame 优化后的 2026-08-05 五分钟回归共 7,158 个处理帧：0 重连、0 错误，
容量 1 队列覆盖 2 个旧帧（约 `0.028%`）。capture→反馈记录完成 P50/P95/P99
为 `109.3/146.8/180.3 ms`；JPEG ready→host read start P95 为 `7.2 ms`，较旧
30 分钟基线的 `179.4 ms` 显著降低。不同持续时间不能替代压力测试的同长度比较，
但阶段账本确认旧尾延迟的主要机制是串行接收造成的 host backlog。

### 设备慢帧归因

R4 在冻结的 XGA/quality 10/自动曝光配置下，逐帧增加 frame-ready interval、
`esp_camera_fb_get`/mutex acquire、JPEG/metadata prepare、前一帧 HTTP write、JPEG
大小、实际曝光值、RSSI、heap 和 ToF update 计数。由于当前帧的 interval 会受到
前一帧 write 影响，write duration 由下一帧 header 回报并按 frame sequence 回填。
主机另记 first byte、full frame、decode、latest queue wait；被覆盖帧写入独立
`overwritten_frames.jsonl`。

慢帧定义固定为：`frame_ready_interval > median + 3×MAD` 或
`frame_ready_interval > 2×median`。2026-08-05 五分钟 R1 的 median/MAD 为
`36.047/0.091 ms`，1,252/7,070 个可评 interval 为慢帧。慢/普通 acquire P50
为 `48.4/13.7 ms`，前一帧 write P50 为 `22.8/21.4 ms`；慢/普通 JPEG 中位数
`31,360/31,371 B`，实际曝光值始终 `490`。诊断分层中 981/1,252 个慢帧属于
`acquire≥30 ms 且 preceding write<40 ms`，130/1,252 属于
`preceding write≥40 ms`。因此主机制是相机 framebuffer/帧节拍等待，少数由 Wi-Fi
写阻塞直接传导；本轮不支持把 JPEG 大小或曝光变化当作主因。ToF 相关性尚不能排除
等待窗口长度混杂，下一合法对照是仅关闭 ToF 读取，其余配置不变。

## 停止条件与下一步

首轮只要求：I2C `0x29` 可见、连续 JSONL 可解析、时间戳/序号单调、有效/无效状态
可复现。任一项失败时停止在对应电气、驱动或协议层，不修改多区 adapter 来迁就数据。
当前时间账本和 latest-frame 接收策略已建立。下一性能工作可继续分析设备端 JPEG
慢帧及真实手机输出；相机与 ToF 的空间标定仍按用户要求暂缓。单区数据不得填充成
三个或更多伪 zone。
