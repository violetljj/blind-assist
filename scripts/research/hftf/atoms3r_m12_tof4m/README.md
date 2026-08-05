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
  boot sequence、抓拍时间、最近 ToF 样本时间/年龄/状态、实际 JPEG 宽高和 quality；
  这是 nearest-sample binding，不代表完成了 RGB-ToF 硬件同步或外参标定；
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
| GET/POST | `/api/camera` | 读取或应用当前 session 相机参数 |
| GET | `/api/snapshot` | JPEG；`X-Capture-Metadata` 响应头携带配套 JSON |
| GET | `/status` | 人可读设备状态页 |

浏览器可能要求用户允许同一站点连续下载两个文件；若只出现一个文件，请在浏览器
下载提示中允许多个下载后重试。

## 输出合同

串口只输出 JSONL。事件行使用 `blindassist_atoms3r_tof4m_event_r0`；测量行使用
`blindassist_atoms3r_tof4m_sample_r0`。测量时间是当前 ESP32 boot monotonic
时钟中的 `sensor_read_complete`，不能直接与手机或 PC monotonic 时间比较。

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

## 停止条件与下一步

首轮只要求：I2C `0x29` 可见、连续 JSONL 可解析、时间戳/序号单调、有效/无效状态
可复现。任一项失败时停止在对应电气、驱动或协议层，不修改多区 adapter 来迁就数据。
通过后再新增同板 OV3660 帧时间账本，测量 RGB-ToF skew，并为单区中心射线建立独立
注册合同；单区数据不得填充成三个或更多伪 zone。
