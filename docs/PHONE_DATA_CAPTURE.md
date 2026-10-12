# 手机端 Atom RGB + ToF + IMU 数据采集

v10.16.2 的“数据采集”使用 Atom 相机、外接 H3 ToF 与 Atom BMI270 IMU，并同时预览三路读数。RGB 原始 JPEG 保存在手机，供后续离线研究使用。“同时记录 IMU”默认开启，选项会记住；也可在开始前关闭，只采 Atom RGB + ToF。此页不调用手机相机。

## 使用

1. 手机、XIAO ToF 与 Atom 加入同一 Wi-Fi，或打开各设备已配置的手机热点。ToF 需要 H3 Wi-Fi 固件的 TCP 3335 完整 CNH 输出；Atom 提供 `http://地址:81/stream` 相机流，开启 IMU 时还需 TCP 3336 的 BMI270 姿态固件。各板固件与凭据完成一次配置后，日常采集不需要电脑，具体切网方式见下文。App 本身不刷固件、不修改网络凭据。
2. 关闭占用 ToF 的电脑预览/采集；H3 固件仅允许一个 TCP 客户端。
3. 在 App 首页进入“数据采集”，点击“自动寻找设备”。找到唯一 H3 ToF 和 Atom 时填写地址并连接；无法唯一确定时手动填写设备地址，再点击“连接 / 重连”。Atom 地址同时用于相机和 IMU。地址随网络变化，原网络中的默认地址不一定适用；显示实际画面与有效读数后才算就绪。
4. 默认等待 RGB、ToF、IMU 三路均就绪后开始；无需 IMU 时先关闭“同时记录 IMU”，等待 RGB、ToF 两路就绪。可填写场景备注，点击“开始采集”。页面显示秒数和各路已保存的有效帧数；点击“停止并保存”结束本段，再开始下一段。
5. 在已保存列表点击“导出 ZIP”，使用系统文件选择器保存到手机文件夹，再通过 USB、网盘或其他自行选择的方式转移。

顶部同屏显示 Atom 画面、8×8 ToF 距离网格、IMU 俯角／横滚与加速度／陀螺仪读数。无效 ToF 格子不显示距离；输入过期后预览清空。沿用当前支架的 RGB 顺时针 90°、水平镜像和 ToF 180°显示映射，只改变预览，不改变保存的原始数据。IMU 读数是 BMI270 原生轴，手机不施加人工 ToF 安装零点。

v10.16.2 将相机解码、读数刷新、连接和停止分别放到后台工作队列。相机预览只取最新帧，避免排队增加显示延迟；原始记录仍逐帧保存，预览取最新帧不改变已接收数据的保存规则。文件流复用后仍逐条同步落盘；已结束会话优先读取保存计数，损坏或旧记录继续扫描恢复。

v10.16.3 进一步将新图更新限制在画布绘制阶段，按固定周期取最新帧；CNH / IMU 分片批量组包，停止 Atom 时先断开 HTTP 再关闭读取流。ZIP 导出改成 256 KiB 缓冲批量写入，JPEG 不再重复压缩，JSON 使用快速压缩，显示进度与完成耗时。系统文件选择器期间页面重建仍能恢复待导出的会话；原始文件与逐条同步落盘规则保留。ZIP 压缩率可能略降，导出到网盘等目标的速度仍取决于其文档提供器。

采集期间保持页面打开；页面保持屏幕常亮。切到后台、返回首页或任一启用输入超过 2 秒未更新时结束当前段并保留已有数据。返回后可重新连接并开始新段；IMU 开关在一段采集中不可修改。采集原始数据不依赖电脑或模型推理；网络连接使用当前局域网明文传输。

## Wi-Fi 与手机热点

支持双配置的固件可同时记住两个网络：槽位 0 保留原 Wi-Fi，槽位 1 保存手机热点。首次刷写和配置使用电脑；热点名称、密码须与设备中保存的配置一致。可用 [provision_wifi.py](../research/active/hardware-bringup/host/provision_wifi.py) 的 `--slot 1` 添加热点并保留原 Wi-Fi，凭据只通过 USB 配置，不写入采集 ZIP。

设备保持当前可用连接；链路断开后尝试另一个已配置网络，每次尝试最多 12 秒，失败后轮换重试。重启时先尝试上次连接成功的网络。因此，离开原 Wi-Fi 后打开手机热点，等待 ToF 接入；回到原 Wi-Fi 时，若热点仍可用，设备会继续使用热点，可关闭热点使其重新寻找网络。

切网可能改变 ToF、Atom 地址，也会使当前 TCP 采集段结束。各设备接入新网络后，在 App 点击“自动寻找设备”，等待所有启用输入重新就绪，再开始新段；App 不跨断连自动拼接记录。自动寻找使用同一局域网的 UDP 3334 发现接口，识别 H3 CNH ToF 与 Atom 网络身份；IMU 是否可用仍由 TCP 3336 的有效样本决定。未找到时可手动填写设备实际地址。

## 每段文件

采集目录位于 App 私有且不自动备份的 `noBackupFilesDir/phone-capture/<session-id>/`。安装覆盖升级保留已有记录；卸载或清除 App 数据会删除私有记录，请先手动导出。

| 文件 | 内容 |
| --- | --- |
| `manifest.json` | 会话身份、备注、源信息、IMU 是否启用、各路有效帧/原始记录计数、开始/结束和状态 |
| `rgb/frame-000001.jpg` 等 | Atom MJPEG 流中的原始 JPEG，不重编码；实际尺寸/帧率以索引为准 |
| `rgb.jsonl` | 文件名、实际尺寸、显示方向、Atom 帧头/序号/boot、设备 capture/JPEG ready 时间、独立 Android 接收时间及相机来源 |
| `tof.raw.jsonl` | 收到的原始字节：8×8 zone、每 zone 16 bin 的 histogram/scaler、ambient、距离/status/target、seq、ms、read_us、stream、header、boot/config 等 |
| `tof.index.jsonl` | 原始记录字节位置/长度、接收单调时间、序号、设备毫秒、完整 CNH 校验状态及 `imu_alignment` 接收时间匹配 |
| `imu.raw.jsonl` | Atom IMU 原始字节：`imu_boot`（含配置）、`imu_frame`、`imu_status`、坏包与未完成尾包 |
| `imu.index.jsonl` | 原始字节位置/长度、Android 接收纳秒、序号、Atom 设备微秒、设备/boot 身份、六轴原始整数与 accel_g/gyro_dps、pitch/roll、可用 gravity_unit、校正/采样间隔/读时/缺口标记及有效帧状态 |
| `events.jsonl` | 开始、ToF/IMU 连接异常、缺帧/重复/时钟异常等记录 |

原始传输出现坏包或未完成尾包时保留字节，按 index 的 offset/length 恢复；它们不计作有效 ToF/IMU 帧。关闭 IMU 的新会话中对应文件为空；早期无 IMU 文件的会话仍可列出和导出，IMU 计数为 0。程序异常退出后目录仍出现在列表，可导出完整或部分文件。ZIP 包含全部已保存原始文件，不把画面编码成大段 JSON/base64。

## 时间与空间

Atom 相机帧头的 capture/JPEG ready 时间、ToF `ms`、IMU `device_us`、手机 `elapsedRealtimeNanos` 接收时间分别保存。Atom 相机和 IMU 保留各自来源与 boot，ToF 属于另一设备时钟；手机不映射这些设备时钟。

每条有效 ToF 记录可引用同一会话中最近的因果 IMU 接收记录：IMU 的 Android 接收时间不晚于 ToF，差距最多 100 ms。`imu_alignment` 保存原始 IMU 索引引用与有符号 `imu_minus_tof_ns`（IMU 接收时间减 ToF 接收时间）；无符合条件的记录时为 `UNAVAILABLE` 且引用/差值为空，关闭 IMU 时为 `DISABLED`。这个匹配是接收时间匹配，不代表传感器采样或曝光同步。

JPEG 原字节、ToF zone/bin 顺序与数值不改。IMU 六轴、pitch/roll 与 gravity_unit 保留 BMI270 本机轴，标为 `BMI270_NATIVE_NOT_TOF_OPTICAL`；手机不施加安装旋转、零偏或主机侧姿态变换。Atom RGB 与 ToF 的显示朝向沿用已有安装核对，精确光轴/图像外参尚未标定。场景备注是用户描述，不是障碍真值或安全标签。

## 验证

v10.16.3 在同一 SM-S9280 / Android16 验证两段各30秒：保存794/708 RGB、155/155 CNH、3036/3024 IMU，停止保存329/88ms；JPEG SHA、ToF/IMU原字节索引和 ZIP 文件内容独立核对通过。约15.78/15.46MB原始会话的本机缓存文件导出从29.091/36.513秒降到0.165/0.466秒，ZIP体积增加约18–20%。系统文件选择器实际向 Downloads 导出另一份5.05MB会话用了0.3秒，ZIP与缓存导出的SHA完全一致；其他文档提供器、网盘和文件缓存条件不在该速度结论内。

最终16ms预览检查的10秒采集测试通过，保存165RGB/52CNH/1014IMU，停止185ms。Android接收至Canvas提交年龄中位19.874ms/P9539.254ms，旧版本5秒窗口为40.485/71.746ms。最终显示156幅独立画面/10.021秒；源RGB本轮在16–28Hz之间变化，ToF约5.103Hz、IMU约100Hz。Atom帧头显示较大JPEG之后取下一帧变慢；固件同步发送JPEG后才归还framebuffer，发送序号连续不代表传感器帧没有被覆盖，因此不把本轮显示帧率比较作为源采集帧率提高的证明。40项聚焦单测、最终构建/lint/APK校验通过；测试收尾界面查询失败留证，修正后的3项短流程及最终10秒真机测试通过。原始证据在 ignored `artifacts.local/evidence/phone-capture-opt-20261012/`。

框架外未录制窗口，系统慢帧由248/721（34.40%）降至349/1241（28.12%），帧耗时中位7→5ms、P9536→32ms；两窗约24/38秒且源图像负载变化，仅作描述性检查，仍有尾部慢帧。验证结束回到首页，采集连接释放；实际导出的ZIP保留在手机 Downloads。

同手机／LAN／硬件摆放的框架外 12 秒对比：整页渲染速率从 10.145 增至 29.728 帧/秒，系统慢帧比例从 94.31% 降至 32.60%，最近帧耗时中位数从 27.990 降至 6.354 ms。P95 从 38.203 到 39.078 ms，尾部慢帧仍存在。整页渲染速率不是相机源帧率；完整口径与收据见 [IMU/手机报告](../research/active/hardware-bringup/ATOM_IMU_PITCH_20261012.md#手机性能优化-v10162)。v10.16.2 当轮结束时保留三路预览；本轮结束状态见上文。

v10.16.2 真机约 5 秒保存 138 Atom RGB／26 CNH／507 有效 IMU（508 条原始记录），状态 `COMPLETED/user_stop`；CNH 5.10412 Hz、序号连续，IMU 100.02507 Hz、sample_gap 0。26/26 ToF 因果匹配，年龄 P95 13.600 ms。ZIP 145 条目逐项原字节一致，Activity 重建、设备重连与完整三路面板显示通过。测试框架内录制期间画面独立更新 24.246 帧/秒，Android 接收至 Canvas 提交年龄 P95 67.641 ms；这不是曝光至显示延迟。35 项聚焦单测、最终 debug/androidTest/lint、APK 身份/签名/16KB 校验通过。最终安装包 SHA256 `1B7AD5E47EC8527596A2442D1B2A086240CA72FCC59A144DDDC981FEB26334D3`；原始收据在 ignored `artifacts.local/evidence/phone-capture-atom-v1/phone-live-lan-20261012-021241/`。

v10.16.1 的真实 Atom 来源检查已通过：约 5 秒保存 138 Atom RGB／26 ToF／508 IMU，26/26 ToF 因果匹配、年龄 P95 12.339 ms，IMU 99.99765 Hz、零 sample_gap。核对了 Atom 帧头、设备 capture 时间、原 JPEG SHA／Content-Length／尺寸、显示旋转／镜像与相机／IMU boot 一致；三路同屏及 Activity 重建重连通过。ZIP 145 个条目与手机原文件逐字节一致，状态 `COMPLETED`。源码／存储 33 项聚焦单测、debug/androidTest 构建与 lint 通过；证据在 ignored `artifacts.local/evidence/phone-capture-atom-v1/phone-live-lan-20261012-014800/`。

此前 v10.16.0 使用手机后置相机：SM-S9280（Android API 36）约 5 秒保存 50 RGB、25 ToF、507 IMU／508 条 IMU 原始记录，状态 `COMPLETED`。IMU 设备速率 100.0003 Hz，sample_gap 0；25/25 ToF 因果匹配，年龄 P95 10.438 ms。ZIP 57 个文件与保存会话逐字节一致。该历史段保留，不将其当成 Atom 相机采集证据。

Galaxy 热点段保存 51 RGB／27 ToF／509 IMU，26/27 因果匹配、年龄 P95 10.465 ms；首个 ToF 前尚无 IMU，明确不可用。关闭热点后两板在同一 boot 自动回原 LAN（Atom 4.250 秒、ToF 6.469 秒），恢复段保存 50／25／506、25/25 匹配、P95 11.196 ms。三段均正常停止，ZIP 每个条目与手机原文件一致。测试结束后热点关闭，App 首页待用，所有采集连接已释放。

27 项聚焦数据完整性单测、debug/androidTest 构建、lint、APK 身份/签名/16KB 检查及真机相机重建通过。此前两路版本的 50 RGB／27 ToF／30 原始记录、24 帧 Android 原包解码与系统文件选择器实际导出结果继续保留。

首次 H3 TCP 检查的 EOF、刷写后的 `is_alive` 失败及随后冷启动恢复均保留在证据中。完整顺序与回执见 [DEVELOPMENT_LOG.md](../DEVELOPMENT_LOG.md) 本版本条目及 ignored `artifacts.local/evidence/phone-capture-v1/`、`phone-capture-imu-v1/`。Atom 实机 I²C 冲突、修复、静止测量、人工俯角参考和主机对齐见 [IMU 报告](../research/active/hardware-bringup/ATOM_IMU_PITCH_20261012.md)。双网络实际切换结果在该报告另记。
