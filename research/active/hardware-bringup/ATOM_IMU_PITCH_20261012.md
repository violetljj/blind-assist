# Atom BMI270 姿态与 CNH 对齐

2026-10-12 · EXPLORE · 工程接入与实机采集完成，安装零点为人工 pilot 参考。

用户授权 Atom 固件刷写、IMU 实测、电脑采集台/实时引擎与手机采集接入。
本次不训练或修改冻结模型，不访问保护 480/test；IMU 接入阶段只刷 Atom。
先完成全 flash 备份，再写入集成 IMU 与双网络的 Atom 固件。
交付包括源码、原始记录/校验收据、安装标定、1 分钟静止和手转检查及本报告。
无训练/GPU预算；采用改变工程结论所需的聚焦构建、单测和实际传感器检查。

## 已完成的刷写前备份

- ROM 确認 Atom MAC `b4:3a:45:bd:12:d8`，原运行身份 `atom-camera-wifi-v1`。
- 全 flash：`8,388,608` 字节，SHA256
  `a46ed2a0b5ed73fec719523d9c493dadcdc20ac23e4f0020e0bf7468658cd68e`。
- 原 app：`1,092,176` 字节，SHA256
  `17bbb17a3b819cbf418db7014336664fbc01c0082db9d085ed5b28f4f4a26201`；
  镜像内部 checksum 和附加 SHA 校验通过，并与此前构建对应。
- 全 flash、原 app、NVS、分区表及回滚命令保留在
  `artifacts.local/evidence/phone-capture-v1/firmware-backups/dual-wifi-20261012/`。
  `atom-full-backup-receipt.json` 记录实际读回、散列及回滚步骤；备份包含私有网络配置，保存在 ignored 数据目录。
- 备份前没有 Atom 写入。原相机正常运行到下载模式准备；软件复位未进入 ROM 的失败留存，用户按键后 ROM 确认通过。

## 输出与时间

已实现 TCP `3336` NDJSON 与 `/api/imu` 状态。BMI270 原始六轴、物理单位、
native pitch/roll、融合重力方向、seq、Atom boot_id 和单调微秒一起保留。
板载轴姿态需经安装旋转才成为 ToF 光轴俯角，不把显示画面的旋转当作外参。

电脑按主机接收时间做有界最近邻对齐，手机记录同一 Android 单调时钟下的最近已接收 IMU。
各设备时钟尚未映射；对齐残差代表接收时间差，不是曝光同步误差。

固件配置 100 Hz、±4 g、±500°/s，采用重力向量互补滤波（时间常数 0.5 s）。
Bosch 初始化数据 8192 字节与官方固定提交逐字节一致；14 项滤波检查通过。
官方板轴采用 `(nativeY, -nativeX, nativeZ)`，相机法线沿 native +Z；
本固件保存原生轴，不把相机轴或屏幕显示方向推定成 ToF 安装角。
主机实现与安装步骤见 [ATOM_IMU_HOST.md](ATOM_IMU_HOST.md)。

逐帧格式为 `type=imu_frame, schema_version=1, seq, device_id, boot_id, device_us,
accel_raw[3], gyro_raw[3], accel_g[3], gyro_dps[3], gravity_unit[3], pitch_deg,
roll_deg, valid, accel_correction, sample_gap, dt_us, read_us`。TCP 3336 最多两个
消费者；HTTP 状态不替代原始流记录。

## 实机失败与修复顺序

首次集成镜像 1,128,768 字节，SHA256
`a10153c753ec940e598ded34d63292d0f6249641fed5d48304b1bb25f8833ff5`。
只写 Atom app0 `0x10000`，esptool hash 核验通过；没有擦除 NVS 或分区表。
软件 USB reset 与 ROM run 未退出下载模式，用户普通短按 RESET 后应用启动，
boot `61ac66fa755b3851`、COM11、LAN `192.168.5.9`。

第一次实机 IMU 初始化成功，但只到 seq 51，之后读错误持续累计；
`/api/imu` 的旧样本 age 29.5 秒、read_errors 1476，5 秒流检查有效样本为 0。
并行相机入口返回 `503 camera_not_ready`。这轮不能算 100 Hz 或相机兼容通过。
失败保存在 `atom-physical-reset-status.json`、
`lan-smoke-20261012-010402/`、`mjpeg-smoke-20261012-010510/`；
随后核对本机实际 SDK：启用 SCCB 新 I²C 驱动、默认 PORT1；带显式 SCCB
引脚时 `sccb_i2c_port` 不决定总线。本机链接库 `sccb-ng.c.obj` 的 `SCCB_Init`
反汇编亦确认使用 I²C1。原 BMI Wire1 与之冲突是已核对事实；相机失败清理
导致后续 BMI 失活是对时间顺序的解释。诊断收据和反汇编已留存。

修复版先初始化相机，再让 BMI270 使用空闲 I²C0（GPIO45 SDA/GPIO0 SCL），
编译时检查两者总线不同；相机 SCCB 引脚和 MJPEG 处理未变。
同时使用 SDK 读取 STA MAC，避免 Wi-Fi 初始化前得到全零 device_id。
修复镜像 1,129,040 字节，SHA256
`3587d565af48cf9cb949d3cc271b8c3fef331efddf02936f83213a4d4074e4f3`，
只写同一 Atom app0、hash 核验通过；原完整备份未覆盖，失败镜像单独保留。

用户先头戴正对屏幕，随后确认整套支架已放稳桌上，朝向任意。
因此后续桌面段可测静止波动，不能把这个姿态当成 ToF 光轴水平零点。

修复后 boot `f3ea73ec6b8d7fba`，device_id 正确为 `b4:3a:45:bd:12:d8`。
并行短测收到 478 个有效 IMU 样本／4.959999 设备秒，实测 96.169 Hz，
sample_gap 0；同期 MJPEG 52 张完整 JPEG、失败 0，图像 boot 身份一致。
前后 API seq 1995→5178、read_errors 0、末样本仅 4502 µs 旧。
这轮证明并行工作，仍不承诺固定相机帧率或 IMU 每个设备样本都送达。
证据在 `artifacts.local/hardware-bringup/atom-imu-v1/i2c-mac-fixed-*`。

## 1 分钟静止与 CNH 并行

用户确认整套支架放稳桌上，任意朝向，60.016 主机秒内收到 6002 个有效 IMU
和 306 个 CNH 帧。IMU 设备时间跨度 60.014893 秒，实测速率 99.99185 Hz，
sample_gap 0，全部启用加速度校正；融合重力相对均值的方向 RMS 为 0.06414°。

| BMI270 原生姿态 | 均值（°） | 样本标准差（°，6002 帧） | 首末各 5 秒均值差（°） |
| --- | --- | --- | --- |
| pitch | 58.53199 | 0.059765 | −0.188655 |
| roll | −3.96321 | 0.044614 | −0.073359 |

首末窗口各 501 帧；上述漂移为末窗减首窗，标准差采用 ddof=1。
均值属于任意桌面朝向的 BMI 原生轴，不是 ToF 光轴俯角。
这次静止读数包含传感器与实际放置的微小变化，不作为长期温漂或绝对角度精度。

CNH 306/306 帧为完整 8×8×16，设备速率 5.102980 Hz，接收速率 5.103321 Hz；
序号缺帧、重复、重排、设备错误均 0。与此前约 5.1 Hz 相符。
CNH 原始流 4,590,632 字节，SHA256
`5e312ffa178d8fb7aa92cb41f2b76282a923793db1e9f79239fcbe57c43dd09b`。
收据在 `artifacts.local/hardware-bringup/atom-imu-v1/combined-static-20261012-0548/`。

这一旧主机时钟段 306/306 对齐，有符号均值 −0.11765 ms、绝对 P95 15 ms、
最大 16 ms。实际 Python/Windows `monotonic_ns` 使用 15.625 ms 分辨率的
GetTickCount64，因此这些值包含量化，不能当作高精度时间证据。
后续统一 IMU 与 CNH 接收/age 时钟为 `perf_counter_ns`（QPC），保留原始段，
另做短测报告高精度接收残差；不重新解释这段静止统计。

QPC 后续 20.009 主机秒短测收到 102 CNH、1999 IMU；CNH 5.10230 Hz，
102/102 对齐，有符号残差均值 +0.07347 ms、绝对 P95 8.7625 ms、最大 11.5717 ms。
QPC API 报告本机分辨率 100 ns，这不是网络同步准确度。
这段包含用户摆放水平参考的实际移动，只核验链路与对齐，不作为静止噪声或漂移。
收据在 `combined-qpc-20261012-20s/`，原粗时钟段完整保留。

## 人工安装参考与手转

用户表示无法严格摆角，按简化操作确认 ToF 正面竖直朝前、整套下俯、上仰、
再回到正面对墙／竖直屏幕。每段 8 秒，原始流与 six-axis 均保留。
下俯／上仰按“约 30°”要求，用户明确角度只是估计，没有独立量角真值。
由最初水平重力和下俯方向得到 proper 3×3 旋转，不按回正读数重调矩阵：

```text
R_tof_from_imu = [ [ 0.0477284, -0.9833030,  0.1756054],
                  [ 0.9987405,  0.0497033,  0.0068625],
                  [-0.0154761,  0.1750567,  0.9844367] ]
```

将融合向上重力变换后，pitch=`asin(gToF.z)`，下俯为负；
roll=`atan2(-gToF.x,-gToF.y)`。该操作确定人工俯转平面与零参考，
没有独立测量两板光轴安装角，也没有图像配准或 yaw/平移标定。

| 人工阶段 | 有效 IMU 帧 | 参考坐标 pitch（°） | roll（°） |
| --- | --- | --- | --- |
| 首次近似水平 | 798 | 0（按定义） | 0（按定义） |
| 下俯 | 799 | −16.580803 | 0（拟合俯转平面） |
| 上仰 | 796 | +11.206296 | +16.057073 |
| 回正面对墙／屏幕 | 798 | +3.856718 | +5.032365 |

下俯 pitch SD 0.016487°，回正 pitch SD 0.220113°。
方向与操作相符，但上仰伴横滚、回正相对初始零点差 3.856718°；
不能宣称独立验证了精确 ±30°，也不能将 3.856718° 当绝对误差上界。
这对于先前 ±5° 敏感性是实质限制，需用该人工参考时保留质量标签。

安装记录 `imu-mounting-operator-pilot.json` 的 verification 为
`OPERATOR_PILOT_ZERO`，实际操作核对完成；
`absolute_optical_pitch_accuracy_verified=false`，
`tof_atom_extrinsic_independently_measured=false`。
模型姿态来源明确为 `MEASURED_IMU_OPERATOR_PILOT_ZERO`，页面同时显示人工零点与回正差；
未配置、过期或未核验记录时仍明确使用名义俯角。
记录 SHA256 `2e2f32c624ab82f6408434ae062584a27eb9fcde0996df4e634a159a72d63d05`。
原候选和四段 raw 留在 `artifacts.local/hardware-bringup/atom-imu-v1/`；
同字节安装副本 `installation-imu-operator-pilot-20261012.json` 放在既有
`artifacts.local/work/real-labeled-capture-20261011/installation-view.json` 旁边，
记录其路径和 SHA。旧 zone/显示安装记录未覆盖；安装关系未变，沿用已有区序核对。

## 手机三路保存与双 Wi-Fi

用户随后明确 RGB 来源应为头戴 Atom，并要求相机、ToF 与 IMU 同屏预览。v10.16.1（versionCode 47）已在同一手机验证 Atom 原始 MJPEG 来源：5 秒段保存 138 RGB／26 CNH／508 有效 IMU（509 条原始 IMU 记录），状态 `COMPLETED/user_stop`。相机与 IMU boot 均为 `f3ea73ec6b8d7fba`；JPEG 帧头、序号、设备 capture 时间、Content-Length、尺寸及原字节 SHA 核对通过，没有调用手机相机。26/26 CNH 因果接收对齐，年龄 P95 12.338646 ms、最大 13.014219 ms；IMU 99.997653 Hz、sample_gap 0。

该段 ZIP 145 条目、1,798,861 字节，每条目与手机保留会话一致，设备与主机 SHA256 均为 `10cad4063c990b659f71c285786c8470d3e57fc727fa2a5557de0efde0d652b2`。三路同屏、画面方向及 Activity 重建重连通过，证据在 `artifacts.local/evidence/phone-capture-atom-v1/phone-live-lan-20261012-014800/`。下列 v10.16.0 手机相机段作为历史证据保留。

此前已安装并验证 v10.16.0（versionCode 46），该版 RGB 来源为手机后置相机，原会话保留。默认开启“同时记录 IMU”，
首页“数据采集 · RGB / ToF / IMU”→自动寻找设备→三路就绪→开始→停止保存→导出 ZIP。
新增 `imu.raw.jsonl/imu.index.jsonl` 保存原字节、六轴、native 姿态、重力、
seq/device_us/boot 与 Android 单调接收纳秒。ToF index 引用同会话最近因果 IMU，
最大 100 ms，保存有符号 `imu_minus_tof_ns`；没有先到样本就明确不可用。
手机保存 BMI 本机轴供离线安装变换，不直接把 native pitch 当 ToF 光轴角。

| v10.16.0 约 5 秒真机采集（手机后置 RGB） | RGB / CNH / IMU 帧 | IMU 设备 Hz | ToF 因果匹配 | 接收年龄 P95 / 最大（ms） | ZIP 条目 |
| --- | --- | --- | --- | --- | --- |
| 初始 LAN | 50 / 25 / 507 | 100.0003 | 25/25 | 10.438 / 12.545 | 57 |
| Galaxy 热点 | 51 / 27 / 509 | 99.9018 | 26/27 | 10.465 / 12.831 | 58 |
| 自动恢复 LAN | 50 / 25 / 506 | 100.0005 | 25/25 | 11.196 / 12.969 | 57 |

三段均 `COMPLETED/user_stop`，零 sample_gap；ZIP 每个文件与手机保留会话逐字节一致，
设备/主机 ZIP SHA 一致。热点段首个 CNH 的首个已保存 IMU 在其后 4.533 ms，
所以因果规则下该帧不可用，未伪造对齐。证据见 ignored
`artifacts.local/evidence/phone-capture-imu-v1/phone-live*`。
初始 LAN ZIP SHA256 `f1629690c6d6f86ee6bb10bfd1f0ea6a76fecef8cd2883b33f32e0f38df68cfb`。

两板槽0原 LAN、槽1 Galaxy 均已保存。USB `WIFI_RETRY` 走备用配置路径：
Atom `.9`→`10.238.31.218`、ToF `.10`→`10.238.31.236`；
关闭 Galaxy 引发真实 `link_lost`，Atom 4.250 秒、ToF 6.469 秒自动回原 LAN，
两板 boot 保持，profile/gen 均 0/1→1/2→0/3。切换阶段没有刷写或物理 reset；
原 LAN 失联以 runtime disconnect 触发，没有改路由器配置。
旧 H3 协议 device_id 全零如实留存，以实际 USB 序列身份、boot、角色和接收地址核对；
IMU 阶段未为此改刷 ToF。板端收据为
`artifacts.local/evidence/phone-capture-v1/dual-wifi-switch-20261012-014553/board-transition-summary.json`。
v10.16.0 阶段结束时手机恢复测试前的热点关闭状态，留在 App 原首页，相机／采集 TCP／本轮串口均释放。

## 手机性能优化 v10.16.2

已安装 versionCode 48，保留原采集记录；Atom 相机独立每 33 ms 取最新帧并后台解码，Canvas 绘制时执行既有旋转/镜像，避免第二份变换 Bitmap 和整页逐图重组。HUD 每 200 ms 在后台读取；连接、停止和重连串行后台执行，并以生命周期 generation 拒绝迟到操作。追加文件流与 SHA256 复用，仍逐条 `fd.sync` 确认原始记录；已结束会话优先读持久计数，旧或损坏计数扫描恢复。修复了等待写入锁后的旧时钟误判过期，以及收尾期间新源注册与断连交错。

同一手机、当前 LAN 与硬件摆放，框架外约 12 秒预览比较：

| 系统 HWUI 指标 | v10.16.1 | v10.16.2 |
| --- | ---: | ---: |
| 实测窗口 / 总渲染帧 | 12.123846 s / 123 | 12.176989 s / 362 |
| 整窗界面渲染速率 | 10.1453 帧/秒 | 29.7282 帧/秒（2.93×） |
| 系统标记慢帧 / 总帧 | 116/123（94.31%） | 118/362（32.60%） |
| 最近 120 行帧耗时中位数 | 27.9901 ms | 6.3544 ms |
| 最近 120 行帧耗时 P95 | 38.2028 ms | 39.0784 ms（未改善） |
| 最近渲染间隔 P95 | 191.6667 ms | 74.8907 ms |

整体更新频率与典型帧耗时改善，尾部慢帧仍存在，不能称为消除卡顿。HWUI 总计数覆盖整窗，详细 framestats 是有界最近行样本；旧收据中 120 行/12 秒的 9.89785 仅是样本/窗口，整窗速率应按总计数 123 计算。以上是整个页面的渲染速率，不能当 Atom 源帧率或曝光延迟。原件与对比收据在 `artifacts.local/evidence/phone-capture-atom-v1/perf-baseline-v47-20261012-0153/`、`perf-v48-20261012-021442/`。

最终真机三路段 `capture-20261011T181307397Z-5b113eae` 保存 138 Atom JPEG／26 CNH／507 有效 IMU（508 条原始记录），`COMPLETED/user_stop`。CNH 序号连续、设备时间 5.104124 Hz；IMU 100.025066 Hz、sample_gap 0；26/26 因果接收匹配，年龄 P95 13.600313 ms、最大 15.599532 ms。ZIP 1,988,260 字节／145 条目，全部与手机原文件一致，设备与主机 SHA256 均为 `b669af7ecd8ce1978029f75fbce091c37afd7f61b1ec85df0fe109fa57c45483`。相机帧头/原字节与源身份、Activity 重建重连及三路面板完整边界通过。

内部实际图像绘制计数另记：测试框架内 12 秒预览 250 次独立图像绘制、20.82625 帧/秒，录制 5.03171 秒 122 次、24.24623 帧/秒；Android 接收至 Canvas 提交年龄 P95 分别 75.335053／67.640625 ms。这是接收后软件提交年龄，非曝光至物理屏幕延迟。首次发现至三路就绪 3.870 秒，Activity 重建后 3.671 秒。收据、原段与 ZIP 在 `artifacts.local/evidence/phone-capture-atom-v1/phone-live-lan-20261012-021241/`。

首次新性能断言在 sleep-only 测试窗得到零 draw，失败原件 `phone-live-lan-20261012-020734/` 保留；改为推进 Compose 测试时钟且明确显示预览，产品解码/绘制实现未因该失败改动。框架外截图随后发现 BMI 轴说明仍被默认行高裁切，修复显式行高，测试改查未裁剪的实际文本边界；最终同方法 `OK (1 test)`。首版 48 APK `92DFD3B1...` 和截图证据均保留。最终 35 项聚焦单测、debug/androidTest/lint 与 APK 身份、签名、16KB 校验通过；最终包 260,300,372 字节，SHA256 `1B7AD5E47EC8527596A2442D1B2A086240CA72FCC59A144DDDC981FEB26334D3`，回执在 `perf-v48-final-build/`。

当前手机留在三路实时预览、各路就绪且未录制，点击“开始采集”即可接手；此预览保留手机到两板的用户连接，返回首页释放。测试会话和本轮电脑硬件连接已关闭，热点保持关闭。结构检查中仅既有 hardware CURRENT 超过紧凑大小限制（HEAD 已 15,072 字节），未为本工程任务改写其算法历史；相关新文档链接检查通过。

## 检查与使用边界

电脑实时引擎已用本轮 QPC 实际 CNH seq 12014–12025 做 12 帧 CPU forward；
以另 20 个实际帧作为场景参照，全部逐帧采用 `MEASURED_IMU_OPERATOR_PILOT_ZERO`。
历史长度 1→8 后保持 8：7 帧暖机、5 帧 READY，额外 reset 0、非有限输出 0，
旧曝光投影对象保留。五份冻结 checkpoint SHA 与阈值未改，明确使用 float32
工程表示；CUDA 未初始化。12 帧计算合计 366.598 ms，命令 wall 10.967 s。
这是接入检查，不是模型性能或光轴精度评测；收据／输出在
`cpu-pose-integration-20261012/receipt.json` 和 `predictions.jsonl`。

固件数学 14 项、主机接收／对齐／安装 11 项、既有采集续录回放 1 项、
8 个 Python AST、2 个 HTML 的 4 个内联脚本及人工零参考显示检查通过。
最终主机收据为 `artifacts.local/hardware-bringup/atom-imu-v1/host-checks-final-20261012.json`。
Android 27 项聚焦单测、debug/androidTest 构建、lint、实际相机重建、三次三路采集与 ZIP 核验通过。
APK 包名／版本／签名／16KB 对齐通过；v10.16.0 APK SHA256
`C14E12186E2F42D991F462A3F74C5D03E243E8BA089D393A7708BF89C9A90714`。
采集页保持前台；后台或启用输入超过 2 秒无更新会结束并保留段，切网后重新发现、另开新段。
接收时间对齐不是曝光同步。人工零点误差、未知两板精确安装角和 RGB/ToF 图像外参仍是限制；
本轮没有冻结模型／阈值修改、保护 480/test 访问或障碍准确率／安全结论。
