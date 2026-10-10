# 独立实机接入路线

本路线负责 AtomS3R-M12、XIAO ESP32-S3 + 多区 ToF、安卓手机之间的真实采集与连接。
新增电脑固定台架入口见 [M3 真实 CNH 演示](CNH_M3_LIVE_DEMO_20261010.md)：实时五模型分数与原始回波，当前场景残差/FP32工程输入，非实物准确率结果。
与避障模拟路线并行；不修改其模型、阈值、冻结数据、评测器或运行时默认行为。
当前状态见 [CURRENT.md](CURRENT.md)，数据格式见 [PROTOCOL.md](PROTOCOL.md)。
环境受限时使用[原地四步可视化采集](SIMPLE_CAPTURE.md)：相机 + ToF，150 秒，逐段点击开始。
8×8×16 CNH 的后续现场录制见 [H3 采集清单](CNH_H3_COLLECTION_CHECKLIST.md)，
工具兼容性与旧数据回放见 [H3 录制工具验证](CNH_H3_CAPTURE_VALIDATION_20260928.md)。
既有实机日志的相对路径和校验值见 [证据索引](evidence-index.json)。
本轮环境、编译与离线检查见 [验证记录](VALIDATION.md)。
官方手册、数据手册及板级原理图见 [离线参考资料库](references/README.md)。
Atom 本机图像、短采集结果和恢复说明见 [Atom 实机记录](LOCAL_ATOM_20260921.md)。
固定装配后的两路并行采集见 [并行记录](LOCAL_PAIR_20260921.md) 和
[并行证据索引](pair-evidence-index.json)。
无需尺子的手持靠近/退远检查见 [手持响应记录](LOCAL_HANDHELD_20260921.md)。
模块完成度、唯一下一项和后续顺序见 [硬件审视](REVIEW_NEXT_20260921.md)。
其中诊断固件已完成本机验证，配置、异常结果与恢复记录见 [诊断实测](LOCAL_DIAGNOSTIC_20260921.md)。
本机相机/ToF/CNH 界面、录制和回放的启动方法见 [硬件观测台](DASHBOARD.md)。
最新连续采集、原始字节追查和串口释放结果见 [双路传输验收](TRANSPORT_20260921.md)。
方向检查的屏幕引导、一次采集安排及判读边界见 [方向检查](ORIENTATION_CHECK.md)。
仅补上下轴的“摆好再录”入口见 [手动分段上下检查](MANUAL_VERTICAL_CHECK.md)。
手机通过电脑 USB 中转查看的入口与验证状态见 [手机观测台](PHONE_VIEW.md)。
电脑端 CNH 信息价值的固定波形对照与负结果见 [CNH 指纹检验](CNH_FINGERPRINT_20260921.md)。
新授权的纯背景/纯前景/半遮挡/恢复背景对照见 [CNH 成分分离](CNH_COMPONENTS_20260921.md)。
已消费数据上的局部后峰与纯近物尾部对照见 [CNH 弱后峰检验](CNH_LATE_ECHO_20260921.md)。
无需移动设备的逐帧画面、曲线与已保存状态对照见 [CNH 对照回放](CNH_REPLAY.md)。
相对单值距离到底增加什么信息、稳定多久及其代价见 [CNH 增量价值审计](CNH_INCREMENTAL_20260921.md)。

## 目录与边界

- `firmware/`：I²C 探测、8×8 测距、4×4 CNH 及诊断候选、独立 Atom USB 相机固件和 Wire 适配层。
- `host/`：端口枚举、限时采集、离线回放验证和显示工具。
- `prepare.ps1`、`build.ps1`：准备隔离环境、校验驱动依赖、只编译不刷写。
- `vendor-lock.json`：用户提供的驱动文件指纹；第三方源码和固件数据块不进入 Git。
- `references/`：官方手册、数据手册、原理图的离线资料索引。
- `artifacts.local/hardware-bringup/`：本机配置、虚拟环境、编译产物、采集结果。

电脑暂作开发与采集主机。目标分工：Atom 采 RGB，XIAO 采 ToF，手机接收与计算。
两块主控先独立工作；相机和 ToF 最终需要固定相对位置、完成时间与空间标定。
Wi-Fi 传输、新 ToF 手机适配仍未在本路线验证。Atom 单独使用 USB JPEG 请求协议。

## 环境准备（不需要接硬件）

需要 PowerShell 7、Python 3.10+、Arduino CLI、现有 m5stack ESP32 core 3.3.8，
以及用户提供的 OA-5_6(VL53L7_8CX) 资料包。路径由本机参数传入，不提交到仓库。

```powershell
pwsh -File research/active/hardware-bringup/prepare.ps1 `
  -VendorPackage '<资料包目录>' -ArduinoCli '<arduino-cli.exe>' `
  -ArduinoData '<包含 packages 的 Arduino 数据目录>' `
  -Python '<python.exe>' -InstallDependencies
pwsh -File research/active/hardware-bringup/build.ps1 -Sketch i2c_probe
pwsh -File research/active/hardware-bringup/build.ps1 -Sketch tof_reader
pwsh -File research/active/hardware-bringup/build.ps1 -Sketch tof_cnh
pwsh -File research/active/hardware-bringup/build.ps1 -Sketch tof_cnh_diag
pwsh -File research/active/hardware-bringup/build.ps1 -Sketch atom_camera
```

这里复用已验证的 ESP32-S3 / 8 MB / DIO / hardware-CDC 编译配置。
它的 FQBN 名称来自 M5AtomS3，但代码显式使用 XIAO GPIO5/6，未调用 M5 初始化，
不使用 PSRAM。这不是已安装 Seeed 官方板级配置的声明。
准备脚本不会安装或改变全局 Python 包；编译脚本不会打开串口或刷写设备。
`atom_camera` 单独使用 M5AtomS3R 板型、8 MB Flash、OPI PSRAM，输出 VGA JPEG；
它不使用 XIAO 的板型或接线，不启用 Wi-Fi，也不依赖 Atom 上连接 ToF。

## 采集与离线检查

以下命令从仓库根运行；`<新目录>` 使用 `artifacts.local/hardware-bringup/captures/`
下一个尚不存在的目录。枚举不打开串口，自动选择遇到多个 Espressif 设备会停止。

```powershell
$py = 'artifacts.local/hardware-bringup/venv/Scripts/python.exe'
& $py research/active/hardware-bringup/host/capture.py list-ports
& $py research/active/hardware-bringup/host/capture.py capture --port auto --seconds 20 --output '<新目录>' --label 'wall'
& $py research/active/hardware-bringup/host/capture.py replay --input '<serial.txt 或 raw.bin>' --output '<新目录>'
& $py -m unittest discover -s research/active/hardware-bringup/host -p 'test_*.py'
```

实时矩阵用 `host/monitor.py`，参数与 capture 相同。CNH 绘图用
`host/plot.py --input '<frames.jsonl>' --output '<新文件.png>' --label '<场景>'`。
`--firmware '<固件.bin>'` 可记录预期固件哈希，但不冒充已从设备读回验证。
回放不会扫描或打开串口；无有效帧或有异常时返回非零退出码并保留失败证据。

原始记录、状态、目标数、无效值和设备时间全部保留；不把无返回变成自由空间。
CNH 解码保留原始整数及缩放因子，并注明公式来源，不能凭一张曲线认定多目标分离。
在手机融合前，设备时钟只属于本设备；主机接收时间不是传感器曝光时间。

Atom 已刷入本路线相机固件时，可运行：

```powershell
& $py research/active/hardware-bringup/host/atom_capture.py --port '<Atom串口>' --usb-otg --seconds 20 --output '<新目录>'
```

Atom 端口必须显式指定；两块 ESP32 同时连接时不要向 XIAO 发送相机命令。
每次请求返回一张 JPEG，主机保留字节流、图像、序号、时间和解码检查结果。
这是开发用 USB 串口相机协议，不是 UVC，也不表示手机 Wi-Fi 接入已完成。

两路固件均已确认时，用显式端口并行采集，再离线检查：

```powershell
& $py -B research/active/hardware-bringup/host/pair_capture.py --camera-port '<Atom串口>' --tof-port '<XIAO串口>' --seconds 40 --output '<新采集目录>' --label '<实际场景>'
& $py -B research/active/hardware-bringup/host/pair_inspect.py --input '<采集目录>' --output '<新检查目录>' --plot
```

两个目录均放在 `artifacts.local/hardware-bringup/paired/` 下，拒绝覆盖。
配对只按同一电脑的接收时间取最近图片，不等同于曝光同步；UNKNOWN 保留为空缺。
两块板的 USB VID 相同，端口归属仍需事先确认。主机工具不刷写固件。

已确认 XIAO 运行 `xiao-cnh-diag-v2` 时，单路采集可加 `--query-config`，并行采集
可加 `--tof-query-config`，取得启动时实际配置读回。诊断字段和转换核对见
[协议](PROTOCOL.md)。`tof_cnh_diag` 是独立固件候选，不覆盖原 `tof_cnh` 源码和构建目录。

## 固件与接线

XIAO D4/GPIO5 → SDA，D5/GPIO6 → SCL；GND 共地；INT/SYN 留空，使用轮询。
当前初步读数来自 ToF VCC 接 XIAO 3V3。蓝板内部还有一级稳压器，供电余量待测。
按所给原理图，外部 SDA/SCL 上拉跟随 VCC；不要直接将 VCC 改为 5V 而保留直连信号。
调试时 USB 给主控供电；便携时规划两个 USB 输出分别给 Atom 与 XIAO 供电。

刷写必须选定实物和端口，并先保留原始 Flash。本机已完成一次 CNH 读出与前景对比，
见 [本机记录](LOCAL_CNH_20260921.md)；该早期复测只读串口，未重新刷写。
已编译的测距固件在 artifact 的 `build/tof_reader/`，CNH 候选在 `build/tof_cnh/`；
这两份本地候选未在早期复测时重新刷入；随后独立的 `build/tof_cnh_diag/` 已刷入并验证，
当前 XIAO 使用诊断 v2，见 [诊断记录](LOCAL_DIAGNOSTIC_20260921.md)。
原相机程序备份与早期日志保留在原来的本地工作目录，不提交设备状态到 Git。

## 接线后的最小验收

1. USB 完整断电后，确认唯一端口、I²C 0x29 和设备身份。
2. CNH 初始化、输出块、16×24 原始数组、缩放值和帧序号通过检查。
3. 分别采墙面及墙前物体；记录摆放条件，比较峰形及距离响应。
4. 保存失败记录；握手失败时先定位供电/接线/残留状态，不推断 CNH 不支持。

这轮已完成本机数据读出和场景对比，不训练网络、不接入避障判决、不宣称精度或安全能力。
