# H3 完整 CNH 读出优化（2026-10-14）

EXPLORE 真机工程。用户授权“试试”，目标是缩短完整 ToF 读出，同时观察 RGB/IMU 并行采集。保留 64 zone × 16 bin、全部 ambient/scaler、固件距离/status/target、5 Hz、20 ms integration、autonomous、I²C 1 MHz、TCP3335 NDJSON。手机记录原字节与设备/接收时间，未改变融合或提醒逻辑。主证据目录：`artifacts.local/evidence/tof-read-20261014/`；运行脚本的原输出仍保存在 `receive-tail-20261013/`，按相同 run id 查找。

## 结论与机制

选择 H3 专用的 ESP32 HAL 直接读取：完整 `RdMulti` 一次进入 vendor 缓冲区，仍核实返回状态和完整长度。原 Wire 120 字节读法中位 95.287 ms；512 字节分块为 90.694 ms；直接读取为 88.180 ms，减少 7.107 ms，约 7.5%。这是完整读出调用的节省，不是曝光到手机的延迟改善或更高帧率。

原 6916 字节数据需约 58 次 120 字节读取；512 字节需 14 次。供应商 ESP32 示例允许整笔读取，Arduino Wire 的默认 128 字节缓存是原实现分块的原因。直接 HAL 沿用同一 IDF 底层、总线锁与两字节寄存器地址，避免 Wire 缓冲搬运和重复事务。其他 sketch 默认仍走 Wire 120 字节，写入也保持 120 字节；启动初始化及启动总线恢复后核实专用 bus 0 已初始化，才启用直接读法。

直接读取段中，HAL 调用中位约 87.958 ms、最大 88.030 ms；整个 vendor 调用与 HAL 的均值差约 0.224 ms。`i2c_read_us` 包含 HAL 锁、驱动和总线等待，不是纯线速时间；没有空口或 SCL 波形，不能进一步断言剩余时间来自传感器拉伸、FIFO 中断或实际总线频率。`Wire.read()` 本身没有逐字节加锁，原先怀疑的锁开销没有代码依据。

[ST 数据手册](https://www.st.com/resource/en/datasheet/vl53l8ch.pdf)给出 I²C 最高 1 MHz，本轮未超频。匹配的本地 Arduino 3.3.8 HAL/Wire 和[IDF 5.5.4 I²C 实现](https://raw.githubusercontent.com/espressif/esp-idf/v5.5.4/components/esp_driver_i2c/i2c_master.c)均已核对；直接接口接受完整读取长度，由底层处理 FIFO。

## 真机短段

每段先 5 秒预览，再约 30 秒三路录制；前面三段使用不替换 Compose 帧时钟的 ActivityScenario 测试。固件重启、网络状态与运行顺序保留，属于工程描述，不是交叉重复因果试验。

| run | 有效 CNH 帧 | 完整读出中位 / P95 ms | ToF 接收间隔 P95 / 最大 ms | IMU 接收间隔 P95 / >100ms |
|---|---:|---:|---:|---:|
| `tof_read_base` 原 Wire120 | 153 | 95.287 / 95.455 | 264.947 / 390.271 | 22.398 / 5/3001 |
| `tof_read_512` Wire512 | 152 | 90.694 / 90.835 | 282.369 / 432.959 | 21.484 / 0/3001 |
| `tof_read_native` HAL直接读、固定100ms超时 | 154 | 88.180 / 88.233 | 255.451 / 353.069 | 19.750 / 0/3001 |
| `tof_read_final_compat` 最终固件＋原手机APK | 154 | 88.189 / 88.252 | 282.825 / 355.085 | 25.794 / 4/3020 |

ToF 接收统计裁剪到 RGB 接收窗口，三个分母分别是 151、151、152 个相邻间隔，窗口内有效帧为 152、152、153；表内左列是整段有效 CNH 数。IMU 长间隔分母也按同一窗口裁剪，见每段 `comparison-*.json`；不会把无序号缺口解释为所有物理采样均无遗漏。三个 ToF 原始段均只有连接/boot/config/cnh_frame 记录，没有读出错误；全部完整 8×8×16，归一化数值有限，原字节与索引连续覆盖、记录序号检查通过。前三段共 2460 张 VGA JPEG 的单图边界、完整解码和逐张哈希通过。IMU 原始字节覆盖与 sample_gap 检查也通过，客户端均释放。

ToF 仍约 5.06–5.09 Hz。512 方案没有手机接收收益，直接读法也没有足够匹配证据证明无线长尾改善。基线 RGB 最大接收间隔 1378.249 ms，设备前一 TCP 写入最大 1373.006 ms；候选段没有同样网络停顿，所以不把候选 RGB/IMU 尾部的变化归因于 ToF 优化。

最终兼容性段用恢复后的原 APK（App61/10.19.3）和既有 Compose 测试规则，仅验收三路接入、读出及保存，不用于生产预览性能对比。154 个完整 CNH 帧无读取错误，读出最大 88.299 ms；HAL 最大 88.053 ms，非 HAL 部分最大 0.290 ms。836 张 VGA JPEG 的边界、解码和逐张哈希通过，三路原始字节覆盖及记录序号检查通过，IMU 无 sample_gap、约 99.89 Hz，客户端均释放；四段共 613 个完整 CNH、3296 张 JPEG。本段 ToF 接收统计是 153 个相邻间隔。IMU 接收仍有 4/3020 个超过100ms，ToF 设备输出间隔最大341ms、手机接收最大355ms，所以无线/同步发送长尾仍未解决。一帧 ToF 的手机接收 IMU 匹配为 UNAVAILABLE，保持不可用标记，不伪装成同步成功。

## 最终超时与部署

整笔读取的超时不能直接照搬原来每个小块的 100 ms。最终源码查询 HAL 的控制器配置频率，按 `max(100 ms, 2 × ceil((n+4) × 9 × 1000 / Hz) + 25 ms)` 设置整笔期限，使用 uint64 计算并检查 IDF 有符号毫秒范围。当前 6916 字节、1 MHz 为 151 ms；400 kHz 为 337 ms，这是预算示例，没有在本轮实测 400 kHz。config 保存 `read_backend` 与 `read_timeout_policy`，每帧增加 `i2c_read_us`，原 `read_us` 定义保持。

刷前 H3 USB MAC `98:a3:16:f7:88:1c` 核对。第一次按旧参考长度 1060448 字节读回不完整，未写入候选；扩大读回后提取 1060528 字节，SHA256 `7b9b295054eab6a525c4d82f248fc0a2cb9bba71957de7bf6236e54104799acb` 与最近重连修复部署二进制一致。完整回退文件 `h3-before-verified.bin`；第一次失败及扩大区域读回均保留。

512 候选 SHA256 `f15441164cf2b9f757abc002026d129e3dfc021e02c58c8e4c11f8c6485405c5`，固定100ms直接读取候选 SHA256 `4ae70eaa820455fc38ce96533a45fcd8f194cb36da3fb8c6f409f76b3610081d`；完整源码分别保存在 `source-512/`、`source-native100/`。最终完整构建 `h3-final.bin` SHA256 `29f83079ddb340db76a09f0e85c46419471bcf26bb83677f34c1ac3e000c22a6`，构建/source/vendor hash 见 `build-final-manifest.json`。仅写 app0 `0x10000`，esptool hash 核对，未写 NVS 或分区表。最终启动实际触发既有总线恢复：首次 is_alive 返回错误，恢复后第二次成功；config、直接后端与完整读长均核对。最终部署与兼容性回归回执按 `h3-final-*`、`tof_read_final_compat*` 保存。

手机使用保留数据的覆盖安装，最终 APK 实际 hash `5d6ad18695ec0c165888e68c9fbf80ce06f2ce2cc7d33480bc85a01dbe80bc58` 与测试前备份一致；没有清除采集数据。最后“返回 MainActivity 并读状态”的组合 shell 命令被自动审批以 blocked by policy 拒绝，未执行且未提供具体原因；完成的兼容性回归、Activity teardown 与客户端释放不受影响。全部任务编译/串口/采集进程已结束，二进制、回退镜像、原始数据及失败日志留在 ignored evidence 目录。

IMU 设备仍约 100 Hz，本轮不改其采样、滤波或队列。独立复核前几段发现 RGB 与 IMU 接收长尾常重叠，但设备 IMU dt 仍约 10 ms；这支持继续检查共享发送后/接收等待，不能证明具体无线重传机制。若再提升 ToF 更新频率，需要另测读出与发送能否并行及传感器配置约束；本轮不把减少 7 ms 当作已突破 5 Hz。
