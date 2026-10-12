# Atom IMU 接收、对齐和安装标定

工程入口；完整实测结果与刷写收据见本轮 `ATOM_IMU_PITCH_20261012.md`。不修改冻结模型、切点或受保护评价数据。

Atom TCP `3336` 输出 NDJSON，独立于 MJPEG。`imu_boot` 保留实际 `device_id`、`boot_id`、`firmware` 和 BMI270/滤波配置；`imu_frame` 含 `seq`、`device_us`（Atom 单调微秒，读取完成）、`accel_raw/gyro_raw`（各三项 int16）、`accel_g/gyro_dps`、`gravity_unit`、原生 `pitch_deg/roll_deg`、`valid`、`sample_gap` 等。原生姿态属于 BMI270 轴，不能直接当 ToF 光轴俯角。

`host/atom_imu.py` 保存 TCP 原字节到 `raw.bin`，每行以 `raw_offset/raw_length` 索引到 `receipts.jsonl`，带最后换行字节所在接收块的主机单调纳秒与 UTC。断开前残尾保留，重连及 boot/时钟回退清空可对齐缓存。

主机 IMU、CNH、采集起点及读数 age 统一使用 `time.perf_counter_ns()`，域为 `HOST_PERF_COUNTER_NS`；Windows 上为高精度 QPC，并记录实现和分辨率。既有 Windows Python 的 `monotonic_ns()` 基于 GetTickCount64、分辨率 15.625 ms，首轮 60 秒静止收据保留为粗钟证据；其设备帧率与姿态 SD/漂移不受该主机量化影响，但对齐残差不可当高精度同步。旧收据缺少/不同 clock domain 时不计算新时钟 age。

每个 CNH 帧取 IMU 的最近主机接收时间，最多等待 30 ms 以取得后一个样本；只有绝对残差 ≤100 ms 可用。`residual_ms = IMU receipt − CNH receipt`，负数表示 IMU 先到。无 IMU、无效或过期都明确标记不可用。两块板时钟仍未映射，接收接近不等于曝光同步，网络/传感器延迟仍未知。手机端使用因果先到样本，记录自己的规则，不能与主机这一双侧最近邻混称。

采集台以 `--imu-host <Atom私有IPv4> --imu-tcp-port 3336` 配置，或在页面填写 Atom IMU 地址。开始/预览才接入；每个 `frames.jsonl` 记录增加 `imu`，段 metadata 增加完整覆盖率及有符号均值、绝对 p95/max 残差。连续 IMU 原始流另存在同会话 `imu-<id>/`，保留段间准备过程。

当前原局域网可直接使用本轮人工安装参考启动采集台：

```powershell
& artifacts.local/hardware-bringup/venv/Scripts/python.exe -B research/active/hardware-bringup/host/real_labeled_capture_web.py --tof-host 192.168.5.10 --imu-host 192.168.5.9 --imu-mounting artifacts.local/work/real-labeled-capture-20261011/installation-imu-operator-pilot-20261012.json
```

打开 `http://127.0.0.1:8791/`，输出自动生成到新采集目录；该参考会显示“人工零参考”和回正差 3.856718°。既有实时演示启动命令加同样的 `--imu-host` 和 `--imu-mounting` 即采用该姿态来源。热点地址需重新发现；电脑与手机采集不能同时占用单客户端 ToF。结束电脑服务用 Ctrl+C 释放设备。

## 简单标定步骤

固定同一整套支架，确认 ToF 光轴水平（可用水平仪与正对竖直墙面的操作约束）。不要仅凭支架底座水平推断光轴水平。水平静止记录 fused `gravity_unit`；再整套支架明确向下俯约 30°，记录另一重力均值。无需准确的 30° 就能得到俯转平面，角度准确度仍取决于操作者的水平基准。再向上仰约 30°核对符号/方向，回到正对墙面的水平姿态核对零点。保留各次原始包和实际操作记录。

```powershell
& artifacts.local/hardware-bringup/venv/Scripts/python.exe -B research/active/hardware-bringup/host/atom_imu_probe.py record --host 192.168.5.9 --seconds 60 --out artifacts.local/work/atom-imu/static --phase static
```

用不同未存在的目录分别录 `level/down/up/wall`，然后：

```powershell
& artifacts.local/hardware-bringup/venv/Scripts/python.exe -B research/active/hardware-bringup/host/atom_imu_probe.py derive --level artifacts.local/work/atom-imu/level/summary.json --down artifacts.local/work/atom-imu/down/summary.json --mounting-id head-bracket-20261012 --out artifacts.local/work/atom-imu/imu-mounting.json
```

候选始终为 `verified:false`。算法将水平重力 `g0` 作为 up，由 `g0*dot(g0,gDown)-gDown` 确定 forward，再用 `cross(forward,up)` 确定 right；`tof_from_imu_rotation` 的行为 `[right,-up,forward]`，对应模型 ToF 的 X 向右、Y 向下、Z 向前。实际水平、上仰和墙面核对分别写入 `level_reference/up_check/wall_check`，各设 `verified:true`，最终人工核对通过才设顶层 `verified:true`；没有实测依据不能升级。

安装记录 schema `atom.imu.mount.v1` 还要求 `mounting_id`、`atom_device_id`、`axis_relation`、`evidence`。矩阵必须正交且行列式 +1，设备 ID 必须匹配。把它与当前 `installation-view.json` 的路径一起保存；沿用 `cnh_zone_orientation` 已核对的原始 zone 180°映射，不由相机的显示旋转或镜像推出 IMU 外参。支架改变后，两份安装关系都需要复核。

本轮人工操作形成 `OPERATOR_PILOT_ZERO`：`verified:true` 仅表示记录的水平、下俯、上仰和回正操作已完成。`absolute_optical_pitch_accuracy_verified:false`、`tof_atom_extrinsic_independently_measured:false` 保留未知绝对精度；输出来源为 `MEASURED_IMU_OPERATOR_PILOT_ZERO`，同时携带 `calibration_scope`、`verification_level`、`zero_repeat_residual` 和安装记录路径/校验值。候选矩阵保持，不用回正偏差调零救回结果；人工下俯/上仰约 30°不构成准确角度真值。可用记录为 `artifacts.local/hardware-bringup/atom-imu-v1/imu-mounting-operator-pilot.json`，与既有 `installation-view.json` 关联收据一起保存。

`gToF = R*gIMU` 后，模型 pitch 为 `asin(gToF.z)`，负值表示光轴向下；roll 为 `atan2(-gToF.x,-gToF.y)`。这只是重力俯角/横滚，没有 yaw、平移或曝光时钟标定。

## 实时引擎

`m3_live_demo.py` 加 `--imu-host ... --imu-mounting <已核验安装记录>` 后使用每个 CNH 对齐的实测姿态投影当前曝光，保留旧曝光各自已投影的特征；不按新姿态重投影全部历史，也不因普通角度变化逐帧清空。姿态来源/boot/可用性断裂才重新暖机。未配置、未标定、设备不匹配或过期时明确保留 `NOMINAL_MOUNT_ASSUMPTION`，逐帧输出实际采用的名义俯角。冻结权重、阈值和历史权重保持。当前场景背景减法及未知平移/yaw仍为原工程限制。

`atom_imu_probe.py` 的静止报告包括样本标准差、首/末 5 个设备秒均值之差、样本数和设备时间跨度。设备跨度 ≤10 秒时首尾窗口重叠，漂移字段为 `null` 并说明原因，不能把短预览的零差当静态漂移。静止读数只说明传感器噪声/漂移，不证明光轴零点准确或避障能力。
