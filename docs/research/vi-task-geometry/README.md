# VI-Task Geometry current

状态：`current / WILD_LAB / VITG_G0_PREOUTCOME_PROTOCOL_FROZEN / ATOMS3R_M12_ONBOARD_BMI270_CONFIRMED / BLOCKED_ON_SYNCHRONIZED_CAPTURE_CALIBRATION_IMPLEMENTATION_LOCK_AND_FRESH_ROSTER / REAL_G0_NOT_RUN / NO_TOF / NO_TRAINING / DEFAULT_APP_UNCHANGED`

## 结论

BlindAssist 已有的 AtomS3R-M12 同时提供 OV3660 RGB 与板载 BMI270 六轴 IMU，因此这条路线
不需要购买外部 ToF，也不需要再外挂 IMU。真正缺失的是同一设备时钟域下的可审计 RGB-IMU
capture、camera/IMU calibration、独立高度真值和 fresh roster，而不是传感器芯片本身。

VI-Task Geometry 不是 SATOM-R0 rescue，也不是 GA-SATOM 的无 ToF 改名。它把未来有限预算从
physical range allocation 改为 computation/feature/keyframe/parallax allocation；G0 尚不检验这些
主动策略，只问 RGB+IMU 能否建立 metric camera height 与 sparse ground geometry。

## G0

- [Frozen protocol](VITG_G0_RGB_IMU_METRIC_FRAME_PROTOCOL_2026-08-15.json) 固定 A0 DepthART-only、
  A1 fixed height、A2 VIO sparse ground、A3 A2-aligned DepthART；
- [Capture contract](../../../scripts/research/vi_task_geometry_g0/README.md) 强制同刚体、同 MCU clock、
  连续 camera frame ledger、100–400 Hz IMU、20 ms 最大 gap 和 5 ms 最近样本同步门；
- G0 primary 只允许 A2/A3 用绝对门证明 metric-frame observability。Clearance 指标只作诊断，
  不影响 G0 PASS；
- 当前没有合格 fresh capture，也未冻结具体 VIO source/configuration，故不运行任何 arm，
  `REAL_G0_NOT_RUN`。

## ARCore 边界

Galaxy S24 Ultra 官方支持 Depth API，ARCore Depth 可利用手机相机运动且不强制 ToF；但它管理的是
手机自身 AR camera，不能把 AtomS3R MJPEG 注入为跟踪相机。仓库既有 ARCore D45 又曾在 900 updates
中得到 0 个 exact-timestamp fresh raw-depth observation，因此保持 isolated teacher/benchmark
`NOT_EVALUABLE`，本轮不重跑、不调门，也不把手机 IMU用于眼镜视频。

## 唯一 successor

`VITG_G0_ATOMS3R_RGB_IMU_CAPTURE_PREFLIGHT`：先实现并编译验证 ToF-disabled OV3660+BMI270 writer，
完成 intrinsics、IMU-to-camera extrinsics、同钟同步验证、reference 与 exact roster activation；在此
之前不刷机、不采集、不打开 truth/outcome。

G0 失败或不可评估即关闭 VI metric task-geometry 路线，下一条独立表示只能是 height-free
gravity-normalized angular/TTC risk。默认 App 影响：`否`。
