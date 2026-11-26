# 项目现在做到哪里

本页只覆盖**避障线**；毕设线另见[两条工作线](WORK_LINES.md)。

更新：2026-11-26。跨线决定以[总决定](CURRENT_DECISION.md)为准；分路线详见 [EgoGeom-Omni](../research/active/egogeom-omni-v0/CURRENT.md)、[EgoClear-v0](../research/active/egoclear-v0/CURRENT.md) 与 [实机融合](../research/active/hardware-bringup/CURRENT.md)。

## 当前可用与端侧交付

- **主 App（64 / 10.20.1）**：首页“开始辅助”延续当前 H3 固件基础 ToF 提醒（旧 UDP 保留 A+LOCAL）；数据采集页支持 Atom RGB、H3 完整 8×8×16 CNH 与 BMI270 三路记录与后台 ZIP 导出（写盘与预览脱耦，IMU 缺序号 0）；融合演示页新增“深度模型”开关，可在 **DAV2-S（默认）** 与 **方向 A：UniDepth-V2-S × 单一尺度链（`ToF → 自身地面 → 沿用`，`658×490` HTP `130 + 16 ms`）** 之间切换，5 段旧头戴回放盒内测距误差由 `0.12–0.22 m` 降至 `0.02–0.03 m`。[融合接入记录](../research/active/hardware-bringup/DIRECTION_A_APP_INTEGRATION_20261118.md) · [演示入口](HARDWARE_OBSTACLE_DEMO.md) · [采集说明](PHONE_DATA_CAPTURE.md)
- **EgoGeom-Omni Android 原生接入（debug 隔离，不改默认入口）**：`:core:vision` 落地纯 Kotlin `EgoGeomContracts`、`EgoGeomTemporalBevV3`（含 `1.07 ms` 极线视差门控）与 `EgoGeomWearableRiskEngine`（单测全过）；`:app` 落地 C++17 QNN HTP JNI（`EgoGeomQnnRuntime`）与 debug 演示页 `EgoGeomWearableDemoActivity`，S24 Ultra 真实 APK 进程内 `59.1 MB` `v4.1 pc_fp16dec_a16` 全往返中位 `24.03 ms`、`0 NaN`、零干扰 `0.000000 m`。
- **独立演示与可教应用**：`:egoclear-demo-app`（预装 `EgoClear-v14 / v6.6` 全 NPU 实时手持演示）已装机待实走录回放；`:teachable-object-app` 已完成精确增量 B 教学、持久化缓存、嵌入复用与保存前预警/撤销交付。[独立应用目录](../apps/README.md)

## 三条并行避障研究线（均为 `EXPLORE`，未读官方受保护 `test`）

- **EgoGeom-Omni（头戴有无 ToF 通吃几何主力）**：交付 `v4.1` 硅片精确 + 真实 H3 有符号 CNH 旗舰档（`51.18M / 50.75M`），经 14 道数学实测与 S24U 混合精度优化，落地 `v4.1 pc_fp16dec_a16 + G2` 冻结玻璃语义头合图——`Pure-RGB + G2`（`58.94 MB`）HTP **`14.48 ms`**、`Omni + Signed-H3 + G2`（`60.56 MB`）HTP **`17.14 ms`**，深度与玻璃芯片级零干扰双 `0.000000`，1,800 次满载温升零降频（DDR 峰值较 FP16 低 `9.3 °C`）；1,209 帧真实 H3 走廊回放抖动降 `6.0–7.9×`；`GlassRecon` 穿透玻璃救回 `82.91%`；三通道穿戴融合 `fuse_m6par_ang08_gyro` 将 38 源共享交叉拟合推至 `63/365`（`+40.0%`），零 VIO/零 ToF `cam_imu_novio_fuse` 以 `0.00 cw/km` 达 `40/365`。[当前状态](../research/active/egogeom-omni-v0/CURRENT.md)
- **EgoClear-v0 → v14（纯视觉端到端碰撞感知）**：确立 `EgoClear-v14` Hyper-Omni（`27.97M`，根治 QAIRT 2.47 `Clip(min='')` 缺陷）为纯视觉主力，S24U 300 帧稳态 `37.69 ms`（Mixed `27.82 ms`）；4 域 **126 场景 / 11,569 帧**全新 0 交集无损盲考完结：`Fresh SANPO-Real` 冻结 `thr=0.50` `54.7%` 全场第一且暗光零衰减、`Fresh ARKit` 亮/暗 `0.0%`、4 域光学异常 `UNKNOWN` 拒答 `12/12 = 100%`；诚实暴露低机位机器人 `Fresh CODa` 过触发（双融合 `ENS(v6.6+v14)` 压至 `58.5%`）。[当前状态](../research/active/egoclear-v0/CURRENT.md)
- **实机融合与机理排查（`hardware-bringup`）**：方向 A（UniDepth-S 尺度链）在未见 SANPO 胸前 138 段确认胜过旧 DAV2-S 路径并接入 `10.20.1`；完成悬挑 D8、素墙/横管因果删点与占据软降权、5–15 cm 浅障局部高差与 Mobilio/PING/ADVIO 转向确认系统排查，多项启发式已按负向结果收口。[当前状态](../research/active/hardware-bringup/CURRENT.md)

## 关键缺口与下一步

1. **真实佩戴与手持实走闭环（各线共性缺口）**：眼镜到位后在 `MobileFusionActivity` 试跑方向 A（`10.20.1`）与 `EgoGeom-Omni v4.1 pc_fp16dec_a16 + G2`，并按已冻结规则做独立头戴评测；仅用手机可先打开预装 `:egoclear-demo-app` 实走录回放（走廊/下台阶）跑真机闭环评测。
2. **真·盲杖盲区专项实测与无障碍回归**：补充真实纯悬空头障与透明玻璃门实拍样本，并补齐 TalkBack / 大字无障碍回归（目前标 `NOT_TESTED`）。

定位仍为盲杖互补（cane-complementary）的前视感知研究原型。设备时钟独立，离线回放、半合成与工程检查均不构成真实导航安全证明；`UNKNOWN` 不等于无障碍。

## 按任务进入

[两条工作线](WORK_LINES.md) · [研究决定](CURRENT_DECISION.md) · [EgoGeom-Omni](../research/active/egogeom-omni-v0/CURRENT.md) · [EgoClear-v0](../research/active/egoclear-v0/CURRENT.md) · [实机融合](../research/active/hardware-bringup/CURRENT.md) · [避障历史（DTR）](../research/active/dtr-r0/CURRENT.md) · [提醒与多帧记忆探索（已关闭）](../research/active/walk-alert-r0/STAGE_SUMMARY_20261117.md) · [代码地图](CODE_MAP.md) · [硬件路线](GLASSES_HARDWARE_ROUTE.md) · [设备回归](DEVICE_REGRESSION.md)

本页 2026-10-13 整理前全文保留在 Git `243463f1`（更早版本见 `83fbc28a`）；旧实验结果、失败与各自停止规则仍以原报告为准。
L10 暂停、DTR R2 动态路线保留历史证据（`L10_R0_PAUSED / DTR_R2_DYNAMIC_RETAINED`），`walk-alert-r0` 已于 2026-11-17 正式关闭，均不作为当前推进线。
