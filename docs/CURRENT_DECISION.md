# 当前研究决定

本页只记录**避障线**的跨路线研究决定；毕设线见[两条工作线](WORK_LINES.md)。各路线完整指标、协议与负向明细见各自 `CURRENT.md` 与报告。

## 2026-11-26 跨路线研究决定

主线为盲杖互补前视感知，M3和A+LOCAL保留；原5格局部融合与L2保留。以下三条并行探索线均处于 `EXPLORE`，未触碰官方受保护 `test` 集，不自动替换默认 `:app` 基础 ToF 辅助。

### 1. EgoGeom-Omni（头戴有无 ToF 通吃基础几何主力）
- **保留 `v4.1` 主干，不晋升 `v4.2`**：`v4.1` 集成 M5 芯片级双修（HTP 实片误差降 `3.0–4.2×`）与 M1 真实 H3 有符号相关 CNH 分支（1,209 帧真实 H3 极丢帧走廊帧间抖动降 `6.0–7.9×`）。反幻觉重训 `v4.2` 虽提升 ToF/Aria BEV MAE，但未解 Aria 清路近距幻觉（`519 → 531/954`）且 CODa 退化，不替换 `v4.1`。[主线状态](../research/active/egogeom-omni-v0/CURRENT.md) · [V4 五题](../research/active/egogeom-omni-v0/math_theory/RESULTS.md) · [V5 九题与 Aria 三轮](../research/active/egogeom-omni-v0/math_theory/RESULTS_V5.md)
- **确立 S24U 混合精度 + `G2` 玻璃头端侧方案**：淘汰全 W8A8（保真与零干扰不达标），采用 **`v4.1 pc_fp16dec_a16 + G2`**：`Pure-RGB + G2`（`58.94 MB`）HTP `14.48 ms`、`Omni + Signed-H3 + G2`（`60.56 MB`）HTP `17.14 ms`，深度与玻璃芯片级零干扰双 `0.000000`；1,800 次满载温升零降频（DDR 峰值较 FP16 低 `9.3 °C`）；`GlassRecon` 穿透玻璃救回 `82.91%`。
- **确立三通道穿戴读出与 Android 原生闭环**：保留 `EgoGeomTemporalBevV3`（极线视差 `1.07 ms` CPU，38 源 M6 留一交叉检验 `8 → 32/365`）与 `EgoGeomWearableRiskEngine`（`fuse_m6par_ang08_gyro` `CF=63/365 @ 3.10 cw/km`；零 VIO/零 ToF `cam_imu_novio_fuse` `CF=40/365 @ 0.00 cw/km`）；`:core:vision` 单测与 `:app` 原生 QNN HTP 进程测试（`24.03 ms`）均通过。[穿戴架构与预算](../research/active/egogeom-omni-v0/cane_blind_band_probe/CAMERA_IMU_OPT_TOF_ARCHITECTURE.md)

### 2. EgoClear-v0（纯视觉端到端碰撞感知独立线）
- **确立 `EgoClear-v14` Hyper-Omni 为纯视觉主力**：20 阶段迭代并根治 QAIRT 2.47 `Clip(min='')` 缺陷，S24U 300 帧稳态 `37.69 ms`（Mixed `27.82 ms`）。4 域 126 场景 / 11,569 帧全新 0 交集盲考验证其在真实头戴（冻结 `0.50` `54.7%` 第一、暗光零衰减）、室内公寓（亮/暗 `0.0%`）及 4 域光学异常 `UNKNOWN` 拒答（`12/12 = 100%`）上成立；低机位机器人 `Fresh CODa` 过触发保留 `ENS(v6.6+v14)`（`58.5%`）互补。下一步用预装 `:egoclear-demo-app` 开展实走录回放。[当前状态](../research/active/egoclear-v0/CURRENT.md) · [126 场景盲考](../research/active/egoclear-v0/PHASE20_FRESH_UNSEEN_BLIND_PROBE_20261122.md)

### 3. 实机融合与避障机理排查（`hardware-bringup`）
- **方向 A（UniDepth-V2-S × 单一尺度链）接入 App `10.20.1` 作为可切换路径**：在未见 SANPO 胸前 138 段 / 1,969 帧上确认整条链胜过现行 DAV2-S（C2 通过，C1 退路不可评；室内 ADT 开发集归因为 LOW/BODY 带归属差异）；已以 `658×490` Atom 固化射线 DLC 接入 `MobileFusionActivity`（默认保持 DAV2-S），5 段旧头戴回放盒内测距误差降至 `0.02–0.03 m`，待眼镜实机三路联调。[接入记录](../research/active/hardware-bringup/DIRECTION_A_APP_INTEGRATION_20261118.md) · [胸前确认](../research/active/hardware-bringup/DIRECTION_A_CONFIRM_SANPO_CHEST_20261117.md)
- **明确不采用 / 暂停主张项**：
  - HoloSet 横管回程残余报警的因果多视角 RGB 删点与历史占据软降权（`25%–90%`）均无法在不造成真实横管中途断报下消警，保留原 D8。[HoloSet 排查](../research/active/hardware-bringup/HOLOSET_TUBE_PASSAGE_DEV_20261118.md)
  - 5–15 cm 浅障局部高差/边缘规则在独立新场地同误报下未稳步提升提前量，暂停作为跨地点提前提醒主创新，保留对照与近距保持子实验。[浅障近距保持](../research/active/hardware-bringup/SHALLOW_RETENTION_20261118.md)
  - Mobilio 10 人留出 640 次转向中角度余量候选未过同风险门槛，保留原 `0.2 s` 常速外推；PING 跨队列真实指令复现因公开集缺逐次时标为 `NOT_RUN`。[转向确认](../research/active/hardware-bringup/MOBILIO_TURN_CONFIRMATION_20261117.md)
- **保留既有冻结与实机基线**：原实机头戴冻结检验 `v1 / v2 / v3 / v6`（ADT 独立评测支持 `v6`，误报相对降 23%）、俯角/地面/噪声联合栈及采集 `10.19.1` 性能优化保持。[硬件当前状态](../research/active/hardware-bringup/CURRENT.md)

### 4. 已关闭路线：`walk-alert-r0`（提醒策略与多帧记忆）
- **2026-11-17 正式关闭**：287 段 SANPO-Synth + 视障步速闭环（373 事件）及 LaMAR/TUM 真实深度小试证明——仅改提醒策略（P1）受限于单帧感知下限；多帧记忆与占据栅格虽降漏报下限（`20.1% → 2.4%`），但误报升至 `1.93×–3.1×`，自由空间清除仅追回 `14%`，退化/外扩掩码无法根除行人边缘残点与拖影切碎。[阶段总结](../research/active/walk-alert-r0/STAGE_SUMMARY_20261117.md)

---

## 保留历史基线与证据边界

- **ToF 模拟默认候选**：保持去 `bin0` + 仪器 DR + 新 S + `LOCAL v4` 合同（[默认指针](../research/active/dtr-r0/nearfield/CNH_SIM_DEFAULT_CANDIDATE.json)），仅模拟 Development 口径，不自动接入 App；原 M3/5 格保留为强档来源与对照。
- **半合成融合历史结论**：保留融合确认 v4 主中带低误报排序 PASS（[报告](../research/active/dtr-r0/nearfield/SYNC_FUSION_CONFIRM_V4_DEV_20261011.md)）及 v1–v3 原结果与失败记录，不替代实机验证；RGB 分带混合因 FREE 超预声明门槛未采用。
- **边界**：所有模型与算法当前均为 `EXPLORE` 开发或独立离线评测证据，未构成真实佩戴/手持导航安全证明；实走、佩戴三路联调与 TalkBack/大字无障碍回归仍待补齐。

本页 2026-10-13 全文历史见 Git `243463f1`（更早 v3 整合前见 `95f451e6`）；历史细目保留于各路线 `RUNS.md` 与原报告。
`ALGORITHM_EXPLORATION / M3_RETAINED / HARDWARE_DEFERRED`
