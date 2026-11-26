# 实机接入与融合排查：当前状态

所属：**避障线**（与毕设线的区分见[两条工作线](../../../docs/WORK_LINES.md)）。
更新：2026-11-26。跨路线决定见[总研究决定](../../../docs/CURRENT_DECISION.md)；本页收拢前完整原文见 Git `a0114b6f`（更早 10 月版本见 `cd227114`），各轮详细结果与失败沿用原报告与 [RUNS.md](RUNS.md)。

## 1. 方向 A（UniDepth-V2-S × 单一尺度链）已确认并接入 App `10.20.1`

- **独立确认与室内归因（11-17–18）**：在未见 SANPO 胸前 138 段 / 1,969 帧上按冻结配对协议（`evalkit v1.1`）确认 `UniDepth-V2-S (490×644) × (ToF → 自身地面 → 沿用)` 胜过现行 DAV2-S 路径（C2 通过，C1 退路不可评；[胸前确认](DIRECTION_A_CONFIRM_SANPO_CHEST_20261117.md)）。室内 ADT 开发集不胜旧法（LOW 召回 `-42/1395`），经拆解系 LOW/BODY 带归属差异而非漏障，未采用 ToF `scale+offset`。[室内归因](DIRECTION_A_INDOOR_ADT_DEV_20261118.md) · [手机分辨率](DIRECTION_A3_PHONE_RESOLUTION_20261117.md)
- **App 接入与手机实测（11-18，64 / `10.20.1` demo 包）**：Atom 固化射线版 UniDepth-V2-S（`658×490` HTP，~135 MB）已作为融合页可切换深度路径接入 `MobileFusionActivity`（默认仍为 DAV2-S）。S24 Ultra 实测对研究版中位相对误差 **`0.68%`**，demo 包中位耗时 **`130 ms + 16 ms`**；5 段旧头戴手机回放中，有 ToF 时盒内躯干测距误差由 `0.12–0.22 m` 降至 **`0.02–0.03 m`**。实时三路联调待眼镜硬件到位。[接入与回放记录](DIRECTION_A_APP_INTEGRATION_20261118.md)

## 2. 11 月避障机理与转向确认排查结论（均为 `EXPLORE`）

- **悬挑 D8 与 HoloSet 素墙/横管因果排查（11-17–19）**：
  - Oxford Spires 实图插入：ASE 训练 D8 无重训检出 `17/24`（对照 `2/12`），补全真实支撑与阴影后升至 `21/24`（细杆/隔板 `5/12 → 9/12`），消融证实增益主要来自支撑结构而非阴影。[插图 v2](SPIRES_INSERT_V2_DEV_20261117.md) · [消融](SPIRES_INSERT_ABLATION_DEV_20261117.md)
  - HoloSet 横管回程残余报警定位于紧前 `0.20–0.40 s` 画面右上角偏近历史单目深度；实测严格因果多视角 RGB 删点及 `25%–90%` 历史占据软降权均无法在不引起去程真实横管中途断报下消警，保留原 D8。[横管与因果排查](HOLOSET_TUBE_PASSAGE_DEV_20261118.md) · [真实素墙](REAL_WALL_DATA_DEV_20261118.md)
- **侧墙、地面过滤与 5–15 cm 浅障（挡轮块/路缘，11-18）**：
  - 通道外侧墙剔除未消去整条报警；固定 RGB 地面过滤可消去 `57/94` 户外地面误报帧，但独立低障检出由 `25/28 → 22/28`。[侧墙归因](SIDEWALL_ATTRIBUTION_DEV_20261118.md) · [地面过滤](GROUND_ALERT_FILTER_DEV_20261118.md)
  - 独立户外新场地及配对挡轮块复核中，局部高差/边缘规则在同误报成本下未能稳定提升 5–15 cm 浅障检出与提前量（3 根新挡轮块首报同帧，近距 1.5 m 内边缘 `2/3` vs 直接 `1/3`，空路误报相同），暂停作为跨地点提前提醒主创新，保留对照及两处新停车场近距保持检验方案。[浅障比较](SHALLOW_ALERTS_DEV_20261118.md) · [提前量复核](SHALLOW_EDGE_LEAD_20261118.md) · [近距保持](SHALLOW_RETENTION_20261118.md)
- **视场认知与转向确认（11-16–17）**：
  - ADT/LaMAR 复盘确认：RGB 相对小 ToF 的主要增益来自**更宽视场**，ToF 的核心作用是**公制尺度**。[ADT 视场对比](ADT_RGB_VS_TOF_FOV_DEV_20261117.md) · [头戴数据集调研](HEADWORN_DATA_SURVEY_20261117.md)
  - Mobilio 10 人留出 640 次转向中，原 `0.2 s` 常速外推相对首次入带多 `31` 次正确、少 `192` 次误确认，角度余量候选未过同风险门槛；PING 跨队列真实指令复现因公开集缺逐次时标判 `NOT_RUN`（ADVIO 32 窗口未见新增补漏），已沉淀通用同步回放接口。[转向确认报告](MOBILIO_TURN_CONFIRMATION_20261117.md)

## 3. 手机采集、融合基线与硬件待办

- **手机采集与基础融合**：SM-S9280 当前 `64 / 10.20.1`（延续 `10.19.1` 写盘与预览脱耦修复：12 s 预览 `23.74 fps`、IMU 缺序号 `0`），支持 Atom RGB、H3 完整 `8×8×16` CNH 与 BMI270 录制及后台 ZIP 导出；默认首页辅助仍为 H3 基础 ToF。[采集操作](../../../docs/PHONE_DATA_CAPTURE.md) · [演示入口](../../../docs/HARDWARE_OBSTACLE_DEMO.md)
- **保留的冻结基线**：头戴冻结检验保留 `v1 / v2 / v3 / v6`（ADT 独立评测支持 `v6`，误报 `7.74% → 5.93%`、1.5 m 内检出 `81.2% → 80.6%`）；俯角/地面/噪声联合栈与模拟默认候选（[CNH 指针](../dtr-r0/nearfield/CNH_SIM_DEFAULT_CANDIDATE.json)）保持；ADT 模拟角度细扫推荐实机 ToF 下倾约 `30°`（`25–35°`）。[ADT 真值评测](ADT_EVAL_RESULT_20261013.md) · [ToF 倾角分析](ADT_TOF_AIM_BACKBONE_DEV_20261013.md) · [10-13 阶段总结](STAGE_SUMMARY_20261013.md)
- **下一步（待硬件与佩戴）**：眼镜到位后完成物理安装角（建议 ToF 下倾 ~30°）与外参重标，在 `MobileFusionActivity` 开展方向 A 与既有冻结链路的真实佩戴三路联调，并按[冻结执行入口](FROZEN_FUSION_EVAL_RUNBOOK.md)录制新独立片段。

[硬件账本](README.md) · [无线协议](WIRELESS_PROTOCOL.md) · [运行记录](RUNS.md) · [早期历史](CURRENT_HISTORY_20260924.md)
