# EGOMOTION_COMPENSATED_LOOMING_SIGNAL_R0 公共来源机械与权威审计目标（2026-07-25）

状态：`PREREGISTERED_METADATA_ONLY`

## 一、目的

ADT 的冻结 cohort 已因三个必需 cell 不足而停止。本审计只回答两个下一来源问题：

1. AV2 的 10 Hz truth 到 20 Hz camera join 是否可冻结，以及固定车载刚性相机能否
   产生真实的 `PURE_EGO_ROTATION_NO_CLOSING`；
2. CODa 的 TDR v2.3 / TACC bytes 是否有 immutable binding，官方 tiny 数据是否
   具备连续 10 秒 bbox/camera 分母。

本边界不下载或解码 RGB、LiDAR、Feather/JSON payload member，不生成 cell
proposal，不运行任何 signal，不冻结 role split。

## 二、输入与隔离

- AV2 inventory 固定为
  `av2_official_inventory_r0.json`，SHA-256
  `69cd1a22422dc4a6a1a128399a3b8f268dae6b2378fc45cf2d82add05e1d9e12`；
- AV2 join cohort 只从有 annotation 的 train/val log 中按
  `SHA256(split + TAB + log_id)` 排序取前 24；cohort 永久为
  `SOURCE_PRESCREEN_ONLY`；
- CODa 只读 DOI/TDR API、TACC HEAD、目录列表和 ZIP64 central directory；
- 旧 LILocBench/CrowdBot frame、window、outcome、score、threshold 均不可读；
- candidate RGB signal、route、lifecycle、App、shadow、human、production
  读取/改动计数固定为 0。

## 三、冻结判定

### AV2

- join 仅允许 source-native lidar/annotation `timestamp_ns` 到同一 log、
  `ring_front_center` 文件名 timestamp 的唯一最近邻；
- 容差固定为官方 AV2 API 的半个 20 Hz frame：`25,000,000ns`；
- tie、missing 或超差均 abstain；不得把 10 Hz truth 插值或复制到全部 20 Hz；
- `log_id` 只可作为约 15 秒 session identity。没有发布的 parent
  drive/capture/burst ID 时，不得升级为 `capture_cluster_id` 或冻结三 role；
- 官方采集平台若是刚性固定在车辆上的相机、没有独立头部/相机旋转自由度，则车辆
  yaw 伴随平移不得重标为 `PURE_EGO_ROTATION_NO_CLOSING`。该真实 cell 缺失时，
  AV2 终态只能是
  `AV2_REQUIRED_PURE_ROTATION_CELL_STRUCTURALLY_ABSENT / VALID`。

### CODa

- TDR identity 必须包含 explicit DOI version、datafile ID、size 与发布 checksum；
- TACC 的 filename/size/mtime/短 ETag 不升级为密码学 checksum；
- TDR tiny 与 TACC tiny 容器不同且没有官方 per-member equivalence manifest 时，
  不能互相证明 bytes 或替代 full sequence archive；
- tiny 的连续性只由 central-directory member filename 计算，不读取 member；
- 10 Hz 下需至少 100 个连续 bbox frame 才可能形成一个 10 秒窗口。任一 sequence
  最大连续 run 不足 100 时，禁止下载 9.1 GB tiny 回救；
- full TACC archive 没有官方 checksum/version binding 时，连 pose/bbox member 的
  Range 提取也保持关闭。

## 四、终态与下一边界

- AV2 缺必需真实 cell：
  `AV2_REQUIRED_PURE_ROTATION_CELL_STRUCTURALLY_ABSENT / VALID`；
- CODa full authority 未闭合且 tiny 不连续：
  `HOLD_CODA_BOUNDED_PRESCREEN / VALID`；
- 两者都不得计入 R0 三条真实 source family 分母；
- 下一合法边界只能转向具有独立头部/相机旋转与连续 source-native geometry 的
  头戴来源，或另立新的、预先设计、按 capture/session 隔离的受控采集协议。

这两个终态都不是 looming 算法失败，因为所有 arm 与连续 `G_t` 尚未运行。
