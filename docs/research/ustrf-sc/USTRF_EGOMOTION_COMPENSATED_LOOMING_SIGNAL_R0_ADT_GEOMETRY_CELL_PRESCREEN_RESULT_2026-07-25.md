# EGOMOTION_COMPENSATED_LOOMING_SIGNAL_R0 ADT 几何 cell 预筛结果（2026-07-25）

状态：`ADT_CELL_PRESCREEN_INSUFFICIENT / VALID`

## 结论

ADT 是真实、可下载、pose/geometry 完整的强来源候选，但本轮 16 条
`SOURCE_PRESCREEN_ONLY` sequence 不能闭合 R0 强制的四类反事实 cell。按冻结的
10 秒不滑窗、visibility、camera/object geometry 规则，accepted-eligible
非 skeleton object proposal 为：

| cell | 独立 sequence proposal |
| --- | ---: |
| `PURE_EGO_ROTATION_NO_CLOSING` | 0 |
| `EGO_APPROACH_STATIC_SURFACE` | 5 |
| `STATIONARY_EGO_ACTIVE_TARGET_APPROACH` | 0 |
| `LATERAL_PASS_NO_SUSTAINED_CLOSING` | 0 |

三个必需 cell 小于最低 `2` 个 sequence，因此无需读取 RGB 或计算任何 arm 就已
fail closed。ADT 继续 `HOLD_R0_ADMISSION`；本轮不扩 ADT RGB、不运行 signal、
不拆 discovery/validation/holdout，也不接 App、route 或 lifecycle。

## 边界

这里的三个 `0` 只指 accepted-eligible、source-native visibility 通过的非 skeleton
object proposal，不声称所有允许 target 的几何提案都是零。10/16 archive 含
skeleton，但 skeleton body-center 没有人体表面合同，在预注册中只能是
diagnostic，不能获得 `PRESCREEN_CELL_ACCEPTED` 或修复必需 cell 分母；本轮明确记录
`SKELETON_DIAGNOSTIC_PROPOSAL_COVERAGE_NOT_IMPLEMENTED`。

这也不是 looming 算法失败。raw flow、bbox growth、无补偿、rotation-compensated、
oracle rotation 和 full-6DoF diagnostic 全部未运行；当前失败只发生在 ADT
source/cell availability 门。

## 隔离与输入

- 读取任何 groundtruth 前，以固定 metadata proxy 冻结 16 条 sequence；
- 只取得 16 个 `main_groundtruth.zip`，共 `705,566,181` bytes，逐文件官方
  SHA-1 通过；
- RGB preview、VRS、depth、segmentation 均未取得；
- 16 条永久为 `SOURCE_PRESCREEN_ONLY`，不能进入任何后续 role；
- 旧 LILocBench/CrowdBot frame、outcome、score、threshold 读取为 0；
- visibility firewall 只消费 `stream_id / object_uid / timestamp /
  visibility_ratio`，bbox 坐标/尺寸/面积不进入 producer。

机器终态：
`artifacts.local/evidence/ustrf/egomotion_compensated_looming_r0/source_audit/adt_geometry_cell_prescreen_terminal_r0.json`

proposal SHA-256：
`2d0aba986d40ef004a968df6adc2c3d494e36a9bae3e2228b77b031cda7ea6b3`。

## 独立核验

独立审查先在预注册中发现并闭合 5 个 blocker：skeleton clock 二次映射、
object timestamp exact-match、缺 visibility firewall、prescreen 污染未来 role、
capture-cluster 未联组。

producer 后，独立实现又复算两条 archive：

- `Apartment_release_clean_seq134_M1292` 精确复现最早 static-approach proposal
  的 window、target、device path/displacement 与 range；
- `Lite_release_recognition_Mug_2_seq031_61283` 复现 0 proposal。

审查还发现 skeleton diagnostic 未实现及若干潜在 fail-closed 检查；最终实现补上
device quaternion/finite/orthogonality/determinant、duplicate timestamp/identity、
exact 16-member set 与收窄的 accepted-eligible claim。5 项 focused unit tests 和
terminal validator 均通过。

## 下一边界

不继续扩 ADT 样本回救本次 prescreen。R0 若继续，只能：

1. 对另一条真实 source family 另立同样 outcome-blind 的 bounded cell prescreen；
2. 或在取得新的、预先设计且按 capture/session 隔离的采集授权后，直接采集四类
   反事实 cell。

在至少 3 个真实 source family 同时闭合四 cell、role/session 分母与 clean-room
firewall 前，looming arm comparison 保持关闭。
