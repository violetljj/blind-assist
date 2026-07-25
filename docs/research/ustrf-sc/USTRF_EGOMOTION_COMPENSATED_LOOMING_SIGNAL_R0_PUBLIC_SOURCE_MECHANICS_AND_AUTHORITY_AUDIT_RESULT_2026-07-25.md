# EGOMOTION_COMPENSATED_LOOMING_SIGNAL_R0 公共来源机械与权威审计结果（2026-07-25）

子边界状态：`AV2_REQUIRED_PURE_ROTATION_CELL_STRUCTURALLY_ABSENT / VALID`；
`HOLD_CODA_BOUNDED_PRESCREEN / VALID`

父 R0：`NOT_EXECUTED`。三来源组合只形成非终态审计摘要，不能冒充父 R0
四种合法终态之一。

## 结论

ADT、AV2 与 CODa 三条优先公共来源路径都不能授权 R0 signal comparison：

| source | 终态 | 直接原因 |
| --- | --- | --- |
| ADT | `ADT_CELL_PRESCREEN_INSUFFICIENT / VALID` | 冻结 16 条 cohort 的四 cell proposal 为 `0 / 5 / 0 / 0` |
| AV2 | `AV2_REQUIRED_PURE_ROTATION_CELL_STRUCTURALLY_ABSENT / VALID` | 固定车载刚性相机没有独立头部/相机旋转自由度 |
| CODa | `HOLD_CODA_BOUNDED_PRESCREEN / VALID` | 未绑定的 TACC tiny 不连续；有 MD5 的 TDR tiny 连续性未评价，且 full TACC bytes 缺 DOI-version binding |

当前仍是 `ADMITTED=0`。这不是 looming 算法失败：raw flow、bbox growth、
无补偿局部扩张、rotation-compensated、oracle rotation 与 full-6DoF diagnostic
全部未运行。下一合法边界只能是新的真实头戴来源，或在录制前冻结新的受控采集。

## AV2：同步可用，但反事实机械结构不成立

只从已冻结的 AV2 inventory 中取 train/val，按
`SHA256(split + TAB + log_id)` 排序选择前 24 条
`SOURCE_PRESCREEN_ONLY` log。没有 GET Feather/JPEG/LiDAR payload，只列 S3 key、
size、ETag 与 filename timestamp。

官方 AV2 API `v0.3.6` commit
`b7321d1f71f6ce0ecdd151f4f2b648338c191edd` 的 synchronization contract
已冻结为：

- anchor 是 10 Hz lidar filename `timestamp_ns`；
- 只联结同 log、同 `ring_front_center` 的唯一最近 JPEG filename timestamp；
- 容差是半个 20 Hz frame，即 `25,000,000ns`；
- tie、missing、超差 abstain；不把 10 Hz truth 插值/复制到全部 20 Hz frame。

24 条 metadata cohort 的结果：

- lidar anchor：`3,762`；
- 唯一且 `<=25ms`：`3,761`；
- tie：`0`；
- 超差：`1`，位于
  `train/8184872e-4203-3ff1-b716-af5fad9233ec`，
  最近差值 `40,329,220ns`；
- per-log median delta 的 median：`9,970,532ns`。

所以只能确认 lidar-filename→camera-filename 的机械联结；本轮没有读取
`annotations.feather`，annotation truth→camera join 为 `NOT_EVALUATED`，且审计
goal 未预注册“usable”支持率门，不能把 `3761/3762` 事后升级为 source 准入。
24 log 所需 annotation、ego pose、
extrinsics、intrinsics 四表合计仅 `18,805,176` bytes，但本轮没有下载，因为：

1. 官方采集是刚性固定在 Ford Fusion 车顶的相机；
2. 车辆 yaw 通常伴随平移，不能重标为
   `PURE_EGO_ROTATION_NO_CLOSING`；
3. 合成 rotation-only render 不能补真实来源分母；
4. `log UUID` 只能作为约 15 秒 session ID，官方没有发布 parent
   drive/capture/burst ID，不能证明三 role 隔离。

AV2 可保留为 ego-approach、stationary-ego/active-target、lateral-pass 的车载
diagnostic pressure source，但不能计入父目标要求的三条完整真实来源。

## CODa：运动平台合适，但 authority 与连续性不同时成立

TDR DOI `10.18738/T8/BBOQMV` 的 released `v2.3` 对 tiny 三个 datafile 发布了
明确 datafile ID、size 与 MD5：

| datafile | bytes | MD5 |
| --- | ---: | --- |
| `299625 / part001` | 4,294,967,296 | `61da09d525cd7d2627412eb2a13f7466` |
| `299626 / part002` | 4,294,967,296 | `e97dc0815ff32483d6c2138e092caea1` |
| `299627 / part003` | 518,241,581 | `e0d97f2141c9ee21537e664ab1228993` |

这能 immutable-bind TDR tiny 的三段容器，但本轮没有下载或读取其目录，因而不能
评价其连续性，也不能绑定 23 个 full TACC sequence ZIP：

- TDR tiny 是三段 tar.gz，共 `9,108,176,173` bytes；
- TACC tiny 是 ZIP，`9,108,343,009` bytes；
- 没有官方 per-member equivalence manifest；
- full TACC sequence 只有 filename、size、2023-12-30 mtime 与短十进制
  ETag，没有发布密码学 checksum，也没有绑定到 DOI v2.3 的 manifest。

本轮只读取**未与 TDR v2.3 建立等价绑定的** TACC tiny ZIP64 central directory，
未提取 member。三个 Range 请求均验证 HTTP `206`、精确 `Content-Range` 和长度，
合计只读 `994,820` bytes。其 `7,050` entry
中：

- bbox：`1,374` frame，覆盖 sequence 0–20，任一 sequence 最大连续 run 为 `3`；
- cam0：`1,618` frame，覆盖 sequence 0–21，任一 sequence 最大连续 run 为 `3`；
- 达到 10 Hz 下连续 10 秒所需 `100` frame 的 sequence 数均为 `0`。

因此结论是“TACC tiny 的可见目录不满足连续性门，TDR tiny 连续性
`NOT_EVALUATED`”，不能把二者拼成同一条证据。当前仍禁止下载 9.1 GB TDR tiny，
也禁止从 full TACC sequence Range 提取
pose/bbox member。只有发布方补充 full archive 的官方 checksum/version manifest
后，CODa 才能重开 bounded prescreen。

## 隔离与机器证据

- AV2 receipt：
  `artifacts.local/evidence/ustrf/egomotion_compensated_looming_r0/source_audit/av2_join_and_cell_mechanics_terminal_r0.json`
  ，SHA-256
  `0fb94d7adf11abf1a8e52f45320a666b624343459f7fff23fcd3775af64f5e03`；
- CODa receipt：
  `artifacts.local/evidence/ustrf/egomotion_compensated_looming_r0/source_audit/coda_tiny_continuity_and_binding_terminal_r0.json`
  ，SHA-256
  `7f4ffdc46b9ffe6cc850f810c4742e04021095bcaaade660bab907ba2c9f970c`；
- 三来源非终态摘要：
  `artifacts.local/evidence/ustrf/egomotion_compensated_looming_r0/source_audit/priority_public_source_summary_r0.json`
  ，SHA-256
  `3dd0612ca8e82198a69ee1b9b0828827a98e71b820e708c66490f8fc73f9458e`。

candidate RGB/LiDAR payload decode、旧窗口/outcome 读取、signal、role split、
alarm threshold、App、route 与 lifecycle 改动均为 0。独立只读复核发现并修正了
组合终态越权、TACC/TDR 跨容器归因、AV2 annotation join 越界和 Range 防护问题；
最终由 source validator 复验各 receipt hash、边界字段与非终态 summary。

## 下一边界

1. metadata-only 核验 HOT3D：必须确认 full access、连续 `>=10s`、Aria 6DoF、
   persistent 3D object/hand identity、capture grouping 与四 cell 机械可能性；
2. 若 HOT3D 或其他头戴来源仍不能闭合，必须在录制任何 outcome-bearing video
   之前预注册新的受控采集；
3. 未取得至少三条真实 source family 的完整分母前，不下载候选 RGB，不运行 arm，
   不选择阈值，不接 App/lifecycle。
