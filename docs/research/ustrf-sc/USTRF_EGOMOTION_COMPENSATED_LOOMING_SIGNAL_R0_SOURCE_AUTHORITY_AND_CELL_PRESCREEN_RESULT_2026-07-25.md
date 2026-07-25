# EGOMOTION_COMPENSATED_LOOMING_SIGNAL_R0 来源权威与 cell 预筛结果（2026-07-25）

状态：`SOURCE_AUTHORITY_CANDIDATES_PRESENT_CELL_PRESCREEN_REQUIRED / VALID`

这不是 source admission、数据 split 或算法结果。当前只允许继续对尚未终结的来源
做不读取候选 RGB signal 的 inventory / manifest / bounded prescreen 核验；
`ADMITTED=0`，所以
raw flow、bbox growth、无补偿扩张、旋转补偿扩张和 oracle/self-motion 均未运行。

## 结论

新路线有足够的真实来源候选，因而不是“无数据可研究”；但公开网页元数据不能证明
冻结协议要求的 `source × role × counterfactual cell` 分母。当前最合理的顺序是：

1. `Aria Digital Twin`：头戴域、逐帧 6DoF、深度、对象 pose/geometry 最完整；
   但冻结的 16 条 groundtruth-only cohort 仅在一个必需 cell 产生 proposal，
   已以 `ADT_CELL_PRESCREEN_INSUFFICIENT / VALID` 停止，不扩 RGB 回救。
2. `UT CODa`：独立机器人域，pose、RGB-D/LiDAR 与 3D track 有潜力，下一步先核验
   被标注的分钟实际覆盖哪些独立 sequence。
3. `Argoverse 2 Sensor`：车载压力源，日志、6DoF 与 3D track 充足；只能代表
   低位车载域，且固定刚性相机结构上缺少纯相机转动 cell，已停止 R0 payload
   prescreen。
4. `Waymo Open Perception`：保留为 AV2 备选或第四压力源，不能和 AV2 一起
   冒充两个独立头戴/手持域。
5. `HoloAssist` 与 `Bonn RGB-D Dynamic`：允许 bounded payload prescreen，
   但当前没有 source-native 持续对象/表面 identity 足以直接闭合连续 `G_t`。

因此总体仍是 `SOURCE_AUTHORITY_CANDIDATES_PRESENT_CELL_PRESCREEN_REQUIRED`，
不是 `GO_ADMITTED`；ADT 自己已从 metadata probe 进入并停止于
`PRESCREEN_INSUFFICIENT`。任何候选若不能在
discovery / validation / sealed holdout 三个 role 中分别满足冻结的四类反事实
cell 与 session 分母，就直接
`FAIL_CLOSED_NEW_DATA_OR_TRUTH_AUTHORITY_BLOCKED`，不能用合成数据、切分同一次
capture 或旧窗口补分母。

## 本轮证据

机器清单：
`artifacts.local/evidence/ustrf/egomotion_compensated_looming_r0/source_audit/source_authority_inventory_r0.json`

官方来源页：

- [Aria Digital Twin](https://facebookresearch.github.io/projectaria_tools/docs/open_datasets/aria_digital_twin_dataset)
- [UT Campus Object Dataset](https://amrl.cs.utexas.edu/coda/)
- [Argoverse 2 Sensor](https://argoverse.github.io/user-guide/datasets/sensor.html)
- [Waymo Open Dataset](https://waymo.com/open/about/)
- [HoloAssist](https://holoassist.github.io/)
- [Bonn RGB-D Dynamic Dataset](https://www.ipb.uni-bonn.de/data/rgbd-dynamic-dataset/)

HoloAssist 只下载了两个 metadata/annotation 文件，没有下载 10.8 GB 相机包或视频：

| member | bytes | SHA-256 | 能证明什么 |
| --- | ---: | --- | --- |
| `holoassist_data_splits_v1_2.zip` | 10,554 | `e10674e7ac32957386d5e88afa60acf3335251b92ba2f8cfb97a736dc85e1621` | split 中共有 2,111 个 recording ID |
| `holoassist_annotations_trainval_v1_1.json` | 117,011,015 | `cc7898b49958a62fe021ae2ffa53a709c3fd6f45fd3f893960b8aac6d13dfe9c` | 1,758 个 train/validation video；action 标签可用于 prescreen |

这些 action 标签不等于 looming truth，也不证明任一 10 秒 session-cell。候选 signal
没有被读取或计算。

AV2 的官方匿名 S3 已通过 ListObjectsV2 做了实际 inventory，不下载任何 object
payload：

- `train / val / test = 700 / 150 / 150`，共 1,000 个 log UUID；
- 三个 split 的 deterministic first log 均存在 9 路 camera、lidar、intrinsics、
  extrinsics、6DoF ego pose 与 map；train/val 有 `annotations.feather`，test 没有；
- 示例 train log 列出 3,035 个 object key、319 帧/相机、157 个 lidar member，
  仅由 server key/size/ETag 形成 metadata identity；
- receipt SHA-256：
  `69cd1a22422dc4a6a1a128399a3b8f268dae6b2378fc45cf2d82add05e1d9e12`。

后续 [公共来源机械与权威审计](USTRF_EGOMOTION_COMPENSATED_LOOMING_SIGNAL_R0_PUBLIC_SOURCE_MECHANICS_AND_AUTHORITY_AUDIT_RESULT_2026-07-25.md)
已证明 10/20 Hz join 可冻结：24 条 outcome-blind metadata cohort 中
`3,761/3,762` anchor 唯一匹配到 `25ms` 内，0 tie。但固定车载刚性相机没有
独立头部/相机旋转自由度，且 log UUID 不是 parent capture authority，因此终态为
`AV2_REQUIRED_PURE_ROTATION_CELL_STRUCTURALLY_ABSENT / VALID`；不下载四表或 RGB
回救。

ADT Explorer 的四个官方 metadata API 也已实际列出：

- 236 条 sequence（Apartment 184、LiteOffice 52），共 29,252.65 秒；
- 2,832 个 download entry，全部有 filename、40-hex SHA-1 与 size，总压缩体积
  2,281,582,033,333 bytes；
- CDN URL 是短期签名链接，immutable identity 必须使用
  `sequence_id + filename + SHA-1 + size`，不能冻结 URL；
- manifest 没有官方 capture-cluster UID。由 sequence 名推导的 218 个 name-base
  与 18 个双设备簇不是官方 schema，不能直接用于 role split；
- overview 同页的 `236 total` 与 `284 apartment + 52 office` 自相矛盾；实时
  Explorer 与官方 HF card 支持 `184 + 52 = 236`，本轮以可下载 inventory 为准并
  披露该退化；
- 数据适用专用 ADT license，不把 Project Aria Tools 的 Apache-2.0 当数据许可。

因此 ADT 已到 `DOWNLOADABLE_INVENTORY_VERIFIED`，但仍须通过有界 geometry
prescreen 才可能进入 admission。

在读取任何 groundtruth payload 前，已按四个 metadata proxy stratum 各冻结 4 条、
共 16 条 singleton-name-base sequence；之后只取得对应的 16 个
`main_groundtruth.zip`，总计 `705,566,181` bytes，逐文件官方 SHA-1 通过。
`main_vrs`、RGB preview、depth 与 segmentation 均未取得，RGB/VRS member count
为 0。activity proxy 不是 cell truth；下一步必须先按独立预注册的
[ADT geometry cell prescreen](USTRF_EGOMOTION_COMPENSATED_LOOMING_SIGNAL_R0_ADT_GEOMETRY_CELL_PRESCREEN_GOAL_2026-07-25.md)
只读 source-native geometry，再经双模型 review。该预筛现已闭合，结果见
[ADT geometry cell prescreen result](USTRF_EGOMOTION_COMPENSATED_LOOMING_SIGNAL_R0_ADT_GEOMETRY_CELL_PRESCREEN_RESULT_2026-07-25.md)：

- 16 条 sequence / 16 个 singleton component 永久标记为
  `SOURCE_PRESCREEN_ONLY`；
- accepted-eligible 非 skeleton object proposal 依次为
  `0 / 5 / 0 / 0`；
- `PURE_EGO_ROTATION_NO_CLOSING`、
  `STATIONARY_EGO_ACTIVE_TARGET_APPROACH` 与
  `LATERAL_PASS_NO_SUSTAINED_CLOSING` 均不足；
- skeleton proposal 未实现且按预注册只能 diagnostic，不能修复 accepted 分母；
- 终态为 `ADT_CELL_PRESCREEN_INSUFFICIENT / VALID`，candidate signal、旧窗口和
  RGB/VRS 读取均为 0。

所以 ADT 当前 cohort 已停止：不扩 ADT RGB、不冻结 role split、不运行任何 arm。
这只是 source/cell availability 失败，不是 looming 算法失败。

UT CODa 的官方 TACC/TDR inventory 同样只做 HEAD 与 ZIP64 central-directory
range，不解码 payload：

- 23 个 sequence archive、总计 1,784,055,483,972 bytes；23 条仅对应 12 个
  capture date，role split 至少必须按 date 联组；
- 3D bbox member 共 27,876 帧、覆盖 sequence 0–20；3D semantic 共 5,085 帧、
  覆盖 18 条；存在碎片化，不能从 member 名推导四 cell；
- dense local pose 覆盖 23/23，dense global 缺 sequence 8/14/15；sequence 22
  缺 metadata JSON，sequence 21 的 cam0/cam1 数相差 13 帧；
- TACC archive 没有发布密码学 checksum，且其 2023-12-30 文件与 TDR v2.3
  （2026-03-28）缺 immutable binding；ZIP CRC32 不能升级为密码学 authority。

后续只读 TDR/TACC 审计进一步闭合为
`HOLD_CODA_BOUNDED_PRESCREEN / VALID`：TDR v2.3 tiny 有 datafile ID、size 与
MD5，但 TACC full archive 没有官方 checksum/version binding；tiny bbox/cam0
任一 sequence 最大连续 run 均仅 3 frame，达到 100-frame/10-second 的 sequence
为 0。因此不下载 9.1 GB tiny，也不 Range 提取 full TACC member。

## 旧证据防火墙

机器 receipt：
`artifacts.local/evidence/ustrf/egomotion_compensated_looming_r0/source_audit/old_window_admission_firewall_r0.json`

只读身份审计纠正了一个容易混淆的点：

- 旧 15 正 / 15 负窗口本体只包含两个 LILocBench source：
  `lilocbench_dynamics_0_front` 与
  `lilocbench_lt_changes_dynamics_0_front`，合计 4,594 个唯一帧；
- 四个 CrowdBot source 属于后续 canonical 41-sequence / 62,229-frame
  inventory，并不是旧 30-window manifest 的成员；
- firewall 同时拒收整个 LILocBench 与 CrowdBot family，但把上述两个
  identity domain 分开记录。

admission firewall 可以读取 source/session/member/raw-frame identity 与 hash，
不得读取旧 report、frame score、paired outcome、排名或阈值。新 source producer
只能收到 clean-room pass/reject receipt。当前旧 manifest 还缺 decoded-pixel hash
与转码/裁剪 near-duplicate fingerprint；因此真正解码新 payload 前仍须补齐这两层
检查，缺失时 fail closed。

## 来源状态

| 来源 | 当前许可 | 未闭合的 admission 条件 |
| --- | --- | --- |
| ADT | `PRESCREEN_INSUFFICIENT / HOLD_R0_ADMISSION` | 16 条 cohort 仅一个 cell 有 proposal；禁止扩 RGB 回救 |
| UT CODa | `HOLD_CODA_BOUNDED_PRESCREEN / HOLD_R0_ADMISSION` | tiny 不连续；full TACC 缺 checksum/version binding |
| AV2 Sensor | `REJECT_REQUIRED_CELL_STRUCTURALLY_ABSENT / HOLD_R0_ADMISSION` | join 可用，但固定车载 rig 缺纯相机转动 cell 与 parent capture ID |
| Waymo | `GO_METADATA_PROBE_BACKUP / HOLD_R0_ADMISSION` | cell、drive clustering、与 AV2 域重叠 |
| HoloAssist | `GO_BOUNDED_PAYLOAD_PRESCREEN / HOLD_R0_ADMISSION` | source-native identity 与连续 closing truth |
| Bonn RGB-D Dynamic | `GO_BOUNDED_PAYLOAD_PRESCREEN / HOLD_R0_ADMISSION` | persistent dynamic identity、cell、session 独立性 |
| HOT3D / EgoBody / Ego-Exo4D | `HOLD_ACCESS*` | 注册/许可或 truth binding 未闭合 |
| ASE / TartanAir V2 | `HOLD_DIAGNOSTIC_ONLY` | 合成；不得计入三条真实来源 |
| nuScenes | `REJECT_R0` | 500ms 单位所需 truth/pose authority 被 AV2/Waymo 严格替代 |

## 下一道门

ADT、AV2、CODa 三条优先路径都已 fail closed，不继续扩样本或 payload。下一合法
工作只能对 HOT3D 等真实头戴 source family 另立不含候选 RGB signal 的 immutable
inventory / bounded cell prescreen，或者先取得新的、预先设计并按
capture/session 隔离的采集授权：

- canonical version、license、download member、server checksum/size；
- sequence/log ID、真正 capture-cluster ID 与同主体/相邻 burst grouping key；
- timestamp clock/rate、intrinsics、pose、depth/geometry、object track member；
- source-native object/surface identity 是否存在；
- 只由官方 metadata 可以确认的 scene 字段。

之后才允许从官方 preview 或极小 sample 建立 cell review bundle，并按 R0 的双模型
独立复核与必要时第三模型裁决冻结 cell。cell receipt 完成前：

- 不冻结 discovery / validation / holdout split；
- 不下载大包或解码候选 RGB 做 flow；
- 不计算任何 arm；
- 不选择报警阈值；
- 不接 App、route、lifecycle、shadow、human 或 production。
