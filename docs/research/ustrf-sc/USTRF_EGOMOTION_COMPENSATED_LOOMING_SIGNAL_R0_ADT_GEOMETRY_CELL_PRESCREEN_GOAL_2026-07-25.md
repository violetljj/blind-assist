# EGOMOTION_COMPENSATED_LOOMING_SIGNAL_R0 ADT 几何 cell 预筛目标（2026-07-25）

状态：`PREREGISTERED / GROUNDTRUTH_ONLY / NO_SIGNAL`

## 一、目的与权限

本边界只回答：ADT 的 source-native camera/object/skeleton geometry 中，是否存在足以
进入人工智能双审的 10 秒反事实候选窗。它不做 source admission，不冻结
discovery/validation/holdout，不下载或解码 RGB/VRS，不计算任何 optical-flow、
bbox-growth 或 looming arm。

这 16 条 sequence 永久标为 `SOURCE_PRESCREEN_ONLY`，不得在现在或未来计入
discovery、validation、sealed holdout 的 session/cell 分母。后续三 role 只能在本轮
未读取 groundtruth 的新 capture cluster 上，使用本轮冻结且不再改变的规则建立。

父目标、旧窗口防火墙、四 cell、每 session-cell `10s / 20 × 500ms epoch` 及
App/route/lifecycle 禁止条件全部不变。metadata activity 只用于在读取 groundtruth
前形成 bounded prescreen strata，不是真值标签。

## 二、冻结输入

- metadata freeze：
  `artifacts.local/evidence/ustrf/egomotion_compensated_looming_r0/source_audit/adt_groundtruth_prescreen_freeze_r0.json`
  （SHA-256 `c71885c2d7ad9ff448fbafd593d1b3fa6b45fd6454bb0c60f3e1ddd58f08c6a6`）；
- acquisition receipt：
  `artifacts.local/evidence/ustrf/egomotion_compensated_looming_r0/source_audit/adt_groundtruth_prescreen_acquisition_r0.json`
  （SHA-256 `ba7354ccd4436663ff658277fd2f079c11ffbb8316511bd0999b5406cac2d8fe`）；
- 16 个 `main_groundtruth.zip`，总计 `705,566,181` bytes，逐 member 官方 SHA-1
  已通过；RGB/VRS member 为 `0`；
- 只允许读取 `metadata.json`、`aria_trajectory.csv`、`scene_objects.csv`、
  `3d_bounding_box.csv`、`instances.json`、可选 `Skeleton_*.json` 与
  `skeleton_aria_association.json`；
- `2d_bounding_box.csv` / `2d_bounding_box_with_skeleton.csv` 只允许在隔离
  visibility firewall 中读取 `stream_id / object_uid / timestamp /
  visibility_ratio`。`x/y/min/max` 不得传给 producer/reviewer，不计算面积或
  growth；`eyegaze.csv` 不读取。

## 三、时间、坐标与距离

- `metadata.json` 必须同时满足 `dataset_version=="2.0"`、
  `gt_time_domain=="DEVICE_CAPTURE"`；否则整条 sequence abstain；
- 以 `aria_trajectory.tracking_timestamp_us × 1000` 为 device clock；
- timestamped object pose 对每个 device timestamp 只取唯一最近项，要求
  `abs(t_object_ns - 1000*t_trajectory_us)<=1000ns`；等距 tie、同 identity
  多项或超容差均 abstain。`timestamp=-1` 表示静态；
- V2.0 / `DEVICE_CAPTURE` 的 skeleton `frames[].timestamp_ns` 已是 device
  capture clock，直接使用，不应用 `dt_optitrack_minus_device_ns`；后者只原样
  记录。最近匹配绝对差必须 `<=20ms`，等距 tie/多义即 abstain；
- device/object quaternion 均按字段名重排为 `wxyz` 后归一化；norm 偏差
  `>1e-3` 或 rotation 非正交即该 sample invalid；
- object surface range：将 device world position 变换到 object local frame，再求
  point-to-local-AABB 的欧氏距离；AABB 任一边 `<=0` 或对角线 `<0.20m` 不作
  prescreen target；
- skeleton 只用全部有限 joint 的逐轴 median 作为 body-center range。它不是人体
  表面距离或最终 `G_t` authority，只可输出
  `SKELETON_GEOMETRY_DIAGNOSTIC_CANDIDATE`，不能获得
  `PRESCREEN_CELL_ACCEPTED`；
- device angular speed 固定为
  `norm([angular_velocity_x_device, angular_velocity_y_device, angular_velocity_z_device])`；
  device path length 为相邻有效 `t_world_device` 欧氏步长之和，endpoint
  displacement 为首末有效位置欧氏距离；
- signed closing intensity为 `C_t=-d log(max(r_t,0.20m))/dt`，用相邻 source
  timestamp 一阶差分；prescreen 只报告它，不替代父目标尚待冻结的最终导数器。

## 四、capture cluster 与 visibility firewall

- 校验 `metadata.serial` 以 sequence UID 的 device token 结尾；不一致则 sequence
  abstain；
- 在全部 236 个 sequence UID 上先按去除 device token 的 name-base 建无向 sibling
  边；再读取本 16 条的 `metadata.concurrent_sequence`：非空引用必须指向 inventory
  中存在的对端，并且只有两端 metadata 均已取得且互相引用时才视为可核验边；
- sibling 与可核验 concurrent 边取无向 connected component。引用悬空、单向、
  serial 冲突或 component 歧义时，整个相关 component abstain，禁止拆分；
- 本轮报告 sequence 与 component 两种分母。即使 component 合格，它仍永久是
  `SOURCE_PRESCREEN_ONLY`；
- visibility firewall 只接受 RGB stream `214-1`、同 target identity、唯一最近
  timestamp `<=1000ns` 且 `visibility_ratio>=0.50` 的 sample；bbox 坐标与尺寸
  读取计数必须为 0。

## 五、窗口与质量门

- 每个 sequence 从首个 device timestamp 起按 `[0,10s)、[10s,20s)…` 划分，
  不滑窗、不按结果平移；
- 窗口必须覆盖 `>=9.9s`，含 `>=270` 个有效 device sample，最大 device gap
  `<=100ms`；
- `persistent target` 固定指同一 target identity 在窗口的 20/20 epoch 中，每个
  epoch 都有至少 10 个同时满足 range 与 visibility firewall 的 sample，且全窗
  有效 device timestamp coverage `>=90%`；
- `C_t` 只由相邻两个都有效且时间差 `<=100ms` 的 range sample 形成；每个 epoch
  至少 9 个有效 `C_t`，否则该 target/window 不可提案；
- 每个 epoch 的 `r` 与 `C` 分别为该 500ms 内有效 sample 的 median；
  `r_start/r_end` 是首/末 epoch median，`r_min` 是 20 个 epoch-median 的最小值；
  “首/末 30%”固定为前 6 / 后 6 个 epoch-median 的 median；
- 同一 sequence/cell 若有多个合格提案，只保留最早 start；同 start 有多个 target
  时按 `target_type + target_identity` 字典序取首个，不按 margin、closing 强度或
  视觉结果选择。一个 sequence 可以提出多个不同 cell，但后续
  role split 必须整 sequence/推断 capture cluster 联组。

## 六、冻结几何 proposal 规则

下列数值只筛 review bundle，不是报警阈值、算法接受阈值或最终 cell truth。

### `PURE_EGO_ROTATION_NO_CLOSING`

- 10 秒 device endpoint displacement `<=0.35m`；
- device path length `<=0.75m`；
- angular-speed median `>=0.25rad/s`；
- 对被提名 persistent static-object / dynamic-object target，
  `abs(r_end-r_start)<=0.25m` 且 `P90(max(C_t,0))<=0.05/s`。

### `EGO_APPROACH_STATIC_SURFACE`

- device endpoint displacement `>=0.75m`；
- target 为 `scene_objects.timestamp=-1` 的 persistent static object；
- `r_start-r_end>=0.60m`、`r_end<=3.0m`；
- `median(max(C_t,0))>=0.025/s`。

### `STATIONARY_EGO_ACTIVE_TARGET_APPROACH`

- device endpoint displacement `<=0.35m`、path length `<=0.75m`；
- target 为 timestamped object 或 mapped skeleton，且 target world endpoint
  displacement `>=0.60m`；
- `r_start-r_end>=0.60m`、`r_end<=3.0m`；
- `median(max(C_t,0))>=0.025/s`。

### `LATERAL_PASS_NO_SUSTAINED_CLOSING`

- target 可为 static object 或 timestamped object；mapped skeleton 只形成独立
  diagnostic candidate，不进入 accepted-cell 分母；
- range minimum 落在窗口 `[20%,80%]` 内；
- `r_start-r_min>=0.50m` 且 `r_end-r_min>=0.50m`；
- 首 30% 的 `median(C_t)>=0.02/s`，末 30% 的 `median(C_t)<=-0.02/s`；
- `abs(r_end-r_start)<=0.50m`。

## 七、模型复核与 fail-closed

visibility firewall 只向 producer 发布 target/timestamp 的 boolean visible receipt
和计数。producer 只能输出每个提案的 source/component/sequence/window/target
identity、输入 hash、quality denominator 和上述几何量；bbox 坐标/尺寸/面积、
candidate signal、旧窗口、旧 outcome、route、lifecycle 读取计数必须为 0。

cell 需要语义判断，因此几何提案不是最终 cell：

1. 先冻结 review bundle、允许字段、prompt 与 SHA；
2. 两个互不可见的新上下文独立输出 `ACCEPT / REJECT / ABSTAIN` 及单一首因；
3. 分歧才由第三个新上下文裁决；
4. 只有双审/裁决接受、target 非 skeleton 且 source-native
   time/cluster/geometry/visibility 门通过的窗口，才记为
   `PRESCREEN_CELL_ACCEPTED`；
5. 不足 16 条、任一 cell 少于 2 个独立 sequence，或 capture-cluster identity
   仍不能安全联组，终态为
   `ADT_CELL_PRESCREEN_INSUFFICIENT / VALID`，不得扩 RGB、运行 signal 或拆 role
   回救。

即使四 cell 均出现，本边界最高也只是
`ADT_CELL_PRESCREEN_CANDIDATES_PRESENT / VALID`；ADT R0 admission 仍需至少 8
session、三 role 分母、正式 split、最终 `G_t`/projection contract 与全部父门。
