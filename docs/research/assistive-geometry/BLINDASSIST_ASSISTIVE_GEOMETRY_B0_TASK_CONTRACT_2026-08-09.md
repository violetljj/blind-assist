# BlindAssist Assistive Geometry B0 task contract

状态：`PARTIALLY_FROZEN / PRE_OUTCOME / EXECUTION_NOT_AUTHORIZED / UNRESOLVED_BLOCKERS_PRESENT`

日期：`2026-08-09`

机器合同：[JSON](BLINDASSIST_ASSISTIVE_GEOMETRY_B0_TASK_CONTRACT_2026-08-09.json)

## 1. B0 只回答什么

B0 冻结 Assistive Geometry 的输入、输出、任务语义、评价与数据防火墙。它不训练模型、
不选择数据 outcome、不激活 DepthART D1/R2，也不修改默认 App。

研究问题是：

> 在同一产品成像与 GeometryState 合同下，DepthART-initialized 多任务模型是否比
> DepthART depth-only + 冻结后处理更好地保持真实通行空间，尤其减少 false-clear？

## 2. 产品成像合同

绑定的生产请求真源为 `core/device/.../CameraXFrameSource.kt`：默认后摄、请求 `640×480`、
4:3 优先、`CLOSEST_HIGHER_THEN_LOWER`、RGBA8888、`KEEP_ONLY_LATEST`、24 FPS。
请求尺寸不是实际尺寸证明；每个 session 必须记录实际 camera id、buffer shape、`cropRect`、
`rotationDegrees`、row/pixel stride、capture timestamp 和 Camera2 intrinsics/transform authority。

模型坐标冻结为：

1. 只使用有效 `cropRect`，不得额外 center crop；
2. 按 `rotationDegrees ∈ {0,90,180,270}` 转到 display-upright；
3. 保持全有效 FOV，沿用 DepthART 官方 `keep_aspect_ratio=true`、`lower_bound`、
   `ensure_multiple_of=32`；
4. ImageNet RGB normalization：mean `[0.485,0.456,0.406]`、std
   `[0.229,0.224,0.225]`；
5. 名义 portrait 有效图 `480×640 (W×H)` 映射到固定 NCHW
   `1×3×608×448`，即 tensor `H=608,W=448`；不 pad、不机械拉伸到 square；
6. bicubic 用于 RGB，dense truth/validity 必须分别使用预冻结的 metric-safe
   resampling，不能用 RGB 插值规则静默处理标签；
7. K 必须依次应用 crop、rotation 与最终 `sx=448/W_display`、`sy=608/H_display`；
8. graph 输入为 `image + dynamic transformed K`，不允许写死 SM-S9280 的数值 K。

任何实际 shape、crop、rotation、K 或映射 receipt 缺失/不支持时，整帧为
`UNKNOWN_INPUT_GEOMETRY`，不得回退成 nominal K 或 `CLEAR`。

当前 `CameraAnalysisGeometryMapper` 及 r832/r834 只提供 benchmark/deterministic evidence，
尚未建立产品 runtime intrinsic mapping authority。这是 B0 的显式 blocker。

## 3. GeometryState R0

`GeometryState` 必须绑定：

```text
identity:
  session_id / frame_id / captured_at_ns / source_id

camera:
  buffer_shape / crop_rect / rotation_degrees
  K_sensor / sensor_to_buffer / K_display / K_tensor
  transform_receipt_sha256

outputs:
  dense_depth_m + depth_valid
  ground_probability + ground_valid
  optional ground_plane_normal_camera + camera_height_m
  clearance_m[left, center, right]
  occupancy_probability[left, center, right][1.0m, 1.5m, 2.0m]
  task_confidence[left, center, right][1.0m, 1.5m, 2.0m]
  final_state = CLEAR_OBSERVED / OCCUPIED_OBSERVED / UNKNOWN
  unknown_reason[]

provenance:
  encoder / checkpoint / head / postprocess / precision / backend
```

三 band 在 display-upright 有效视场按归一化 x 冻结为：

```text
left   [0, 1/3)
center [1/3, 2/3)
right  [2/3, 1]
```

研究用身体扫掠包络冻结为 `body_half_width=0.32 m`、`lateral_margin=0.10 m`，总半宽
`0.42 m`；horizon 为 `[1.0,1.5,2.0] m`。这是沿用既有 shadow geometry 的 R0 研究
profile，不代表所有用户，也不产生产品安全包络权限。后续个性化必须另立 profile/version。

clearance 使用有效 intrusion support 的第 2 百分位；depth 有效范围先固定为
`[0.25,6.0] m`。实际 ground/obstacle truth reader 必须在 roster outcome 前绑定实现与哈希。

## 4. UNKNOWN 与 Confidence 必须拆开

最终 UNKNOWN 不是一个可被 student 学成“负类”的类别，而是两层合取：

```text
deterministic_observability_valid
AND
model_task_confidence >= frozen_threshold
```

deterministic invalid 包括：输入几何/K receipt 缺失、truth 不可配对、depth/ground/support
不足、timestamp 过期、非有限输出或 unsupported transform。模型 confidence 只表达任务正确性，
不能覆盖 deterministic invalid。

confidence correctness target 冻结为：

```text
abs(clearance_error) <= 0.25 m
AND
三个 horizon occupancy decision 全部正确
```

confidence operating threshold 暂为 `UNRESOLVED`，必须在 Development-train 内部 calibration
split 上预冻结，不能使用 Development-selection 或 confirmation outcome。

## 5. Truth、teacher 与 pseudo-label

| 字段 | 可用 truth | teacher/pseudo-label 用途 | 权限上限 |
|---|---|---|---|
| depth | 注册 RGB-D/可验证 metric geometry | DA2/DepthART/Metric3D depth distillation | teacher preservation，不是安全 truth |
| ground | 注册 depth + K + 有效 ground reader；或独立 ground annotation | teacher mask 只作辅助 | 缺 ground truth 时该 loss/metric 为 UNKNOWN |
| clearance/occupancy | 注册 depth/K/body profile 生成的独立 task truth | teacher clearance 可作蒸馏辅助 | false-clear 必须相对独立 truth 计算 |
| confidence | 相对独立 truth 的任务正确性 | teacher disagreement 仅在 C0 验证相关后使用 | disagreement 不能自动叫 uncertainty |

任何字段缺失只降低对应 claim capability；不能把整份异质数据判废，也不能把 UNKNOWN
当 negative。Metric3D/DA3 在 B1/B2 不得参与 selector；它们只在 C0 互补性门后进入 C1。

## 6. 基线与消融顺序

保持一个共享 encoder、输入、数据、optimizer family 和训练预算，按以下 additive arms 执行：

| Arm | 增量 |
|---|---|
| `A0_DEPTH_ONLY` | DepthART-initialized depth head + 冻结 geometry postprocess |
| `A1_PLUS_GROUND` | `A0 + ground` |
| `A2_PLUS_CLEARANCE` | `A1 + direct clearance/occupancy` |
| `A3_PLUS_FALSE_CLEAR` | `A2 + asymmetric false-clear` |
| `A4_PLUS_CONFIDENCE` | `A3 + confidence/UNKNOWN interface` |

loss family 冻结为 masked log-depth、valid-neighbor gradient、ground BCE/plane、clearance
Huber `delta=0.25 m`、三 horizon occupancy BCE、truth-occupied 正例权重 `3.0` 的 false-clear
项，以及 confidence correctness BCE。具体 lambda、optimizer 数值、epoch 和 batch 只能在 B1
训练协议中一次性冻结；B0 不授权默认值。

至少使用三个预声明 seed。checkpoint selection 必须先过 coverage/finite/UNKNOWN 前门，再按
`false-clear → clearance MAE → temporal delta` 字典序选择；不能用 AbsRel 救任务失败。

## 7. 冻结评价门

沿用已经预冻结、不是为本路线结果事后设计的任务门：

| 指标 | 绝对门 |
|---|---:|
| known coverage | `>= 0.90` |
| ground recovery | `>= 0.90` |
| clearance MAE | `<= 0.20 m` |
| false-clear / all known | `<= 0.08` |
| false-block / truth clear | `<= 0.02` |
| temporal clearance-delta MAE | `<= 0.15 m` |
| geometry transition agreement | `>= 0.90` |
| valid-to-UNKNOWN rate | `<= 0.10` |
| confidence ECE, 10 bins | `<= 0.10` |
| worst-parent false-clear / all known | `<= 0.12` |

所有指标同时报告 pooled 与 unweighted parent/session macro；pooled 不能救 macro 失败。还必须
报告 false-clear / truth occupied、false-block / truth clear、每 band/horizon、indoor/outdoor、
near-field、低光/模糊和最差 parent。

B2 算法贡献门预冻结为：A4 相对 A0 不得恶化任一风险绝对门，并至少满足一项：

- clearance MAE 相对改善 `>=5%`；
- false-clear/all-known 绝对改善 `>=0.01`。

三 seed 中至少 2/3 同方向，median seed 通过贡献门，且任一 seed 不得依赖 coverage 降至
绝对门以下。该门只产生 Development algorithm candidate，不授权部署或替换 DA2。

## 8. 数据角色与防火墙

B0 后续必须新建三个 parent/session/capture-disjoint roster：

```text
TRAIN            teacher/materialization + model fitting
DEVELOPMENT      checkpoint/arm selection + confidence calibration
CONFIRMATION     one-time independent algorithm confirmation
```

已消费 120-frame cohort 不得进入任何角色。现有 8-session ARKitScenes R2 roster 专属于部署
confirmation，保持 metadata-only，不得用于本路线训练、calibration 或 selection。

在读取 RGB-D body/outcome 前必须完成 license、identity、ancestry、SHA、pHash/crop/mirror、
parent/session 和 label-capability 审计。任何 `SAME_CAPTURE`、`UNKNOWN`、分歧或未审边保持 HOLD。

## 9. 当前未解决 blocker

以下项目使 B0 保持 `EXECUTION_NOT_AUTHORIZED`：

1. 产品 CameraX 实际 crop/rotation/intrinsics → display/tensor K 的 runtime receipt 尚未冻结；
2. TRAIN/DEVELOPMENT/CONFIRMATION 新 roster 尚未选择，license 与 near-duplicate audit 未完成；
3. ground/clearance truth reader 的 exact implementation/hash 尚未绑定；
4. confidence threshold 与 B1 optimizer/lambda/epoch/batch 尚未由无 outcome protocol 冻结；
5. `1×3×608×448` DepthART/Assistive Geometry PyTorch shape smoke 与 export shape smoke 未完成。

## 10. 唯一 successor

```text
BLINDASSIST_ASSISTIVE_GEOMETRY_B0_INPUT_DATA_PREFLIGHT
```

该 preflight 只允许：静态代码/metadata/shape 检查、数据能力盘点、identity/license 审计设计、
合成 K 变换测试和随机权重/冻结 checkpoint shape smoke。禁止读取 candidate outcome、训练正式
student、激活 D1/R2 或修改默认 App。
