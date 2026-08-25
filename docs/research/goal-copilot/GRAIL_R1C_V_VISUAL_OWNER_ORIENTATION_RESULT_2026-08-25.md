# GRAIL-R1C-V Visual Owner Orientation Result

日期：2026-08-25（Asia/Hong_Kong）

状态：`DEVELOPMENT / DETERMINISTIC_RGB_PROPOSAL_PROBE / FINAL_SLOT_AGREEMENT_39_OF_78 / REFERENT_38_OF_78 / COMPLETE_27_OF_78 / WRONG_TARGET_9_OF_43 / ABSENCE_0_OF_78 / AXIS_FRONT_DOOR_FAILED / SIGN_ALSO_FAILED / CURRENT_ESTIMATOR_CLOSED / NO_SUCCESSOR_AUTHORIZED / FORMAL_TEST_UNOPENED / STOP_BEFORE_M2 / DEFAULT_APP_UNCHANGED`

## 结果前冻结与输入防火墙

R1C-V 先以独立提交 `cae276ac5d69df365265fb7dab1c4c53fff11017` 冻结[协议](GRAIL_R1C_V_VISUAL_OWNER_ORIENTATION_PROTOCOL_2026-08-25.md)，再以 `1f31818ea5f66d5487c3c23612410ec1bc93353c` 固化实现，之后才运行唯一一次 probe。结果暴露后未改变 `10%` support padding、PCA axis、gradient first moment、sign threshold=`0.05`、UNKNOWN、3×3 slot 或下游。

Prediction-only 函数只读取 full-scene RGB、独立视角内 frozen predicted proposal group、bbox 与 semantic type。native position/yaw/coordinate、camera/world pose、object ID、sample order、reference-query joint alignment 与 evaluator truth 均不进入 predictor。evaluator-only native supplement 对 6/6 houses 的 79 个 query/reference required objects 全部可评估，仅在视觉预测持久化后用于 axis/sign 归因。

## 方向可观测性

| 指标 | 结果 |
|---|---:|
| Axis evaluable pair | 29/78 |
| Axis pair within 20° | 13/29 |
| Undirected axis error median / mean | 4.58° / 29.09° |
| Sign evaluable pair | 16/78 |
| Sign pair correct | evaluable | 4/16 = 25.0% |
| 至少一侧 sign UNKNOWN | 45/78 rows |
| Sign UNKNOWN target views | 53/156 |

axis error 呈明显长尾：中位数较小不能覆盖仅 `29/78` 的 pair-evaluable coverage 与 `13/29` 的双侧 20° 命中。sign 更直接失败：仅 16 个 pair 两侧均可给 sign，其中只有 4 个双侧方向正确；UNKNOWN 按协议保留在 78-case 分母，未用 camera direction 补值。

## 三 arm 与端到端

| Arm | Cross-view slot | Referent | Complete | Wrong-target | Absence | R1B 23-case rescue |
|---|---:|---:|---:|---:|---:|---:|
| Axis-only diagnostic（estimated axis + oracle sign） | 45/78 | 45/78 | 32/78 | 13/43 | 0/78 | 3/23 |
| Sign-only diagnostic（oracle axis + estimated sign） | 40/78 | 40/78 | 29/78 | 11/43 | 0/78 | 2/23 |
| **R1C-V final（estimated + estimated）** | **39/78** | **38/78** | **27/78** | **9/43** | **0/78** | **1/23** |

三 arm permutation 均为 `156/156`。final selector collateral=`11`、complete collateral=`5`；相对 R1C-O uplift recovery 为 referent=`-19.4%`、complete=`13.9%`。absence=`0` 主要来自 UNKNOWN/无 exact match 后的 abstention，不能抵消 referent、wrong-target 与 collateral 失败。

final 的 40 个 referent failures 以结果中持久化的 per-record diagnostics 做互斥归因：

| Failure class | Count |
|---|---:|
| `OWNER_GROUP_OR_AXIS_UNKNOWN` | 12 |
| `AXIS_ERROR` | 10 |
| `SIGN_UNKNOWN` | 11 |
| `SIGN_FLIP` | 5 |
| `SLOT_COLLISION_OR_TIEBREAK` | 2 |

其中 Drawer=`34`、Doorway=`6`。即使使用 oracle sign，estimated-axis arm 的 cross-view slot 也只有 `45/78`，因此按预注册顺序先判 axis front door 失败；oracle axis 下 sign arm 仍只有 `40/78`，说明 sign 同时没有被该固定 RGB-gradient rule 获得。

## 裁决

```text
GRAIL_R1C_V_AXIS_NOT_VISUALLY_OBTAINABLE_BY_DETERMINISTIC_PROBE_STOP
```

R1C-O 的 owner-centric canonical relation ceiling 保持成立；本结果只关闭当前 deterministic `proposal PCA axis + RGB gradient sign` estimator。它不否证 learned orientation、独立 depth/geometry evidence 或其他新信息源，但这些均未获授权。不得在已消费 artifact 上调 PCA、padding、sign threshold、UNKNOWN、bin、matcher、fusion、pose head，也不得选择 diagnostic arm 作为系统。

当前没有自动 successor。若未来继续，必须先改变 orientation 的独立信息源并另立结果后协议；formal test、M2、Android/default-App 继续关闭。

## Evidence identity 与 claim ceiling

- result artifact SHA-256：`6420cc705ca67bbe40fea6d9130f2c4deae74079a0cd8cdf7f23d763828d35db`
- evaluator-native oracle SHA-256：`2eb029872e61c6a505fc1244e7ef4deb4be56983e3626e60577ab5b06a94b5f7`
- schema：`blindassist_grail_r1c_v_visual_owner_orientation_probe_v1`

本结果仅为 `PROJECT_CONSUMED_DEVELOPMENT_DETERMINISTIC_RGB_PROPOSAL_ORIENTATION_PROBE` synthetic ProcTHOR/AI2-THOR evidence，不建立自然 RGB、学习、formal generalization、设备、产品或安全 authority。

