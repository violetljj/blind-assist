# R3 rotation-leakage source-localization contract preflight R0

## 结论

`CONTRACT_PREFLIGHT_VALID / EXECUTION_NOT_ACTIVATED`。

Stage B 的终态保持为 `B_ORACLE_NOT_EVALUABLE`，原始 Stage B 不重跑、不替换
identity，也不修改 R3。本轮只冻结一个观测型、消元式 successor preflight，用于
未来在 8 个 sealed rotation-only clusters 内报告 residual leakage 最早在哪一层
变得可见。独立 validator 为 `10/10 PASS`；当前 activation decision 为
`HOLD_ROTATION_LEAKAGE_LOCALIZATION_EXECUTION_PENDING_SEPARATE_ACTIVATION`。

本轮没有读取或生成新的 Stage B response，没有运行 source-localization
workload，也没有消费正式 `480+16`。

## 冻结的 estimand 与分析单位

主 estimand 是：对每个 sealed cluster，在最终 unchanged R3 absolute leakage
仍高于 `0.01/s` 的前提下，找出第一个违反本层冻结 gate、且所有必要前层均通过的
pipeline layer。

- 分析单位固定为 `cluster`，共 8 个；每个 cluster 的 601 pairs 是有序纵向重复，
  不是 601 个独立样本。
- signed response 与 absolute leakage 分开存储、分开聚合。
- pair signed 为 common evaluable cells 的 signed median；pair absolute 为
  `median(abs(cell expansion))`。禁止用 `abs(median(signed))` 代替 leakage。
- cluster absolute leakage 为 evaluable pair-absolute 的 Hyndman–Fan type-7
  P90；固定 denominator 仍为 601。
- 不做 pair-level 显著性检验、cluster pooling 或多数投票。

## 冻结的 identities、坐标和实现边界

identity/input lock 只绑定 Stage B 的 8 个
`EGO_ROTATION_STATIC_SCENE__CLEAN` sequences，逐 cluster 绑定：

- scene-geometry SHA；
- pose SHA；
- render-input SHA；
- 602 frames / 601 pairs；
- Stage B identity lock、geometry manifest 和 closeout receipt。

坐标固定为 360×640、左上 pixel center、`+x right / +y down`；pose 为
`world_from_camera`，translation 用 metre，homography 为
`K @ (R_current.T @ R_previous) @ inv(K)`。R3 tracks 保持 float32，独立
geometry/audit 重算用 float64。image warp 保持 `INTER_LINEAR`，mask warp
保持 `INTER_NEAREST`，二者均用 `BORDER_CONSTANT=0`。R3 pair core、
rotation warp、Sparse LK、support manager、local affine 和参数文件全部按 SHA
冻结。

## 分层观察与路由

顺序固定为：

1. `INPUT_GEOMETRY`：source-known material/pose projection、rotation-aligned
   coordinate residual 和 source-known cell expansion；
2. `ROTATION_WARP`：inverse-map round trip、interior normalized photometric
   residual 和 overlap；
3. `MASK_BOUNDARY`：由 unchanged 21×21 LK window 固定出的 10 px boundary
   ring，与 interior 分层比较 mask/photometric/track attrition；
4. `SPARSE_LK_AND_TRACK_FILTERING`：requested、forward、FB、mask-accepted、
   carried、supplemented tracks 及相对 source-known endpoint 的误差；
5. `LOCAL_AFFINE_AND_FINAL_AGGREGATION`：support、RANSAC、hull、condition、
   residual、coefficients、cell expansion、common cells、pair/cluster reduction。

未来每个 cluster 只能得到下列之一：

- `LEAKAGE_ALREADY_PRESENT_IN_INPUT_GEOMETRY`
- `LEAKAGE_FIRST_VISIBLE_AT_WARP`
- `LEAKAGE_FIRST_VISIBLE_AT_MASK_BOUNDARY`
- `LEAKAGE_FIRST_VISIBLE_AT_FLOW`
- `LEAKAGE_FIRST_VISIBLE_AT_LOCAL_FIT`
- `MULTIPLE_SOURCES_NOT_SEPARABLE`
- `NOT_EVALUABLE`

“first visible” 是冻结审计顺序下的可观察位置，不是因果识别。若多个 native
primitives 不能唯一分开来源，必须返回 `MULTIPLE_SOURCES_NOT_SEPARABLE`；任一
输入、support、coverage 或必需 primitive 缺失则返回 `NOT_EVALUABLE`。

## coverage、停止规则与资源门

mask、flow、managed support、RANSAC/local-fit 和 final pair coverage 分开记账。
coverage 只能导致 abstention，不能修复、补零、替换或事后 pooling。

successor launch/refill memory gate 固定为 6 GiB；所有 worker 已在飞时的 emergency
floor 保持 4 GiB，worker 数固定为 4。发生 binding/input hash drift、内存门失败、
持续 paging、worker residue、nonfinite、输出冲突或 required coverage failure 时
fail-closed。未来最多一次完整执行，不允许 retry、replacement、reseed 或
post-response rule change。

## 独立验证与 execution authority

独立 validator 交叉验证了 predecessor terminal、8 个 rotation-only identities、
geometry/input hashes、R3 source bindings、坐标/单位/warp/numeric contract、
`0.01/s` boundary、signed/absolute aggregation、coverage、6 GiB gate 和 firewall，
共 `10/10 PASS`。

当前明确没有：

- localization execution authority；
- Stage B rerun authority；
- R3 修改或新 rotation compensation authority；
- single-variable repair authority；
- C、D、Android 或正式 `480+16` authority。

只有另行签署 execution activation 后，才可运行这一个 localization workload。
其后即使定位成功，也只能另立一个 single-variable candidate contract；必须再做
独立 calibration 和新的 Stage B successor，不能回头重跑原始 Stage B。

## 冻结文件

- [contract](RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_QMS_R1_R3_ROTATION_LEAKAGE_SOURCE_LOCALIZATION_CONTRACT_PREFLIGHT_R0_CONTRACT_2026-07-29.json)
- [identity/input lock](RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_QMS_R1_R3_ROTATION_LEAKAGE_SOURCE_LOCALIZATION_CONTRACT_PREFLIGHT_R0_IDENTITY_INPUT_LOCK_2026-07-29.json)
- [independent receipt](RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_QMS_R1_R3_ROTATION_LEAKAGE_SOURCE_LOCALIZATION_CONTRACT_PREFLIGHT_R0_INDEPENDENT_RECEIPT_2026-07-29.json)
- [execution activation decision](RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_QMS_R1_R3_ROTATION_LEAKAGE_SOURCE_LOCALIZATION_CONTRACT_PREFLIGHT_R0_EXECUTION_ACTIVATION_DECISION_2026-07-29.json)
