# BlindAssist Assistive Geometry Corridor Bottleneck Field

状态：`current / WILD_LAB / R0_CLOSED_DATA_SUPPORT_NOT_EVALUABLE / ORACLE_NEVER_AUTHORIZED / MODEL_AND_TRAINING_NEVER_AUTHORIZED / DEFAULT_APP_UNCHANGED`

AG-CBF 的科学问题是：ground-aligned、body-profile inflated、拓扑连通的 2.5D corridor
bottleneck 表示，是否比固定 left/center/right 三带摘要保留更多与身体通行空间有关的信息？

本路线严格按 `DATA SUPPORT → ORACLE CEILING → REPRESENTATION VALUE → TRAIN` 推进。当前只允许
第一步；不先造模型，也不从现有 H3 合成 widest-path mechanics 继承真实数据有效性。

## 唯一 successor

无。重开必须另立 pre-outcome source-geometry/target contract 与路线版本，从 DATA SUPPORT 重新开始。

## 终态

无 successor。R0 已在第一道门关闭：

> `AG_CBF_R0_DATA_SUPPORT_NOT_EVALUABLE_ROUTE_CLOSE`

[冻结协议](BLINDASSIST_ASSISTIVE_GEOMETRY_CBF_R0_DATA_SUPPORT_AUDIT_PROTOCOL_2026-08-09.json)
下的 1,024 帧审计只有 44 帧 evaluable，portrait/landscape 为 `36/8`，0/16 parent 达到
`32/64`。详见 [governed result](BLINDASSIST_ASSISTIVE_GEOMETRY_CBF_R0_DATA_SUPPORT_AUDIT_RESULT_2026-08-09.md)
与 [machine result](BLINDASSIST_ASSISTIVE_GEOMETRY_CBF_R0_DATA_SUPPORT_AUDIT_RESULT_2026-08-09.json)。
因此 oracle、三带对照、模型和训练从未获得授权。

## 冻结边界

- 输入只允许 B1 TRAIN target manifest `A6F809C7...A7C2`，每 parent source-order-even 64 帧；
- 只读 `depth_m_source`、`depth_valid_source`、`ground_probability_source`、
  `ground_label_valid_source`、`intrinsics_source`、`up_camera`、`camera_height_m` 与
  `ground_plane_valid`；
- 不读 RGB、feature、checkpoint、model outcome、A0 consumed Development Selection 或 Confirmation；
- `UNKNOWN` 保持 UNKNOWN，不能充当 free/occupied/negative；
- body half-width `0.32 m` + margin `0.10 m` 只是后续 oracle 的预声明 profile，本审计不计算 inflation；
- H2/PCF 是未来独立候选，不是本 R0 的隐含 fallback。

## Claim ceiling

当前只证明协议/抽样/完整性 mechanics 与数据支撑负终态。没有证明或反证真实 oracle/CBF 数学，
也未证明 CBF 表示优于三带、模型可学、论文 novelty、跨数据泛化、设备可运行、产品可用或助盲安全。

## 输出所有权

只允许写入 `artifacts.local/evidence/assistive-geometry-cbf/`。任何后续 work/model 路径必须由新协议
显式授权，不能写入 Assistive Geometry、AG-QSF 或其他路线的 artifact root。
