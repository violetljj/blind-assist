# BlindAssist Assistive Geometry Corridor Bottleneck Field

状态：`current / WILD_LAB / R0_DATA_SUPPORT_AUDIT_LOCKED_NOT_RUN / ORACLE_NOT_AUTHORIZED / MODEL_AND_TRAINING_NOT_AUTHORIZED / DEFAULT_APP_UNCHANGED`

AG-CBF 的科学问题是：ground-aligned、body-profile inflated、拓扑连通的 2.5D corridor
bottleneck 表示，是否比固定 left/center/right 三带摘要保留更多与身体通行空间有关的信息？

本路线严格按 `DATA SUPPORT → ORACLE CEILING → REPRESENTATION VALUE → TRAIN` 推进。当前只允许
第一步；不先造模型，也不从现有 H3 合成 widest-path mechanics 继承真实数据有效性。

## 唯一 successor

`BLINDASSIST_ASSISTIVE_GEOMETRY_CBF_R0_TRAIN_GRID_DATA_SUPPORT_AUDIT_EXECUTION`

执行唯一冻结的 [machine protocol](BLINDASSIST_ASSISTIVE_GEOMETRY_CBF_R0_DATA_SUPPORT_AUDIT_PROTOCOL_2026-08-09.json)，
仅检查固定 16-parent TRAIN target 中 source geometry 对 `32×31`、forward `0.2–5.0 m`、
lateral `-2.0–2.0 m` 网格的观测与 ground 支撑。若未过 gate，R0 直接关闭；若通过，才可另立
oracle/representation-value lock，且仍不授权模型或训练。

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

当前最多证明协议、抽样、source geometry 完整性和网格支撑审计 mechanics 被锁定。尚未证明真实
oracle 可评价、CBF 表示优于三带、模型可学、论文 novelty、跨数据泛化、设备可运行、产品可用或助盲安全。

## 输出所有权

只允许写入 `artifacts.local/evidence/assistive-geometry-cbf/`。任何后续 work/model 路径必须由新协议
显式授权，不能写入 Assistive Geometry、AG-QSF 或其他路线的 artifact root。
