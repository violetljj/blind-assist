# DepthART-S D3R1 Phase-A result

状态：`D3R1_PHASE_A_PORTRAIT_POSE_CONTINUITY_PASS_32_IDENTITIES_LOCKED`

## 结果

exact-127 frozen pool 已全部处理，没有在达到 32 个合格身份后提前停止。254 个
`lowres_wide_intrinsics.zip` / `lowres_wide.traj` body 共 `133,734,849` bytes，全部一次下载；
GET 状态与冻结 HEAD 的 Content-Length、ETag、Last-Modified 精确一致。

全量 source-integrity 审计覆盖 127 checkpoints、254 bodies、`603,634` 个 `.pincam` payload
与 `99,155` 个 trajectory rows。独立离线 validator 重新校验 bytes/SHA、ZIP CRC/path safety、
全部 intrinsics schema、trajectory 与 portrait/pose continuity，得到与 producer 一致的
`53/127` eligible；按 frozen pool order 锁定最早的 32 个身份。

完整 selected roster 保存在 [machine result](DEPTHART_TASK_PRESERVING_D3R1_PHASE_A_RESULT_2026-08-12.json)
及本地 SHA-bound manifest 中。

## 边界

本步只读取 intrinsics/trajectory。没有读取 RGB、depth、confidence、source-derived task truth、
model output 或 R2；没有分配 TRAIN/DEVELOPMENT 角色，也没有建立性能、默认 App、production 或
safety 权限。

## 唯一 successor

`EXPLICIT_D3R1_PHASE_B_DEPTH_CONFIDENCE_SOURCE_SCOPE_REGISTRATION_FOR_EXACT_32_PHASE_A_SELECTION`

D3R1 既有 source receipt 明确不包含 Phase-B depth/confidence，因此必须先登记 exact-32 范围；
本次 PASS 不直接授权 Phase-B HEAD 或 GET。
