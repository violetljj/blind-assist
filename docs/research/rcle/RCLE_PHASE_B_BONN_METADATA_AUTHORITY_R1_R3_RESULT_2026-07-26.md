# RCLE Phase B Bonn Metadata Authority R1–R3 结果

状态：`CANONICAL_METADATA_AUTHORITY_R3_PASS_FORMAL_PHASE_B_B0_READY`

日期：2026-07-26

## 结论

固定的 Bonn metadata cohort 已获得 canonical execution authority。R3 在任何
project module、lock、receipt、metadata 或 control read 之前，先对唯一
canonical `run_claim.json` 做内联 exclusive create 并 fsync；之后才延迟导入
受 implementation lock 约束的项目实现。唯一正式运行通过，validate-only
独立复算为 `VALID`。

- official metadata denominator：`26`
- historical exclusion：`9`
- selected cohort：`6`
- cohort identity：
  `513b770d18489fd0caf84874e9fb89456eb3a992fc262b037220b66b5caae86e`
- R3 design lock：
  `e033cbc3e6a4eca6e27f7ea65fe0397aabb64922fbaa06a5dac0a6634755a7dc`
- R3 implementation lock：
  `25058156e0af22ba9fdfea7962065c5731e583d1587b45546155bed9d2bdd2b6`
- run claim：
  `88d1f53ba5f259e2df7ecbeef83de3ce23de936436806f5fd852b1bcee6c2667`
- receipt：
  `05a283b84f62bee000447bb567eadd63b424afaa9d81f5f0d83d36a9ed02489b`

Canonical receipt 位于
`artifacts.local/evidence/rcle_phase_b_bonn_entry_r3/authority_gate_r3/receipt.json`；
生成 evidence 保持 ignored，不进入 Git。

## R1 / R2 负结果

R1 与 R2 均保留为 diagnostic history，不提供 execution authority：

1. R1 在 claim 前调用 implementation-lock validation，因而先读取了 control、
   R0 receipt 与 official metadata；preclaim failure 可被重试。
2. R2 把 claim 移到 receipt build 前，但 runner 顶层 project import 与
   `Path.resolve()` 仍发生在 claim 前。
3. R3 只允许解释器读取 hash-bound bootstrap runner 与 Python runtime/stdlib；
   runner preclaim 无 project import、argparse、resolve/stat/exists/glob/listdir、
   metadata/control/network read。

因此 R1 receipt
`c2efac24585890f83fe9311e2d0bd6fd6155746a4c585934e5ad42fb27e9ed92`
与 R2 receipt
`9ea6915e84c6d93faeaab59f4705bcd8dc04263c38a382a067ea96ceaed52912`
只用于审计历史。

## Firewall

本门没有读取 archive payload、RGB/depth、pose map、legacy outcome，也没有计算
任何 Phase B metric。它只授权独立冻结的 B0 acquisition/timestamp-inventory
入口进入执行准备；不授权 B0 之外的 decode、metric、Replay、Android、人体、
安全或生产工作。
