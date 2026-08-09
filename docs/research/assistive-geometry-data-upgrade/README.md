# Assistive Geometry Data Upgrade Engine

状态：`current / AG-DUE_R0_GAP_PROTOCOL_LOCKED / NO_SOURCE_MANIFEST_LOCKED / NO_PAYLOAD_OR_TEACHER_AUTHORITY`

AG-DUE 是 Assistive Geometry 的数据研究并行线，不是新算法路线。它把 AG-DCA 已观测到的
right-censor、corridor、R2 factor 与 temporal 缺口变成可重放的 source-admission 合同，目标是在
下载、适配或 Teacher 生成之前回答：某个数据源是否值得进入来源特定的完整性/载荷审计、
只能保留为候选，还是应当拒绝？metadata 不能直接证明数据支持。

## 唯一 successor

`BLINDASSIST_ASSISTIVE_GEOMETRY_DUE_R0_INITIAL_SOURCE_MANIFEST_LOCK`

下一步只允许为 SANPO Real、SANPO Synthetic 和随后明确选定的公开候选源冻结 metadata-only
source manifest。不得在 source identity、license、ancestry、independence、native signal 与 access
receipt 被机器锁定前下载或打开 payload；更不得生成 pseudo-label 或训练模型。

## R0 合同

- [machine protocol](BLINDASSIST_ASSISTIVE_GEOMETRY_DUE_R0_GAP_DRIVEN_SOURCE_ADMISSION_PROTOCOL_2026-08-10.json)
  冻结研究问题、DCA predecessor、代码身份、权限和 successor；
- [gap contract](BLINDASSIST_ASSISTIVE_GEOMETRY_DUE_R0_GAP_CONTRACT_2026-08-10.json)
  原样继承 QSF/CBF/FCI 的已版本化 requirements，并增加 timestamp + pose 的 presence-only discovery screen；
- [source manifest schema](BLINDASSIST_ASSISTIVE_GEOMETRY_DUE_R0_SOURCE_MANIFEST_SCHEMA_2026-08-10.json)
  冻结 license、ancestry、parent independence、metadata-only access、capability counts、provenance、
  quality 和 upgrade path；
- [validator module](../../../scripts/research/assistive_geometry_data_upgrade/README.md)
  计算 joint-parent 与 orientation gate，输出 `PRESCREEN_ADMIT / PARTIAL / REJECT`；即使
  `PRESCREEN_ADMIT`，`source_data_support_established` 仍固定为 false。

## 证据分层

`source-native` 只说明来源，不自动等于 GT。直接满足 capability gate 需要同时满足：

1. provenance 为 source-native annotation、source-native sensor 或 deterministic derived；
2. 对具体 claim 的 quality status 已是 `VALIDATED_FOR_CLAIM`；
3. frame、orientation、每-parent 与 joint-parent 数量全部达门；
4. `UNKNOWN` 从未被当作 negative。

deterministic derivation 可以标记为 upgradeable，但在升级协议执行前仍只产生 `PARTIAL`。
multi-Teacher consensus、single Teacher、heuristic 与 VLM proposal 只能用于 candidate/mining，不能直接
形成 task truth。公开可下载也不等于允许重分发、商业使用、Confirmation 或安全主张。

R2 F1 当前还存在与来源无关的 `FactorTensorAdapter` ABI blocker；因此即使未来来源通过载荷审计，
也只能继续满足 source/label 前门，不能绕过 adapter protocol/schema/mutation canary lock。

## Claim ceiling

当前只建立 gap contract、source manifest schema 与静态 checker mechanics。尚无真实 source manifest
被锁定或执行，未读取 payload/RGB/geometry/model outcome，未调用 Teacher，未生成 pseudo-label，
未物化数据或训练模型，也不改变 R2、SANPO、QSF、CBF、FCI、默认 App、产品或 safety authority。
