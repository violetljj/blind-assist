# Assistive Geometry Data Upgrade Engine

状态：`current / AG-DUE_R0_SANPO_INITIAL_SOURCE_MANIFESTS_LOCKED / NOT_PRESCREENED / NO_PAYLOAD_OR_TEACHER_AUTHORITY`

AG-DUE 是 Assistive Geometry 的数据研究并行线，不是新算法路线。它把 AG-DCA 已观测到的
right-censor、corridor、R2 factor 与 temporal 缺口变成可重放的 source-admission 合同，目标是在
下载、适配或 Teacher 生成之前回答：某个数据源是否值得进入来源特定的完整性/载荷审计、
只能保留为候选，还是应当拒绝？metadata 不能直接证明数据支持。

## 唯一 successor

`BLINDASSIST_ASSISTIVE_GEOMETRY_DUE_R0_SANPO_INITIAL_MANIFEST_STATIC_PRESCREEN_EXECUTION`

下一步只允许对已锁定的 SANPO Real/Synthetic 两份 manifest 运行 R0 static prescreen；不得联网刷新
source metadata、下载或打开 payload，也不得把预期的 `PARTIAL` 当成 source data support。任何
source-specific integrity/capability-count audit 仍需 prescreen 结果另行授权。

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

## 已锁定 SANPO manifests

- [manifest lock](BLINDASSIST_ASSISTIVE_GEOMETRY_DUE_R0_SANPO_INITIAL_SOURCE_MANIFEST_LOCK_2026-08-10.json)
  绑定 metadata bootstrap receipt、两份 manifest、validator 与 7 项 mutation test；
- [SANPO Real manifest](BLINDASSIST_ASSISTIVE_GEOMETRY_DUE_R0_SANPO_REAL_INITIAL_SOURCE_MANIFEST_2026-08-10.json)
  只锁一个 discovery-fresh official-train session；
- [SANPO Synthetic manifest](BLINDASSIST_ASSISTIVE_GEOMETRY_DUE_R0_SANPO_SYNTHETIC_INITIAL_SOURCE_MANIFEST_2026-08-10.json)
  同样只锁一个 discovery-fresh official-train session；
- [bootstrap receipt](BLINDASSIST_ASSISTIVE_GEOMETRY_DUE_R0_SANPO_METADATA_BOOTSTRAP_RECEIPT_2026-08-10.json)
  披露官方 repo/README/split metadata 访问与 deterministic last-ID selection；正式 prescreen=false。

两份 manifest 的 capability frame、orientation 与 parent count 全部为 0，quality 统一为
`CHARACTERIZED_NOT_VALIDATED`，camera/upright basis 为 `UNKNOWN`。官方发布的 video/depth/pose/
panoptic 字段只是候选 signal inventory，不是已验证 truth。

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

当前建立 gap contract、source manifest schema、静态 checker mechanics，并 byte/hash 锁定两份 SANPO
metadata-only manifest；尚未执行正式 prescreen，未读取 payload/RGB/geometry/model outcome，未调用 Teacher，未生成 pseudo-label，
未物化数据或训练模型，也不改变 R2、SANPO、QSF、CBF、FCI、默认 App、产品或 safety authority。
