# Assistive Geometry Data Upgrade Engine

状态：`current / AG-DUE_R1_SANPO_SYNTHETIC_PREFLIGHT_NOT_EVALUABLE / EXACT_METADATA_OBJECT_MISSING / ROUTE_CLOSED`

AG-DUE 是 Assistive Geometry 的数据研究并行线，不是新算法路线。它把 AG-DCA 已观测到的
right-censor、corridor、R2 factor 与 temporal 缺口变成可重放的 source-admission 合同，目标是在
下载、适配或 Teacher 生成之前回答：某个数据源是否值得进入来源特定的完整性/载荷审计、
只能保留为候选，还是应当拒绝？metadata 不能直接证明数据支持。

## 当前终态

`NONE_STOP_AT_PREFLIGHT_TERMINAL`

exact Synthetic session 的 metadata/object-inventory preflight 已执行并以 `NOT_EVALUABLE` 停止：
冻结的 annotation-type object 返回 `404`。协议禁止猜测替代路径、换 session/camera/lens 或扩大 LIST，
因此三类 frame prefix 未开始 LIST，最低 25 个 numeric-index intersection 未产生，frame-body canary
不具备 protocol-lock 资格。Real 仅保留为 R0 `PARTIAL` 候选，没有活动执行 successor。

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

## R0 static prescreen 结果

- [governed result](BLINDASSIST_ASSISTIVE_GEOMETRY_DUE_R0_SANPO_INITIAL_STATIC_PRESCREEN_RESULT_2026-08-10.json)
  由专用 exact-path validator 从冻结 manifest 重新计算；
- SANPO Real：`PARTIAL`，hard rejection 为空，完整 screening match 为 0；
- SANPO Synthetic：`PARTIAL`，hard rejection 为空，完整 screening match 为 0；
- 两者只有 R2 F1 supervision 与 temporal presence 形成 relevant partial，只有 R2 F1 标为
  upgradeable；QSF right-censor、corridor 与 FCI truth bundle 均未出现；
- `source_data_support_established=false`、`supported_for_protocol_lock=false`、
  `execution_authorized=false` 对两者均保持不变。

Synthetic 被选作下一份窄审计协议对象，只因为其锁定 inventory 已把 metric depth 与 panoptic
列为 source-native candidate，而 Real 的 depth 尚不能表述为 oracle source-native truth。这是
Discovery 审计顺序，不是 source admission、模型选择或数据质量排名。

## R1 Synthetic source-specific audit lock

- [R1 machine protocol](BLINDASSIST_ASSISTIVE_GEOMETRY_DUE_R1_SANPO_SYNTHETIC_SOURCE_SPECIFIC_INTEGRITY_AND_CAPABILITY_AUDIT_PROTOCOL_LOCK_2026-08-10.json)
  只绑定 official TRAIN session `17c7d6bc...179cb`、`camera_chest/left` 与 exact GCS object paths；
- 当前只完成 protocol/static validation，network、remote object access 与 frame body 均为 false；
- 下一 preflight 只能记录 object name/generation/size/hash、metadata schema、相机参数和 RGB/mask/depth
  numeric index inventory，并在读取 frame body 前冻结最低 25 个完整 aligned index；不足 25 个即
  `NOT_EVALUABLE`，不得换 session 或按内容挑帧。inventory count 不是 capability truth count；
- pose table 只允许审计 header 和 row count。row order/coverage 不等于 frame binding，quaternion order、
  handedness、transform direction 与 coordinate receipt 均保持 unresolved；
- metric-depth inventory 最多成为 `SOURCE_OBJECTS_PRESENT_NOT_VALIDATED_FOR_CLAIM`；还需 body canary
  验证单位、invalid/finite policy、分辨率和 RGB registration；
- panoptic inventory 最多成为 `PANOPTIC_OBJECTS_PRESENT_DERIVATION_NOT_RUN`；未冻结 label taxonomy、
  void/UNKNOWN、connectivity 与 boundary derivation 前，不是连续 obstacle-boundary truth；
- support factor 明确 `ABSENT`。depth、pose 或 semantic ground 都不能自动升级为 support truth；
- 未来 frame-body canary 仍需独立 protocol lock，最多且必须读取上述 25 个 depth + mask object；
  RGB 只允许 object metadata，不允许 body/visual access；
- roster 只有 1 parent，而 R2 F1 要求 12 joint parents，因此即使本 session 审计全过，仍不能建立
  F1 source support 或执行权限。

## R1 metadata/object-inventory preflight 结果

- [execution lock](BLINDASSIST_ASSISTIVE_GEOMETRY_DUE_R1_SANPO_SYNTHETIC_METADATA_AND_OBJECT_INVENTORY_PREFLIGHT_EXECUTION_LOCK_2026-08-10.json)
  固定 `gresearch` bucket/host、四个 metadata object、三个 frame prefix、请求/字节/分页预算、
  metadata receipt 与 frame provider-only receipt 的分层，以及 frame-body byte budget=0；
- [governed result](BLINDASSIST_ASSISTIVE_GEOMETRY_DUE_R1_SANPO_SYNTHETIC_METADATA_AND_OBJECT_INVENTORY_PREFLIGHT_RESULT_2026-08-10.json)
  确定为 `NOT_EVALUABLE / STOP_SOURCE_OBJECT_INVENTORY_OR_SCHEMA_INCOMPLETE`；
- description 与 global labelmap body 只在内存读取，没有持久化 raw bytes；exact annotation-type object
  两个 attempt 均在 HEAD 返回 `404`，pose table 未读取，frame prefix LIST 未开始；
- 两次 attempt 共 12 个实际 network request。首次 retry receipt 曾把 prior request count 记为 2，
  governed result 按 hash-locked 顺序控制流更正为每 attempt 6、总计 12，并显式保留该 receipt 字段错误；
- artifact root 保存两个 attempt receipt、fail-closed inventory/schema receipt 与 final result；四类
  inventory/capability count、selected-lowest-25 全为 0，frame-body request/read/bytes 全为 0；
- body-canary execution、source support、DCA/F1 PASS、pose/timestamp/support/boundary truth 与训练权限
  全为 false。本 exact R1 route 已关闭，不能通过 fallback 或 path guessing 救援。

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

当前建立 gap contract、source manifest schema、静态 checker mechanics，完成 `PARTIAL/PARTIAL`
prescreen、R1 source-specific audit lock 与一次 exact metadata preflight。preflight 已联网读取 description/
labelmap metadata body，但未持久化 raw bytes；annotation-type HEAD 为 `404` 后 fail-close。没有 LIST 或读取
RGB/mask/depth frame body，没有读取 pose table，没有调用 Teacher、生成 pseudo-label、物化数据或训练模型，
也不改变 R2、SANPO、QSF、CBF、FCI、默认 App、产品或 safety authority。若要引入新 source/session/path，
必须从另行版本化的 R0 manifest 与 source-specific protocol 重新准入；当前没有活动 successor。
