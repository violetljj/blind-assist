# RCLE Phase B real positive-approach role admission R2 CID-SIMS result

日期：2026-07-27

## 终态

`INVALID_R2_EVIDENCE / INVALID`

R2 没有得到 `REAL_POSITIVE_APPROACH_ROLE_ADMITTED / VALID`。因此：

- 不创建、不授权 performance qualification；
- 不运行或读取 RCLE RGB algorithm；
- 不重试 `floor3_1`；
- 不改用 `floor3_2`、`floor3_3`、CoRBS、KITTI-360、mirror 或 repack；
- 不把本次终态改写为 source HOLD 或 algorithm negative。

## 唯一 claim 与零 payload access

唯一 exclusive claim 已于 `2026-07-27T04:49:50.743678+00:00` 由 formal
runner 使用 `O_CREAT|O_EXCL` 创建并 `fsync`。claim SHA-256：

`0fd0ae94f50b38ff5cb9b50233ac3a0bf870b21aecf05d41f734ebf814da49a2`

claim 绑定：

- candidate：`CID_SIMS_V6_FLOOR3_1`；
- official run：`floor3_1`；
- file ID：`c595882daafe788a29d687872cc1fc2a`；
- official bytes：`2,211,008,069`；
- official MD5：`585d38855ad7d04817991cdbbb72016b`；
- canonical URL：
  `https://china.scidb.cn/download?fileId=c595882daafe788a29d687872cc1fc2a`；
- source descriptor SHA-256：
  `54c2f5e207cd94b7ca8e2e6f5e795ee6dd61018b5698ae9ad4f08d48d196dc7f`；
- implementation lock SHA-256：
  `ded8e564db085872854fae3d02568d1568407a7d5d681d64d8aa0bfe37776815`。

失败发生在 acquisition 的 `_validate_preaccess_bindings`，早于 archive path
创建和唯一 GET。正式目录只包含 `claim.json`、`FAILURE.json` 和
`progress.json`；不存在 source archive、partial payload、source receipt 或 formal
geometry 输出。因此本次：

- HTTP request count：`0`；
- payload bytes：`0`；
- RGB/depth/pose/archive body reads：`0`；
- RGB algorithm outcome reads：`0`；
- retry/fallback/replacement：`0/0/0`。

## INVALID 原因

冻结 source-authority lock 使用字段：

`candidate.official_run_id = floor3_1`

冻结 acquisition 实现的 preaccess validator 要求字段：

`candidate.sequence_id`

claim 成功后，acquisition 在读取任何 payload 前因
`R2_AUTHORITY_CANDIDATE_FIELD:sequence_id` fail closed。按照预注册的
`invalid_exception_scope`，claim 后的 binding/implementation failure 属于
`INVALID_R2_EVIDENCE / INVALID`，不是普通 transport、MD5、bytes、ZIP 或
source-role 不闭合所对应的 valid HOLD。

这是一项实现/文档 schema 一致性缺陷，不是 CID-SIMS source failure，不是
positive-approach geometry negative，也不提供任何 algorithm 性能信息。

## 终态证据

- `FAILURE.json` SHA-256：
  `66b69c58538ff6676f34c36960632b40e1732f7be17a61d8f9918f6cf7c3f8b3`；
- `progress.json` SHA-256：
  `54a1bc4640584a1e6eeca4829db12cddc3f0e86a197ace72ffa62051056bdfbe`；
- terminal phase：`FAILED_NO_RETRY`；
- completed units：`1 / 4`；
- validation terminal：`INVALID`；
- formal process 已退出。

## 权限边界

本次唯一 claim 已消费。R2 永久关闭，不允许修补或重跑。由于没有任何 Floor3
payload byte 或 geometry access，本结果不声称已经触发
`CID_SIMS_V6_OFFICE_BUILDING_FLOOR3_THREE_RUN_CAPTURE_FAMILY` 的
payload-access confirmation burn；但 `floor3_1` 仍不得在 R2 中重试或替换。

如未来还要研究 CID-SIMS，必须另立新的、明确独立的协议和 claim lineage；
该新协议不能冒充 R2 retry，也不能自动获得 performance qualification 权限。
