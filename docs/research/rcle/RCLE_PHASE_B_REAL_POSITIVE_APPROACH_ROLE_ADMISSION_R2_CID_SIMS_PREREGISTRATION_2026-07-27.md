# RCLE Phase B real positive-approach role admission R2 CID-SIMS 预注册

日期：2026-07-27

状态：

`PREREGISTERED / SOURCE_AUTHORITY_LOCKED / CANDIDATE_PAYLOAD_NOT_ACCESSED`

本文件只完成 pre-access freeze。未创建 implementation lock、exclusive claim 或
formal run；未访问网络、candidate payload、geometry、RGB pixels、RCLE RGB
algorithm outcome 或 performance outcome。

## 唯一范围与终态

任务 ID：

`RCLE_REAL_POSITIVE_APPROACH_ROLE_ADMISSION_R2_CID_SIMS`

R2 只判断一个真实数据源是否可承担 positive-approach 数据角色。它不实现、不导入、
不运行 RCLE RGB algorithm，也不建立或运行 performance qualification。

合法终态只有：

- `REAL_POSITIVE_APPROACH_ROLE_ADMITTED / VALID`
- `HOLD_ALGORITHM_CANARY_APPROACH_ROLE_INCOMPLETE / VALID`
- `INVALID_R2_EVIDENCE / INVALID`

只有第一种终态允许另立 RGB algorithm implementation 或 performance
qualification；R2 自身无论终态如何都不运行这些后继。

机器合同：
[RCLE_PHASE_B_REAL_POSITIVE_APPROACH_ROLE_ADMISSION_R2_CID_SIMS_CONTRACT_2026-07-27.json](RCLE_PHASE_B_REAL_POSITIVE_APPROACH_ROLE_ADMISSION_R2_CID_SIMS_CONTRACT_2026-07-27.json)

合同 SHA-256：

`52ceebbe7727952c0fb963dfc855ae9115bfa3a5b2187649665dac49f4c5f6b4`

## 结果盲唯一选择

Discovery R0 的 candidate lock 固定了唯一候选数据族和成员全集：

`CID-SIMS V6 / office_building / floor3 / {floor3_1, floor3_2, floor3_3}`

其 SHA-256 为：

`683ee8486d19a2ef3818a9c3b397166ef14ee73125dd791801c1053f02002728`

R2 对这三个 exact official run ID 使用升序 Unicode scalar/code-point
lexicographic order。三个字符串都是 ASCII，因此该顺序与 unsigned ASCII byte
lexicographic order 完全相同。禁止 Unicode normalization、大小写折叠、数字后缀
解析或依赖列表排列。唯一结果为：

`floor3_1 < floor3_2 < floor3_3`

因此唯一候选冻结为：

| 字段 | 冻结值 |
| --- | --- |
| official run ID | `floor3_1` |
| ScienceDB file ID | `c595882daafe788a29d687872cc1fc2a` |
| official MD5 | `585d38855ad7d04817991cdbbb72016b` |
| exact bytes | `2,211,008,069` |
| canonical URL | `https://china.scidb.cn/download?fileId=c595882daafe788a29d687872cc1fc2a` |

`floor3_2`、`floor3_3`、CoRBS、KITTI-360、mirror、repack 或其他 archive 均不是
失败后的替补。

source authority/candidate lock：
[RCLE_PHASE_B_REAL_POSITIVE_APPROACH_ROLE_ADMISSION_R2_CID_SIMS_SOURCE_AUTHORITY_AND_CANDIDATE_LOCK_2026-07-27.json](RCLE_PHASE_B_REAL_POSITIVE_APPROACH_ROLE_ADMISSION_R2_CID_SIMS_SOURCE_AUTHORITY_AND_CANDIDATE_LOCK_2026-07-27.json)

SHA-256：

`49fdf51620aeb5b0c06fe7ce5c8d0944d78768c958153644f9ba64ebb4119659`

source descriptor SHA-256：

`54c2f5e207cd94b7ca8e2e6f5e795ee6dd61018b5698ae9ad4f08d48d196dc7f`

## Burned、独立性与 confirmation isolation

burned/exclusion manifest：
[RCLE_PHASE_B_REAL_POSITIVE_APPROACH_ROLE_ADMISSION_R2_CID_SIMS_BURNED_AND_EXCLUSION_MANIFEST_2026-07-27.json](RCLE_PHASE_B_REAL_POSITIVE_APPROACH_ROLE_ADMISSION_R2_CID_SIMS_BURNED_AND_EXCLUSION_MANIFEST_2026-07-27.json)

SHA-256：

`0ce9494307c3a872edc8bb7aa00aa061965165cae27a6a6bf34fb47f28c74a26`

ancestry 冻结为：

`SCIENCEDB_OFFICIAL -> CID_SIMS_V6 -> CID_SIMS_V6_OFFICE_BUILDING ->
CID_SIMS_V6_OFFICE_BUILDING_FLOOR3 ->
CID_SIMS_V6_OFFICE_BUILDING_FLOOR3_1 -> R2`

independence group：

`CID_SIMS_V6_OFFICE_BUILDING_FLOOR3_THREE_RUN_CAPTURE_FAMILY`

任何 `floor3_1` payload byte 或 geometry access 都会永久烧掉完整 Floor3 family：
`floor3_1/2/3`、共享 pointcloud、所有 archive variant 与 derivative 均不得用于本
claim 的 future confirmation。访问后只可保留为本次 role admission、
source characterization、counterexample 或 regression。

## Claim 前后的硬顺序

1. 只验证 Discovery lock、burned manifest、source-authority lock、contract 及冻结 SHA；
2. 在 claim 前另行冻结 minimal bootstrap、acquisition、source-role resolver、
   geometry producer、独立 validator 与测试的 implementation lock；
3. frozen bootstrap 只读取具名 control documents 和 implementation files，
   随后用 `O_CREAT|O_EXCL` 建立并 `fsync` 唯一 claim，绑定全部文档、
   source descriptor 和 implementation SHA；claim 前禁止对 candidate payload
   path/URL 执行 `resolve/stat/exists/glob/listdir/open/create/request`；
4. 只有 claim 成功后，才允许触及 canonical URL 或任何 candidate payload path；
5. 只允许 canonical URL 的一次完整 GET，requested URL 必须等于 final URL；
   禁止 redirect、HEAD、range、retry、resume、fallback、mirror、repack 或换源；
6. 流式写入时同时计算 byte count、官方 MD5 与本地 SHA-256；只有 bytes 和 MD5
   同时精确匹配，才允许打开 ZIP；
7. geometry 前冻结 exact normalized member name/size/CRC inventory 和 source-role
   mapping。

传输、HTTP、partial body、bytes/MD5、ZIP 可读性或冻结 source-role resolution
不闭合时如实 HOLD；禁止访问、binding drift、危险/重复/加密路径、多个 candidate
root、control/index inconsistency、实现或独立验证失败属于
`INVALID_R2_EVIDENCE / INVALID`。claim 一经创建，无论 HOLD、INVALID、中断或成功均
永久保留，不重试、不续传、不替换。

## 唯一窗口与冻结门

只评估 `floor3_1` 首个共同 RGB-D/pose timestamp 开始的第一个完整半开
`10.000 s` 窗。只读 source-native calibration、camera-to-world pose、RGB/depth
index 与该窗 depth；禁止解码 RGB。

门与 R0/R1 保持一致：

- candidate-pair coverage `>= 0.80`
- evaluable pair `>= 8`
- window median signed radial expansion `>= 0.05 s^-1`
- window median positive fraction `>= 0.75`

任何门失败即 HOLD。禁止第二窗、滑窗、best-window rescue、重心化、翻转 pose
方向、调整分母、降低阈值或查看 RGB。

producer 与 validator 必须分别从 source-native pose/depth 重算；validator 不得
import producer。identity、official MD5/bytes、pair、window、gate、ancestry、
reuse、forbidden access、retry/fallback/replacement count 必须逐项核对。
