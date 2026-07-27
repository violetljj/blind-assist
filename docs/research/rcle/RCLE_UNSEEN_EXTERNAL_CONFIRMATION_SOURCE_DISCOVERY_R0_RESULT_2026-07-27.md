# RCLE 未见数据外部确认：来源发现 R0 结果

- 日期：2026-07-27
- 协议：`RCLE_UNSEEN_EXTERNAL_CONFIRMATION_SOURCE_DISCOVERY_R0`
- 执行状态：`VALID_METADATA_CHARACTERIZATION`
- 允许声明：`DATA_CHARACTERIZED / CANDIDATE_NOT_FOUND`
- 当前处置：`EXTERNAL_COHORT_NOT_EVALUABLE`
- RGB 算法 outcome：`NOT_ACCESSED`
- 候选 payload / geometry 数值：`NOT_ACCESSED`

## 结论

本轮不能签发候选锁，也不能开始 payload 下载。F0 来源发现 evidence version 已按冻结
stop condition 关闭为 `EXTERNAL_COHORT_NOT_EVALUABLE`。

四个新来源 family 均具备真实 RGB-D 与 6DoF pose 的公开证据，但没有两个来源同时闭合：
精确 capture 身份、至少 30 秒的可验证时长、RGB/depth/pose 时间绑定、许可，以及可复核的
payload bytes/hash。此处只成立 `DATA_CHARACTERIZED`，不成立 `CANDIDATE_FOUND`。

这不是算法失败，也不是外部确认 `FAIL`。正式 old/R1 one-shot claim 尚未消费；当前终态是
`EXTERNAL_COHORT_NOT_EVALUABLE`。不得用开发窗、synthetic 数据、pooled aggregate 或语义
挑窗来补足来源缺口。

## 绑定

- preregistration SHA256：
  `c763a2254c90f9247eede1f55d58bac0eb4ed16c8b6a20dbc7467c91bbeab1fa`
- discovery contract SHA256：
  `56ebb5927bf85a45978cbdb95822976f4694839af1fccb206e8c1816bf599c7a`
- cross-program access overlay SHA256：
  `6b974a3eed243f41d5fbce7d00ff603c247642bfc443296ee9a517dba5257c40`

## Metadata-only 来源审计

| 确定性顺序 | 来源 | 已闭合 | 未闭合 | 处置 |
| --- | --- | --- | --- | --- |
| 1 | Ground-Challenge | D435i 真实 RGB/depth、15 Hz、36 条轨迹、公开时长和体量；发布方提供 pseudo-GT 文件 | rosbag exact bytes/hash、pseudo-GT 列格式及 timestamp 绑定、明确数据许可 | `HOLD_SOURCE_AUTHORITY` |
| 2 | MultiScan | 真实移动设备 RGB-D；metadata 定义 60 Hz RGB/depth/camera-info、米制 depth、逐帧相机参数；CC BY-NC 4.0 | 下载需登录并接受许可；第二 exact capture 的公开时长/帧数与 member identity 未闭合 | `HOLD_ACCESS_AND_IDENTITY` |
| 3 | ARKitScenes raw | 真实 LiDAR capture；公开 split、RGB/depth/intrinsics/trajectory 文件结构和 direct URL | exact capture cadence/duration、跨模态 timestamp intersection、数据许可适用范围 | `HOLD_TIMING_AND_LICENSE` |
| 4 | SUN3D | 真实 RGB-D、metric depth、intrinsics 与 reconstruction extrinsics；公开 sequence registry | filename clock 单位、extrinsics-to-frame 绑定、archive identity/hash、明确数据许可 | `HOLD_TIMING_AND_LICENSE` |

Ground-Challenge 的官方页面足以证明 D435i RGB/depth 为 640×480、15 Hz，并公布轨迹数量、
时长和下载入口；其 “License” 小节当前只给引用要求，不能替代明确数据许可：
<https://github.com/sjtuyinjie/Ground-Challenge>。

MultiScan 的官方格式页明确给出 RGB、depth 和 camera-info 的 frame rate、frame count、
depth unit 与编码：
<https://3dlg-hcvc.github.io/multiscan/read-the-docs/dataset/files/acquired.html>。

ARKitScenes 与 SUN3D 只保留为 metadata hypothesis，不把文件名或场景描述当作 motion role：
<https://github.com/apple/ARKitScenes/blob/main/DATA.md>；
<https://sun3d.cs.princeton.edu/>。

## 未签发候选锁的原因

1. 协议要求 candidate lock 在 payload 之前绑定不超过四个 family、每个 family 不超过两个
   exact sequences。本次审计对部分 exact capture 的选择依赖可见的场景名称和公开描述，
   不满足“不能用语义选择 motion role”的形式要求。
2. Ground-Challenge 官方论文页面自动暴露了其他 SLAM baseline 的结果描述。记录为
   `other_algorithm_outcome_access=YES`、`claim_relevant_outcome_access=NO`、
   `selection_or_tuning_influence=NO`；该内容没有用于排序或选择。
3. OpenLORIS `office1-*` 与 `cafe1-*` 已明确禁止 confirmation；`corridor1-1` 与
   `corridor1-2` 的 exact access vector 尚为 `UNKNOWN`。按最小范围 fail-closed，复核完成前
   不得纳入。
4. CoRBS 的 modality 与许可较完整，但当前官方 payload 入口不可用；未闭合稳定、发布方控制的
   transport identity。

## Authority 修复复核

只读复核仍未解除两个最接近来源的缺口：

| 来源 | 已闭合 | 未闭合 | 终态 |
| --- | --- | --- | --- |
| OpenLORIS `corridor1-1` / `corridor1-2` | exact 名称、CC BY-ND 4.0、官方入口、D435i 真实 aligned RGB-D 与 timestamped offline LiDAR-SLAM pose capability | exact capture access ledger、完整 bytes/hash、member-level RGB/depth/pose binding | `CANDIDATE_AUTHORITY_HOLD` |
| CoRBS | DFKI 发布身份、真实 color/depth、外部 MoCap trajectory capability | exact capture、dataset license、可用公开 payload/作者镜像、bytes/hash、member/timestamp binding | `CANDIDATE_AUTHORITY_HOLD` |

OpenLORIS corridor exact capture 的 access vector 保守记录为：

```text
metadata_identity                YES
payload_presence                 UNKNOWN
geometry_access                  UNKNOWN
rgb_visual_access                UNKNOWN
other_algorithm_outcome_access   YES
claim_relevant_outcome_access    NO
selection_or_tuning_influence    NO
```

`other_algorithm_outcome_access=YES` 来自官方论文页面自动返回的 SLAM benchmark
结果段落；它不是 RCLE outcome，也没有影响排序、选择或调参，但按严格 firewall 披露。
既有 overlay 只覆盖 `office1-*` 与 `cafe1-*`，不得推断 corridor 的 UNKNOWN 为 NO。

## Fail-closed 终止与后继

当前 evidence version 已耗尽冻结的四来源列表及两项 authority 修复路径，终态为
`CANDIDATE_NOT_FOUND / EXTERNAL_COHORT_NOT_EVALUABLE`。不得下载其他候选、扩大当前列表、
改门槛或在结果后换窗。

若继续，只能在未读取 claim-relevant RGB outcome 的前提下另立、review 新的 source-discovery
version；新版本不得改 geometry role、four-gate、局部门 AND 或失败后不补救规则。只有新版本
签发 candidate lock 后，geometry-only 选择才可开始。
