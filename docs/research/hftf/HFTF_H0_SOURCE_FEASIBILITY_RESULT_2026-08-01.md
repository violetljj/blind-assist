# HFTF H0 来源可行性结果

日期：2026-08-01

workflow：`DEVELOPMENT_STANDARD`

终态：`HFTF_H0_SOURCE_FEASIBILITY_PARTIAL`

claim ceiling：`SOURCE_FEASIBILITY_ONLY`

## 结论

当前本地 replay **只足以支持 host-only 静态 metric geometry projection canary**。
multi-height human envelope、short-horizon future teacher 与 student-effect 均为
`NOT_EVALUABLE`，因此 H1、训练和主线挑战均未获授权。

这是一个有效的 `PARTIAL`：它既不是数据整体失败，也不是 HFTF 模型效果或创新性的
正/负结果。

## 输入与权威输出

- replay：
  `artifacts.local/evidence/datasets/sanpo-synthetic-replay-25frames-20260720`
- 声明身份：`SANPO-Synthetic v0 / train`；H0 **未对该来源身份做密码学认证**；
- parent session：1；
- frames：25，10 FPS，单 sequence span 2.4 s；
- 声明 modalities：RGB、panoptic mask、metric depth、intrinsics、camera pose CSV；
- authoritative report：
  `artifacts.local/evidence/hftf/h0-source-feasibility-r0-20260801-final-v3/source_feasibility.json`
- report SHA-256：
  `43e72db3395b698a6b0ee9753e5aa6088c64e85e3cbe396b53a5a732df13d8be`
- audit script SHA-256：
  `dd562a7a238c40443a2fa5e9c8baccabd3236b9b660a57f72bb8370136e35e8a`
- 独立重跑：
  `h0-source-feasibility-r0-20260801-final-v3-repro` 与 authoritative report
  逐字节一致。

[SANPO 原论文](https://arxiv.org/abs/2309.12172)说明其 synthetic source 提供
pixel-accurate depth/masks；这支持来源候选的合理性，但不能替代本地文件身份认证、
HFTF body calibration、pose-frame/time mapping 或人类事件真值。

## 核验结果

| 层级 | 结果 | 直接证据 |
| --- | --- | --- |
| source contract | `PASS_WITH_IDENTITY_CEILING` | 25/25 rows 均有非空 session/sequence；75/75 RGB/mask/depth canonical paths 独立且 SHA-256 一致；完整 observation hash triplet 无重复；所有 PNG 完整解码且尺寸一致 |
| metric depth | `PASS` | 25/25 gzip payload 的 float16 header、shape 与 finite-positive samples 由 H0 独立复算；minimum finite-positive fraction `1.0` |
| QA binding | `PASS_WITH_DECLARATION_CEILING` | metric QA schema/frame/depth-path identity 和 replay validation 一致；QA 的 official split 字段只作为声明，不提升来源身份权威 |
| static metric projection | `ELIGIBLE` | intrinsics 有限，principal point 合法，camera dimensions 与 2208×1242 manifest 一致 |
| multi-height human envelope | `NOT_EVALUABLE` | 无结构化 body-frame/ground contract；且通用 H0 即使读到合同也不能自行认证物理标定 |
| short-horizon future teacher | `NOT_EVALUABLE` | raw pose CSV 有 50 rows，但没有 frame ID/timestamp；且通用 H0 不接受 sidecar 自签 source-native mapping |
| student effect | `NOT_EVALUABLE` | 只有 1 个 parent session；没有独立 hash-bound parent-event ledger |

canonical metric audit 的本轮独立复算报告为
`artifacts.local/evidence/hftf/h0-source-feasibility-r0-20260801-independent-metric-reaudit/metric_replay_audit.json`，
SHA-256
`fb919b4917b45cd65a592aa9f43c19bdc5ef3c0a90dd555fe4038bc806e43e92`；
25/25 depth integrity 通过，raw pose 50 rows，但 explicit binding 为 false。

当前 blockers：

1. `body_frame_contract_structurally_invalid_or_absent`
2. `source_specific_body_calibration_verifier_required`
3. `pose_binding_contract_structurally_invalid_or_absent`
4. `source_specific_pose_time_mapping_verifier_required`
5. `single_parent_session_only`
6. `separate_hash_bound_parent_event_ledger_required_for_effect`

## Fail-closed 加固记录

早期开发输出曾把 SANPO depth payload 的两个 float16 高/宽 header 误判为 shape
mismatch；后续开发版又暴露了 QA 自报、PNG 仅读 header、缺失 group row 被静默排除、
body/pose sidecar 自我认证等 fail-open 风险。当前实现已经：

- 独立重算全部文件 hash、完整 PNG decode、depth header/shape/finite-positive；
- 拒绝重复 canonical path 或完整 RGB/mask/depth hash triplet 冒充独立帧；
- 对 QA 字段要求精确 JSON boolean 与结构合同，字符串 `"false"` 不再因 truthiness 通过；
- frame count、finite-positive fraction 与相机内参必须是非 bool 的正确数值类型；
- 要求每个 manifest row 都进入唯一非空 session/sequence group；
- 对 pose/body 合同做一一映射、有限值、归一化与几何一致性检查；
- 明确把这些合同的结构通过与可信来源映射/物理标定准入分开；
- 永久禁止通用 H0 授予 multi-height、future 或 effect eligibility。

此前所有输出（包括 `final-v1`、`final-v2`）仅保留为非权威 implementation
diagnostics；本文件只引用 `final-v3`。

## 当前授权与禁止

唯一允许的下一步是
`static_metric_geometry_projection_canary_only`。它可以检查 metric point-cloud、坐标和
field interface，但不得凭假定 camera height 产出正式 foot/body/head 标签。

当前不授权：

- 把本地 manifest 的 SANPO/official 字段称为已认证来源身份；
- 以 pose CSV 行号默认对应 source frame，或让 hash-bound sidecar 自签权威；
- multi-height/future teacher、temporal student 训练或效果比较；
- 把 panoptic/depth proxy 称为 human collision truth；
- HFTF 晋级、Android、提醒、TTS、震动、默认 App 或安全主张。

## 后续任务与停止条件

### H0.1 — Source-specific pose/body authority

- 依赖：当前 H0 结果；
- 交付：从一手格式/采集 receipt 独立复算的 pose-frame/time mapping verifier，以及
  camera-to-body/ground calibration verifier；输出必须绑定输入与实现 hash；
- 完成：两个 verifier 均可从原始来源重算，不能只验证候选自己提供的 sidecar；
- 停止：找不到 source-native mapping 或物理标定权威时 `NOT_EVALUABLE`，不进入 H1。

### H0.2 — Independent-session expansion

- 依赖：H0.1；
- 交付：多个独立 source ancestry/parent sessions 的 hash-bound replay 与
  session-first split；
- 完成：每个 session 通过相同 source-specific admission，且 frames 不计作独立样本；
- 停止：只增加同一 source 的 evidence version、session alias 或重复帧时不授予
  replication credit。

### H1 — Geometry teacher canary

- 依赖：H0.1、H0.2；
- 交付：single-height/current、multi-height/current、multi-height/future 三臂及
  unknown/failure atlas；
- 完成：只允许 `GEOMETRY_PROXY_MECHANISM_SUPPORTED` 或精确定义的负/不可评终态；
- 停止：multi-height 或 future 轴对相应简化 arm 没有稳定增量时关闭该 formulation。

独立人类 parent-event truth 在 H2/H3 前另行建立；teacher mechanics 通过不能替代它。
