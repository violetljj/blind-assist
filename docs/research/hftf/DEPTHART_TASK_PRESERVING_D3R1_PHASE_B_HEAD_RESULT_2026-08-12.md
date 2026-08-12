# DepthART-S D3R1 Phase-B HEAD-only 结果

状态：`D3R1_PHASE_B_ASSET_HEADERS_64_OF_64_AVAILABLE_MEDIA_BODY_UNOPENED`

## 结果

- exact-32 Phase-A identities × `lowres_depth.zip` / `confidence.zip` 共 `64/64` 个 HEAD 可用。
- `64/64` 均为 HTTP 200、正整数 Content-Length、非空 ETag 与非空 Last-Modified。
- 声明总大小 `5,580,879,686 bytes`：depth `5,329,635,728`，confidence `251,243,958`。
- 全部一次成功；redirect、recovered error、unresolved error 均为 `0`。
- response/media body bytes read 均为 `0`。没有 GET/Range、archive/member/decode 或 source-truth-support。
- 原离线 validator 重放 exact-64 request plan、attempt history、响应头与 zero-body 边界后 PASS；随后发现其对
  attempt→row 矛盾字段拒绝不足，因此保持原结果/validation 不变，新增只读 post-result repair audit。
- Repair audit 独立派生 availability、status、redirect、recovered/unresolved flag 与 terminal，四类 mismatch
  均为 `0`，确认相同不可变 HEAD artifact 仍为 PASS；没有重发 HEAD 或访问网络/body。

机器结果：[DEPTHART_TASK_PRESERVING_D3R1_PHASE_B_HEAD_RESULT_2026-08-12.json](DEPTHART_TASK_PRESERVING_D3R1_PHASE_B_HEAD_RESULT_2026-08-12.json)

## 权限边界

本结果只建立远端 availability 与 declared-size evidence。它不证明 body 完整性、frame coverage、
source-truth-support、first-16 selection、角色、模型质量或部署能力；不授权 RGB、TRAIN/DEVELOPMENT、
R2、性能、默认 App、production 或 safety。

## 唯一 successor

`EXPLICIT_D3R1_PHASE_B_DEPTH_CONFIDENCE_BODY_AND_SOURCE_TRUTH_SUPPORT_ACTIVATION`。

该门必须另行冻结 exact-64 GET/body、全 32 identity 的完整 source-support audit 与 fail-closed
checkpoint/validator；不得从本 HEAD PASS 自动下载约 `5.58 GB` body，也不得提前形成 first-16 lock。
