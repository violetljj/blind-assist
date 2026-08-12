# DepthART task-preserving D3 fresh metadata roster result

终态：`D3_FRESH_METADATA_POOL_48_LOCKED_MEDIA_UNOPENED`

D3 已在 Apple 官方 ARKitScenes Training split 上锁定 48 个 metadata-only 候选 identity。
规划器扫描当前工作区 `docs/research` 的 1,485 个 JSON/Markdown 文件，其中 136 个文件
命中 394 个官方 visit/session identity；该扫描包含尚未提交的并发研究 receipt。选取规则先
排除全部命中 identity，再要求 unique visit，按 `sha256(visit_id:video_id)` 固定排序取前 48。

D1 roster、D2 source pool 与 sealed R2 roster 的显式 overlap 均为 0；48 个 visit 和 48 个
session 也各自唯一。当前 pool 只是候选集合，尚未分配 TRAIN/DEVELOPMENT：未来 Phase-A 才会
按冻结顺序寻找前 32 个 portrait/pose-qualified identity，Phase-B 再寻找前 16 个
source-truth-support-qualified identity，最终前 8 个为 TRAIN、后 8 个封存为 DEVELOPMENT。

本步没有发出媒体 HEAD、没有读取 body、truth 或模型输出，也没有训练。当前唯一 successor 为
`EXPLICIT_D3_PHASE_A_INTRINSICS_TRAJECTORY_HEAD_ONLY_PREFLIGHT_ACTIVATION`；该 successor 只允许
对精确 48 身份的 intrinsics/trajectory 资产做 HEAD，不自动授权下载或内容扫描。
