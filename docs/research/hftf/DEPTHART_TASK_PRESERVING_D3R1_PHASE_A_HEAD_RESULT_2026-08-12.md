# DepthART-S D3R1 Phase-A HEAD result

状态：`D3R1_PHASE_A_HEAD_PREFLIGHT_PASS_BODY_UNOPENED`

## 结果

冻结的 exact-127 Training metadata roster 上，`lowres_wide_intrinsics.zip` 与
`lowres_wide.traj` 共 254 个请求全部以 HEAD-only 通过：HTTP 200、正整数
`Content-Length`、非空 `ETag` 与非空 `Last-Modified` 均为 254/254。无重定向、重试、
未解决错误或响应正文读取。

远端声明总大小为 `133,734,849` bytes：intrinsics ZIP `123,769,897` bytes，trajectory
`9,964,952` bytes。该数值只来自响应头，不是已经下载或校验的媒体大小。

## 边界

本门的 response/media body 读取均为 0 bytes；没有 GET/Range、archive member、trajectory
pose、truth/model、selection、角色、Phase-B 或 R2 访问。HEAD PASS 不证明 body integrity、
`.pincam`/trajectory schema、portrait/pose continuity、模型质量、性能、产品或安全。

机器结果与离线复验保存在忽略目录
`artifacts.local/evidence/hftf/depthart-task-preserving-d3r1-phase-a-head-20260812-r0/`，
并由 [governed machine result](DEPTHART_TASK_PRESERVING_D3R1_PHASE_A_HEAD_RESULT_2026-08-12.json)
锁定 bytes/SHA。

## 唯一 successor

`EXPLICIT_D3R1_PHASE_A_INTRINSICS_TRAJECTORY_BODY_AND_LABEL_BLIND_CONTINUITY_ACTIVATION`

该门必须另行显式激活；当前结果本身不授权下载。
