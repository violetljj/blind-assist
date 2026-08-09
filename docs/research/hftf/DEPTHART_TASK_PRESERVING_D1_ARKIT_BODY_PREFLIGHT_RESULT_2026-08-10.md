# DepthART task-preserving D1 ARKitScenes body preflight result

状态：`PASS / FINAL_DEVELOPMENT_ROSTER_LOCKED / NO_TASK_OUTCOME_ACCESSED`

冻结的 16 个 Training identity 已按 primary 后 reserve 的顺序完成 label-blind
media integrity 与 product-portrait continuity 检查。4 个 primary 和 4 个 reserve
满足每个 identity 300 个连续 RGB/depth/confidence/K 帧、相邻帧 gap `<=0.5s`、
pose bracket `<=0.25s` 与 pose-derived portrait 门；4 个不合格 primary 仅按冻结
reserve 顺序替换。完整机器结果见同名 JSON，逐资产 SHA 与逐帧 receipt 保存在
`artifacts.local/datasets/depthart-task-preserving-d1-arkitscenes-preflight-20260810-r1/manifest.json`
（SHA-256 `06EBAFD4...3CE5`）。

最终 Development roster 为：

1. `426245/42898254`（reserve，替换 `438803/44358207`）
2. `471146/47204473`（reserve，替换 `421251/42444834`）
3. `382841/41126916`（primary）
4. `470341/47331003`（reserve，替换 `483306/48018537`）
5. `470297/47331084`（reserve，替换 `382145/41098158`）
6. `466159/44796594`（primary）
7. `382312/41159702`（primary）
8. `484724/48458772`（primary）

本次只为 source integrity 解码媒体，未进行 RGB 视觉/语义挑选，未读取 task truth、
模型输出或 R2 cohort，未产生质量、性能、默认 App、生产或 safety 权限。
