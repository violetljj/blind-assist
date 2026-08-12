# DepthART task-preserving D3 Phase-A HEAD result

终态：`D3_PHASE_A_ASSET_HEADERS_96_OF_96_AVAILABLE_MEDIA_BODY_UNOPENED`

冻结的 48 个 D3 metadata identity 各自对应
`lowres_wide_intrinsics.zip` 和 `lowres_wide.traj`。一次性 HEAD-only preflight 共执行 96 个请求：

- `96/96` 返回 HTTP 200；
- `96/96` 具有正的 `Content-Length`、ETag 与 Last-Modified；
- 所有请求第一次即成功，没有 transport error；
- 声明总量 `41,979,912` bytes，其中 intrinsics `38,855,860` bytes、trajectory
  `3,124,052` bytes。

本步没有 GET 或 Range GET，媒体 body bytes read 为 0；也没有打开 archive、读取 trajectory、
做 portrait/pose continuity、选择 Phase-A 32 身份或分配 TRAIN/DEVELOPMENT 角色。RGB、depth、
confidence、truth、模型输出、D3 Development 与 R2 均未访问。

当前唯一 successor：
`EXPLICIT_D3_PHASE_A_INTRINSICS_TRAJECTORY_BODY_AND_LABEL_BLIND_CONTINUITY_ACTIVATION`。
该门若被显式激活，只允许下载并验证这 96 个已锁资产，按冻结 pool order 选择前 32 个满足
300-frame portrait/pose continuity 的身份；不允许读取 depth/confidence/RGB、truth 或模型结果。
