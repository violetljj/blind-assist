# Spatial Calibration Head R1 cohort roster lock

状态：`METADATA_ROSTER_24_LOCKED_MEDIA_UNOPENED_LICENSE_REVIEW_REQUIRED`

官方 ARKitScenes commit、CSV 和 P0 protocol 已绑定。metadata-only planner 在 5,071 行中
排除了 8 条 `visit_id=NA` 以及跨官方 split 的 `visit_id=381879`，随后机械选择了
16 train、4 validation、4 sealed parents；角色间 `visit_id` overlap 为 0，train 的
四个 CV folds 各 4 parents。

最终 P0 数值补齐后的 roster 输出 SHA-256：
`7CE2D9931723EF7517531F7389FF1DFA0E4BF9BD4C8291A9E72A5BBFF7102EEC`。
前两个 metadata-only 输出原样保留为 superseded pre-affine / pre-identity-firewall
receipts，没有媒体访问，名单与 role 未变化。

当前没有读取任何 ARKitScenes RGB/depth/confidence/trajectory 字节。精确 parent/video
列表见同名 JSON。入选视频不得因后续内容、难度、缺帧或结果而替换；asset qualification
失败使本 cohort 进入 `COHORT_NOT_EVALUABLE_NO_REPLACEMENT`。

官方 LICENSE 文本已经哈希绑定，但本地 receipt 仍是
`SOURCE_LICENSE_TEXT_BOUND_USER_REVIEW_REQUIRED_BEFORE_MEDIA_DOWNLOAD`。在明确复核前，
`media_download_authorized=false`；这不阻止实现和 synthetic/unit tests，但阻止 P1 媒体下载。
