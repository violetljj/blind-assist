# TARO O0R R6 untouched cohort and data-use lock

状态：`EXACT_8_PARENT_COHORT_FROZEN / DATA_USE_AUTHORIZED / HEAD_EXECUTION_FALSE`

按冻结的 repository exclusion snapshot 和 salted SHA-256 规则，从官方 ARKitScenes upsampling Training
metadata 的 2,257 行中排除仓库已出现的 186 个官方 identity，剩余 1,757 个 eligible rows。每个 visit 只取
一个 video，得到 8 个 untouched parents：

| visit_id | video_id |
|---|---|
| 467175 | 47333514 |
| 467312 | 45261569 |
| 435329 | 42899445 |
| 423306 | 42897745 |
| 466652 | 45261100 |
| 469650 | 47333562 |
| 470439 | 47115427 |
| 469830 | 47334055 |

这 8 组身份在 upsampling、raw、3DOD 三份官方 split 中均一一存在，与 R4/R5/R6 formation 的 24 个 parents
零交叉。本轮用户授权原文为“授权”；授权被约束到上述 exact cohort 和每个 video 的
`upsampling.zip`、`lowres_wide_intrinsics.zip`、`lowres_wide.traj`，且每个网络/下载/模型/truth 阶段仍必须
另签 one-shot execution lock。

下一个唯一动作是 24 个 exact URL 的 HEAD-only 预检：允许响应 body bytes 为 0，request sequence SHA-256 为
`25B710DD39823754C08FB147FBA74E577FFCB7137CE55A7F82312BC896F2B2B4`。本锁本身未发送请求、未下载、未解码，
也未读取模型输出或 truth。
