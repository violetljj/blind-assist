# TARO O0R ARKitScenes Content-Length HEAD Result — Attempt 02

终态：`TARO_O0R_ASSET_HEADERS_NOT_AVAILABLE_NO_REPLACEMENT`

冻结的 72 个 HEAD target 已执行一次，响应体读取为 0。71 个 asset 返回可用的正 Content-Length，
可用总长度为 `1,105,086,109 bytes`；唯一失败是 ADAPTER_FIT video `47333152` 的
`lowres_wide.traj`，3/3 attempts 均为 HTTP 403，Content-Length 不存在。

HEAD root 已创建并消费，evidence manifest 复核通过。source/work/truth/factor 四个 root 仍不存在，
GET、source body、truth、uncertainty fit、DepthART 与 factorial 均未运行，truth one-shot 未消费。

按 outcome 前冻结规则，不得替换 `47333152`、不得重跑 Attempt 02、不得进入 source/truth。当前 O0R
证据版本终止为 `NOT_EVALUABLE_SOURCE_ASSET_UNAVAILABLE`，successor 为 `null`。
