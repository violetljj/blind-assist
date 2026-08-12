# TARO O1R R10 fresh clear-enriched confirmation result

R10 的正式终态为
`TARO_O1R_R10_FRESH_CLEAR_ENRICHED_NOT_EVALUABLE_DUAL_CLASS_COVERAGE`，不是
PASS。执行与证据均有效，但 definite `CLEAR` 只覆盖 3 个父视频，未达到冻结的 4-parent
可评估门；因此任何效果或路线晋级均为 false。

完整 source-first 防泄漏链已执行：96/96 zero-body HEAD、1,945,902,515 bytes 的 96/96
source 下载、32 parents / 710 frames inventory、710 次 source-only DepthART inference 与
6,390 个 source-only query features。冻结 R9 selector 在 FARO=0 时先封存 32 个 parent scores，
再封存 top eight；Phase B 随后只读取这 8 个 parents 的 260 个 `highres_depth` frames，未选择
parents 的 FARO read 为 0。全链训练为 0，`UNKNOWN` 从未进入 negative。

2,340 个 query labels 为 1,786 `OCCUPIED`、13 `CLEAR`、541 `UNKNOWN`。正占用侧的冻结门均
通过：definite-label precision `0.999435`、单侧 95% Wilson 下界 `0.997472`、recall
`0.990482`、8-parent macro coverage increase `+0.984977`。但是 13 个 definite `CLEAR` 中有
1 个被预测为 occupied，point specificity 为 `12/13 = 0.923077`，其 Wilson 下界只有
`0.717742 < 0.8`。在保持一个误报时，至少需要 20 个 definite-clear observations 才可能达到
同一 Wilson 门。

唯一 definite-clear 误报已定位为 parent/video `421254/42444754`、frame
`42444754:80991.853`、query `lat_-0.35_yaw_+10.0`。该样本只形成下一版本的 development
diagnostic；不得据此修改已消费 R10 的 selector、positive threshold、denominator 或 gate。

260/260 label records 已重新验证，2,340-query summary 与 sealed result 完全一致；manifest 的
263 个 pre-manifest 文件与 258,481 bytes 均逐文件 size/SHA-256 复核通过。正式 result SHA-256 为
`02772027B79DE7D408DEBC29241E8729053C43C5506FFD2472DCA7F5F6BCED36`，manifest SHA-256 为
`E03B0D6B9C9A52AFBC85FAF147354E7B78389B055E2D8A94AB988AE812C898D3`。

下一版本若继续，必须先冻结任何 abstention rule，再使用新的 parent-disjoint confirmation cohort；
R10 不得重跑或事后救活。该结果不产生 CLEAR 输出、路线晋级、部署、设备、产品或安全主张。
