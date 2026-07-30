# Dual-loop R1 unseen natural event R0 — rank-2 truth result

## 结论

Wikimedia Commons Shiraz rank-2 已在 baseline 与 candidate 输出均未打开时冻结为
`TRUTH_FROZEN_ADEQUATE`：7 个正例事件、6 个负窗，其中 4 个相机转向/横看负窗、
2 个路线外静态障碍负窗。该结果只授权固定 10 Hz 输入上的 baseline-only adequacy，
不构成算法效果、Confirmation、产品或安全结论。

## 来源与输入身份

- source：`commons_iran_shiraz_city_tour_2021_5`；
- 480p payload SHA-256：
  `63e5b32d9b08e6a2c17b1e3d0b20b6bde03e64d88da0c844579d424051a65b2e`；
- 1 Hz review manifest SHA-256：
  `303e592812b62ce404f4bb2dce7c21ac3cb55a26431c2175c4a9c5b1d8fd94a8`；
- 10 Hz replay manifest SHA-256：
  `af0ab3c735d96737f451a6e64d1784681966345c7849131ad51bd46c9d7e6571`；
- frozen candidate：`039757b2da41c051373f8ee3189c4b06028f5295`。

## 真值形成

两路隔离 AI reviewer 使用同一 canonical prompt 和 25 张联系表，只检查 RGB，
均声明 candidate 不可见。A 提议 4 个正例/33 个负窗，B 提议 10 个正例/17 个负窗；
由于事件集合存在实质分歧，第三路 fresh adjudicator 显式读取两份哈希绑定的 review，
裁决为 7 个正例和 6 个负窗。`finalize_rank2_truth.py` 校验 reviewer 身份、prompt、
input、可见性、review 哈希与最低真值门后，发布不可覆盖的 ledger：

- truth ledger SHA-256：
  `b2865cbeeb955fab62f02123031fe0f29af0a48a18443cf4581e6572e267a26c`；
- canonical truth-freeze-r2 receipt SHA-256：
  `7ddd0e4d9cf968a7594c9e960b4f76e3b1c2380e5f4f2b13f6e585bbf84aacf0`；
- review A SHA-256：
  `efbb2a27b9a04364f501b4b24fd1e59237f30d3da8f762efcc57e015d4bc404a`；
- review B SHA-256：
  `76819bbdbe2c5e2fa54e65e093b5d4ed4b6897f1fd2eb3a10e043855a5d5b546`；
- adjudication SHA-256：
  `d4e56a483fe1275edaebc503c2c51ea59d29156577ee1db11b2e0a1ad30e6d5e`。

`truth-freeze-r2` 相比最初 r1 不改任何 truth row；它只在发布 receipt 前新增有限
confidence、视频边界、唯一 ID 和闭区间互斥结构门。两版 ledger 字节一致，r2 为后续
设备门的 canonical receipt。

这是 model-reviewed event truth，不等同于人工临床标注。一个连续 capture 是最高
独立单位，7 个事件不得当成 7 个独立 session 做总体统计外推。

## 下一门

设备端先独立运行 baseline-only。完整 4,891 帧 trace 必须覆盖全部 truth item；
baseline 至少命中 1 个正例并至少误触发 1 个负窗，才由 host evaluator 生成绑定
truth/input/baseline 哈希的 candidate authorization。candidate 随后只重放 baseline
冻结的 detections 与 detector metrics；不得重跑检测器、调参、改窗口或改变
250 ms 逐事件延迟容差。rank-2 的来源激活、APK identity、授权和 terminal 均由
[machine-readable protocol](DUAL_LOOP_R1_UNSEEN_NATURAL_EVENT_R0_RANK2_PROTOCOL_2026-07-31.json)
绑定；窗口内任意 simulated accepted feedback 只作 window-level 命中，不声称
target-specific attribution。

## 数据复用边界

项目不再把“旧研究使用过”解释为数据集永久失效。它只取消该 exact session 对同一
R1 候选的 unseen 身份；仍可用于 Development、回归、失败分析或新问题。缺少原生提醒
标签的数据也可按本轮做法在输出盲条件下由多模型复核补齐，但来源独立单位、先标后看
输出与事件去重仍必须保留。
