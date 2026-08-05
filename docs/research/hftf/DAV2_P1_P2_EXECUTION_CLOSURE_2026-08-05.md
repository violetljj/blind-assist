# DA V2 P1/P2 执行收口

日期：2026-08-05

终点：`P1_GATE_ESTABLISHED_P2_CANDIDATE_FAMILY_NOT_SUPPORTED_NO_DEVICE_PROMOTION`

## 结论

P1 已建立为可复现、真值参照、fail-closed 的 120 帧小验证门；P2 已进入用户指定的五类
模型侧优化，但当前没有任何候选通过全部准确率与 false-clear 门，因此没有候选获准进入
Android QNN/App 性能评价。`95 ms -> 40 ms` 一类速度收益在本轮不能覆盖质量失败。

## P1 冻结门与基线画像

| 维度 | canonical 结果 |
|---|---:|
| raw depth AbsRel median | 29.43% |
| scale-aligned AbsRel median | 8.33% |
| ground recovery | 100% |
| clearance MAE | 0.3804 m |
| truth VALID/UNKNOWN exact | 99.17% |
| false-clear / all known | 24.25% |
| false-block / all known | 0.48% |
| truth full-state exact | 2.52% |
| truth transition agreement | 55.65% |
| temporal clearance delta MAE | 0.1131 m |

每个候选还输出 geometry state/transition change、harmful/beneficial truth decision change 和典型
失败帧。R2 补齐了一个重要边界：clearance、false-clear 等门控指标若不可定义，必须失败，
不能把“没有 known decision”包装成零 false-clear。canonical identity self-check 为 14/14。

## P2 实测结果

| 类别 | 固定 arm | 终态/关键 veto |
|---|---|---|
| 降分辨率控制 | A1 392 | 失败；raw AbsRel 52.68%，false-clear 34.76% |
| 降分辨率+蒸馏 | A2 distilled 392 | 失败；静态信号强，但 temporal/state 不过门 |
| 轻量 student | A3 temporal mobile | 失败；保守占用塌缩、false-block |
| 轻量 RGB-D student | A4 R1 | 失败；false-block 60.63%，harmful change 55.72% |
| 选择性混合精度 | A5S W8A16 | 失败 1/14；temporal delta 0.1351 m |
| token/attention 成本 | A4-BS25 | 失败 11/14；0 个可比较 collision decision |

A2 把 raw AbsRel 降到 `10.53%`、false-clear 降到 `2.86%`，说明蒸馏方向值得保留；
但 temporal clearance `0.221 m` 和状态门失败，不能晋级。A5S 是最接近通过的模型变换，
但冻结 AND gate 不允许用 13 个通过项抵消唯一时序失败。A4-BS25 主机 CUDA P95 约
`48.42 ms`，同时质量灾难性退化，正好证明必须先过 P1 再谈速度。

## 多速率架构判定

`student 8–12 Hz + DA V2 teacher 1–3 Hz + YOLO/seg 15–30 Hz` 保留为目标架构，
不是本轮已验证结果。teacher 可以承担周期校准、disagreement 和 confidence supervision；
只有绑定独立 ToF/量距时才可称 metric scale anchor。teacher stale、尺度无效、disagreement
过大或 student 不确定时必须输出 `UNKNOWN`。

当前不应开始 cadence 搜索或设备 profile。下一步只能在不打开 P1 outcome 的 training-only
数据上，先冻结一个新的时序 student 合同：显式监督 teacher clearance delta/state transition、
置信度与 disagreement，并固定单一 seed/checkpoint 规则；随后仍只允许一次 P1 判定。旧 P1
已经 consumed，不能用于调 loss、层集合、token 数、精度分区或 cadence。

本收口只提供 Development regression/Pareto 证据，不是产品、安全或默认 App 替换授权。
