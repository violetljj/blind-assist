# Metric Depth Calibration Head Distillation R0 Development result

日期：2026-08-03

终态：`CALIBRATION_HEAD_CONSUMED_DEVELOPMENT_NOT_SUPPORTED`

## 结论

冻结的 770 参数校准头显示了明显信号，但没有通过。它只读取 DA V2 第 11 层 384 维
CLS token，预测全局 `a,b`；训练标签是 DA dense depth 到 Metric3D dense depth 的
稳健 affine，RGB-D sensor truth 没有进入训练或特征归一化。

首个 evaluator 输出后，控制面复核发现代码虽未把 sensor truth 用于训练，却在完成下一折
预测前已载入上一折 truth。该输出保留，随后只把执行顺序收紧为“四折预测和 all-consumed
teacher-only 模型全部物化后，才打开 hash-bound sensor report”，未改变模型、常数、门或
数据。新路径复核的 folds、aggregates、increment、终态和模型 SHA 与首个输出完全一致。

四序列 LOSO 聚合上，head 把 raw DA 的 clearance MAE 从 `0.38042 m` 降到
`0.19980 m`，四折都优于 raw DA；但只在 `2/4` 折优于训练折 affine 中位数基线，
没有达到冻结的 3 折门。head 的包络一致率 `89.67%`、false-clear `8.92%`，分别未过
90% 与 5% 门，而且 source-macro false-clear 比常数基线更差。

| 臂 | MAE | 包络一致率 | False-clear | 任务门 |
| --- | ---: | ---: | ---: | ---: |
| Raw DA | 0.38042 m | 75.27% | 24.25% | 2/5 |
| 训练折常数 affine | 0.20924 m | 90.25% | 8.71% | 4/5 |
| 770 参数 feature head | 0.19980 m | 89.67% | 8.92% | 3/5 |
| full-rate Metric3D oracle | 0.11928 m | 93.16% | 4.21% | 5/5 |

94/120 帧形成有效 teacher affine，26 帧因冻结的 residual 门无标签。教师在聚合代理
真值上通过 5/5，说明本次负结果更符合“当前单 CLS、全局 affine 小头不足”，而不是
Metric3D 教师在本 cohort 上失效。

## 异质性与停止规则

四个 held-out sequence 的 head MAE 依次为 `0.04950/0.24882/0.33348/0.17306 m`；
对应常数基线为 `0.11834/0.21756/0.37101/0.13390 m`。跨序列异质性很强，禁止在本次
outcome 后搜索 layer、pooling、ridge、head family、target、seed 或 threshold。

保存的 all-consumed `.npz` 只是 Development 候选资产，不是通过模型；未测 head 自身
端侧延迟、导出、量化、共驻内存或温度。TUM 已消费且只有四段，不能把 LOSO 结果称为
fresh 泛化。后续若继续，必须另立新协议，用 parent/session-disjoint 的真实 RGB-D 或
测距数据，预先冻结新的分区/空间 head 假设。

## 可复现性

- 协议 SHA-256：`5C1F6D8521DB31221A59D76830BF5A2B7C74F5598AF175C988C0D5649EA0E54D`
- cache manifest SHA-256：`49C3F9426D0D658B61741569BB5AD1DB6839DBECF76227531E628FE662D9D8B4`
- evaluator SHA-256：`3CAFFE95C62B0D6A0B20925467CDEB10DEE7DAA9F34B022CFE32B525447200EC`
- firewall-tightened ignored result SHA-256：`C0D819C2A102CD9EA2D998FECA4E149C274E45A1612B05CB9ACFD9CBC3DBBA8C`
- preserved initial result SHA-256：`D65D375B877566A1E83EAE58DCC4A8DFA1E88374C2234AED63ACF4D585276E26`
- all-consumed model SHA-256：`47A6595ECBD44703EC935F33917DA56BDAD89F081CA146960A626335319D10A7`

机器摘要见
[METRIC_DEPTH_CALIBRATION_HEAD_DISTILLATION_R0_DEVELOPMENT_RESULT_2026-08-03.json](METRIC_DEPTH_CALIBRATION_HEAD_DISTILLATION_R0_DEVELOPMENT_RESULT_2026-08-03.json)。
