# DepthART task-preserving D1 quality-screen result

状态：`FAIL / D1_TASK_QUALITY_FAIL_STOP_R2_CANDIDATE_NOT_AUTHORIZED`

冻结的 Development screen 已完整执行：48/48 个 device chunk、2400/2400 帧、
21600 个 band×horizon cells 与 7200 个 band clearances 均进入一次性汇总。结果不是设备或
receipt 失败，而是候选没有通过冻结的 task-quality gates。

候选的 pooled known coverage 为 `0.99547`，与 reference 相同；相对 reference，clearance
MAE 从 `0.52047 m` 降至 `0.38443 m`，false-clear 从 `0.28025` 降至 `0.16651`，
geometry transition agreement 从 `0.69003` 升至 `0.79365`。但绝对门仍未满足：
clearance MAE 要求 `<=0.20 m`，false-clear 要求 `<=0.08`，false-block 实测
`0.18648`、要求 `<=0.02`，geometry transition agreement 要求 `>=0.90`。候选的
false-block 还相对 reference 的 `0.05915` 明显恶化，因此对应 noninferiority gate 也失败。

聚合同时按协议 fail-closed：parent `426245` 与 `470297` 的 required metrics 全部不可计算，
parent `382841` 的 clear-truth denominator 为零，导致 false-block 不可计算；因此
parent macro、session macro 与 worst-parent 不能全部 finite，`aggregation_complete=false`。
`UNKNOWN` 没有被当作 negative，也没有改变冻结分母或门限。

完整机器 payload SHA-256 为 `421D749B...34D`，完整结果 SHA-256 为
`A085F207...ABA`。该结果只关闭 D1 Development task-quality screen：R2 candidate lock
不授权，R2 cohort 仍未访问，性能、DA2 替换、默认 App、production 与 safety 均不授权；
strict G4-D 负终态不变。若建立新版本，必须先另立显式 pre-outcome protocol，不能在本次
已消费 Development outcome 上事后修改候选、样本、后处理、分母或门限。
