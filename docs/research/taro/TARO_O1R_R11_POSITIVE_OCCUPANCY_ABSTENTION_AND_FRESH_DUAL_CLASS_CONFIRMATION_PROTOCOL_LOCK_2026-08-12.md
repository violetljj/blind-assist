# TARO O1R R11 positive-occupancy abstention and fresh dual-class protocol lock

状态：`LOCKED_NON_EXECUTING_PRE_NEW_NETWORK_PRE_SOURCE_PRE_OUTCOME / SCIENTIFIC_OUTCOME_NOT_RUN`。

R10 的唯一 definite-clear 误报来自仅满足弱、低、远端 base cell 的 source component。R11 因此冻结
一个 development-only 候选：保留 R7 的 2-pixel / 0.08 m / 2.0 m base positive，但还要求既有
grid 中 `16 pixels`、`0.15 m height`、`1.5 m forward` 三个相邻强度 cell 至少一个成立；否则
`OCCUPIED → UNKNOWN`。它不接受 identity、truth、label、FARO 或 outcome 字段，不输出 `CLEAR`，
并保持 R6 prior OCCUPIED。

在 consumed R10 上的只读形成 replay 抑制 1 个 clear false positive 和 1 个 occupied true positive；
candidate recall 为 `0.989922`，query clear specificity 为 `13/13`。这不是对 R10 的救活或 PASS；R10
仍因 clear 仅覆盖 4 frames / 3 parents 而 `NOT_EVALUABLE`。

新的 metadata-only pool 固定为 48 个 Training parents / 144 个 URL，并显式排除 R10 全 32 个
source-pool parents，而不只排除 selected eight。若以后另获授权，必须先对全部 48 parents 完成
source/model Phase A，同时封存 R7 base、R11 candidate 与 R9 parent scores；R9 只作 parent ranking，
不是 query truth。随后按冻结 score/tie-break 封存 top 24，才可只读取这些 parents 的 FARO；
unselected FARO 必须为 0。

正式 evaluability 至少需要 16 个可评 parent、12 个 occupied parents / 200 occupied queries、4 个
clear parents、12 个 definite-clear physical frames 和 20 个 clear queries。clear-frame success 定义为：
该 frame 的所有 definite-clear queries 均未被输出 OCCUPIED。确认门同时限制 candidate precision/
recall、frame/parent clear specificity、Wilson 下界以及相对 R7 的 micro 与 parent-macro occupied-recall
损失；`UNKNOWN` 永不进入 negative。

本锁只允许静态 validator 和 metadata roster 重算。它不授权 HEAD、GET、source body、模型、FARO、
truth label、训练、设备、部署、产品或安全动作。启动任何 R11 数据动作前，必须先取得 exact 48-parent
数据使用授权，再另立 zero-body HEAD execution lock。
