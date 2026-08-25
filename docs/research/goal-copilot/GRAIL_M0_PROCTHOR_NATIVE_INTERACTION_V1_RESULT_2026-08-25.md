# GRAIL M0 ProcTHOR Native-Interaction V1 Result

日期：2026-08-25（Asia/Hong_Kong）

状态：`NOT_EVALUABLE_RUNTIME_EMPTY_POSITION_PRECONDITION / V1_ROSTER_CONSUMED / STOP_BEFORE_M1`

V1 在冻结 roster 的第一个 house（index 906）执行时，某目标没有任何距离不超过 1.75 m 的 native reachable position。runner 仍以空 `positions=[]` 调用 AI2-THOR `GetInteractablePoses`，Unity 返回 `Every degree of freedom must have at least 1 valid value`，正式运行在产生完整 report 前终止。

这不是 teacher coverage、算法效果或 gate fail：空 native-nearby position 按冻结 interface 应直接产生显式 `NONE`，不应调用要求每个自由度至少一个候选值的查询。V1 没有结果文件，12 个 roster house 全部视为已暴露并永久排除；不得重跑或解释 V1。

唯一允许的修正是 runner 前置条件：`nearby_positions == [] => poses=[] => NONE`，其他 target filter、半径、yaw、visibility、稳定性定义和 gate 均不变。修正在已消耗的 index 906 上作 Development 诊断：26 targets 中 `VALID_SET=24`、`NONE=2`，完整运行结束，oracle pose/path=`24/24`、NONE false commit=`0/3`。该诊断不进入正式证据。

V2 必须在任何 V2 runtime outcome 前以新 salt 冻结新 roster，同时排除 Development index 0 与 V1 全部 12 个 index。M1 在 V2 所有预注册门通过前继续关闭。
