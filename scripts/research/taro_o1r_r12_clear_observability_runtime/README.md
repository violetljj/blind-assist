# TARO R12 clear/task observability runtime

状态：`current / REVERSIBLE_EXPLORATION / BONN_POSE_PAIR_CAPABILITY_PASS / POSITIVE_ORACLE_R1_NOT_EVALUABLE_DENOMINATOR / LEARNED_SCORER_NOT_JUSTIFIED`

## 稳定 Interface

- `development_replay.py`：在 consumed R11 上只改变一个 source-only clear-ranking 轴；只保留为
  高召回 proxy，不输出 `CLEAR`。
- `positive_oracle_canary.py`：先只读 Bonn RGB-D 的 timestamp/path/pose 做 outcome-blind pair 能力筛选，
  再以相同的一帧额外观测预算比较 static R7、passive、固定 `6±2 cm` micro-baseline、generic
  max-parallax 与 task-directed oracle。输出只有 `OCCUPIED/UNKNOWN`；native depth 只承担
  source-derived Development label，不是 fresh Confirmation truth。

R1 在 25/26 pose-capable parents 中 outcome-blind 均匀选出 100 references，实际评价 56 references / 504
queries；label census 为 `404 OCCUPIED / 2 CLEAR / 98 UNKNOWN`，recovery opportunity 与 CLEAR denominator
都只覆盖 1 parent，低于各自 4-parent gate。因此所有臂间 decision 为 `null`，不得从表面 recovery 训练
scorer。跟踪结果见
[`TARO_TASK_DIRECTED_OBSERVABILITY_POSITIVE_ORACLE_CANARY_RESULT_2026-08-13.json`](../../../docs/research/taro/TARO_TASK_DIRECTED_OBSERVABILITY_POSITIVE_ORACLE_CANARY_RESULT_2026-08-13.json)。

## 输出

正式运行只写新的 exclusive `artifacts.local/evidence/taro/` 子目录；tracked result 只摘录 terminal、冻结分母、
指标与 local artifact hash，不提交 source payload。重复使用同一输出目录必须 fail closed。

## 安全边界

- source-native depth 只作 source-derived Development label，不是 fresh Confirmation truth；
- source selection 不读 task outcome，`UNKNOWN` 不作 negative，所有 arm 只输出 `OCCUPIED/UNKNOWN`；
- task oracle 仅用于估计可达上限，不是可部署算法；不授权模型、训练、Android、产品或 safety。

## 停止条件

任一 source/pose/calibration binding 无效即停止；少于 48 evaluable references、4 recovery-opportunity parents
或 4 CLEAR-denominator parents 时，所有臂间 decision 必须为 `null`。当前 Bonn R1 已触发后两项，不能调门、
重采样或训练 scorer 回救；唯一 successor 由 TARO current 定义。

定向验证：

```powershell
E:\codex-tools\bin\blindassist-python.cmd -m unittest `
  scripts.research.taro_o1r_r12_clear_observability_runtime.test_development_replay `
  scripts.research.taro_o1r_r12_clear_observability_runtime.test_positive_oracle_canary
```

动态状态、唯一 successor 与 claim ceiling 以
[`docs/research/taro/README.md`](../../../docs/research/taro/README.md) 为准。
