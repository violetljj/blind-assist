# TARO R12 clear/task observability runtime

状态：`current / REVERSIBLE_EXPLORATION / R12_THREE_SOURCE_LABEL_SATURATION_LOCALIZED / R13_TASK_EVIDENCE_ORACLE_HEADROOM_PASS / POSE_SCORER_NOT_YET_RUN`

## 稳定 Interface

- `development_replay.py`：在 consumed R11 上只改变一个 source-only clear-ranking 轴；只保留为
  高召回 proxy，不输出 `CLEAR`。
- `positive_oracle_canary.py`：先只读 Bonn RGB-D 的 timestamp/path/pose 做 outcome-blind pair 能力筛选，
  再以相同的一帧额外观测预算比较 static R7、passive、固定 `6±2 cm` micro-baseline、generic
  max-parallax 与 task-directed oracle。输出只有 `OCCUPIED/UNKNOWN`；native depth 只承担
  source-derived Development label，不是 fresh Confirmation truth。
- `balanced_pose_source_frontdoor.py`：从两个已物化、outcome-open 的 TartanGround Development manifest
  冻结 15-parent、完整 10 Hz pose roster；reference selection 只读 manifest/path/pose，不能让历史 5 Hz
  media 抽帧决定微基线能力。加 `--materialize-missing` 后只补取已选 reference 的缺失 depth，随后检查
  `48 references / 4 recovery parents / 4 CLEAR parents` 分母门。只有 PASS 才授权另立五臂 R2。
- `arkitscenes_balanced_pose_source_frontdoor.py`：只用冻结 manifest、timestamp 与 trajectory 选 pose pair，
  只保留不会改变 `256x192` contract 的 0/180° orientation；选择后才读取 depth/confidence/intrinsics。
- `tum_balanced_pose_source_frontdoor.py`：联合两个 pre-outcome frozen TUM cohort 的 14 个独立序列；支持目录与
  tgz source，selection 只读 index/pose，选择后以 `256x192` 观察和 native `640x480` label 做 R12 census。
- `task_evidence_oracle_canary.py`：R12 三源共同暴露 zero-CLEAR 饱和后另立的 R13 新任务。每个 reference
  在读取 neighbor depth 前冻结相同 pose-only proposal pool；一帧预算下比较 passive、fixed-micro、generic
  与 task-evidence oracle 对九个 body/path capsule 内 observed evidence cells 的增量。未观察 cell 永远保持 UNKNOWN。

R1 在 25/26 pose-capable parents 中 outcome-blind 均匀选出 100 references，实际评价 56 references / 504
queries；label census 为 `404 OCCUPIED / 2 CLEAR / 98 UNKNOWN`，recovery opportunity 与 CLEAR denominator
都只覆盖 1 parent，低于各自 4-parent gate。因此所有臂间 decision 为 `null`，不得从表面 recovery 训练
scorer。跟踪结果见
[`TARO_TASK_DIRECTED_OBSERVABILITY_POSITIVE_ORACLE_CANARY_RESULT_2026-08-13.json`](../../../docs/research/taro/TARO_TASK_DIRECTED_OBSERVABILITY_POSITIVE_ORACLE_CANARY_RESULT_2026-08-13.json)。

R12 的 TartanGround、ARKitScenes 与 TUM frontdoor 分别终止于 pair support、same-resolution denominator 和
native-depth zero-CLEAR saturation；旧 label 不回调。R13 在 TUM 48 evaluable references 上取得 task oracle
`17.96` parent-macro novel cells/reference，高于 generic `14.02` 与 passive `13.88`；12 个机会 parents、
10 个 strict-win parents、所有 arm retention failure=0。跟踪结果见
[`TARO_TASK_OBSERVABILITY_BALANCED_SOURCE_FRONTDOOR_AND_QUERY_EVIDENCE_ORACLE_RESULT_2026-08-13.json`](../../../docs/research/taro/TARO_TASK_OBSERVABILITY_BALANCED_SOURCE_FRONTDOOR_AND_QUERY_EVIDENCE_ORACLE_RESULT_2026-08-13.json)。

## 输出

正式运行只写新的 exclusive `artifacts.local/evidence/taro/` 子目录；tracked result 只摘录 terminal、冻结分母、
指标与 local artifact hash，不提交 source payload。重复使用同一输出目录必须 fail closed。

## 安全边界

- source-native depth 只作 source-derived Development label，不是 fresh Confirmation truth；
- source selection 不读 task outcome，`UNKNOWN` 不作 negative，所有 arm 只输出 `OCCUPIED/UNKNOWN`；
- task oracle 仅用于估计可达上限，不是可部署算法；不授权模型、训练、Android、产品或 safety。

## 停止条件

R12 任一 source/pose/calibration binding 无效即停止；少于 48 evaluable references、4 recovery-opportunity
parents 或 4 CLEAR-denominator parents 时，旧 occupancy 臂间 decision 必须为 `null`。R13 不更改该负结论；
它的新任务只授权设计 pose/static-evidence scorer。唯一 successor 由 TARO current 定义。

定向验证：

```powershell
E:\codex-tools\bin\blindassist-python.cmd -m unittest `
  scripts.research.taro_o1r_r12_clear_observability_runtime.test_development_replay `
  scripts.research.taro_o1r_r12_clear_observability_runtime.test_positive_oracle_canary `
  scripts.research.taro_o1r_r12_clear_observability_runtime.test_balanced_pose_source_frontdoor `
  scripts.research.taro_o1r_r12_clear_observability_runtime.test_arkitscenes_balanced_pose_source_frontdoor `
  scripts.research.taro_o1r_r12_clear_observability_runtime.test_tum_balanced_pose_source_frontdoor `
  scripts.research.taro_o1r_r12_clear_observability_runtime.test_task_evidence_oracle_canary
```

动态状态、唯一 successor 与 claim ceiling 以
[`docs/research/taro/README.md`](../../../docs/research/taro/README.md) 为准。
