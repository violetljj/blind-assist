# HFTF Stage C SANPO F0.1 teacher-opportunity execution contract

日期：2026-08-01

状态：`FROZEN_BEFORE_FIRST_F0_1_TEACHER_GEOMETRY_OUTCOME`

这份执行合同在读取首个 F0.1 teacher geometry outcome 之前，消除父协议中
“future label unions observations”的实现歧义。它不改变 F0/F0.1 的阈值、来源、
角色或终态，只把冻结语义写成可执行断言。

## 1. 因果时间与场

- 5 FPS 来源的可用 anchor 为目标时间线 `4..22`，共 19 个；10 FPS 来源为
  `8..20`，共 13 个。
- current 场使用 anchor 的 local-ground projection 与 anchor yaw。
- future 场的 origin 只由 `anchor` 与精确 `-0.4s` 历史 pose 算出的地面切向
  速度推进 `+0.4s`；方向仍是 anchor yaw。
- future pose 只能把 `+0.4s` 观测反投影到 world 并检查同一 world probe，
  不能选择 origin、方向、anchor 或样本。

## 2. 冻结 teacher union

- current support 只来自 anchor observation。
- future obstacle support 对 anchor observation 与精确 `+0.4s` observation
  分别在同一 causal future field 中计数，再逐 cell 取最大值。这样实现
  observation union，同时避免同一表面跨时刻重复采样把阈值 2 人为抬高。
- future known 对每个 world probe 做 observation union：anchor 或 future
  任一观测通过，该 probe 即通过；9 个 probes 中至少 5 个通过才是 KNOWN。
- UNKNOWN 永远不转成 SAFE。

## 3. 两个 teacher view 与角色门

- candidate：stride 8 / offset 4；
- reference：stride 4 / offset 2；
- 两个 pixel lattices 不相交。

所有来源同时计算两个 view，防止实现分叉；但冻结 gate view 为：

- train：candidate；
- dev：reference；
- heldout：reference。

另一个 view 只作预声明诊断，不能改变角色 gate view、阈值或来源。
每个 role/height 的“具有 positive future cells 的 source”定义为：其角色
gate view 至少有 1 个 positive-known future cell。

## 4. 唯一停止规则

必须同时满足父 F0 的所有 pretraining gates，并且第二次 payload 字节一致，
才能到 `F0_1_SANPO_TEACHER_OPPORTUNITY_READY_FOR_CORPUS`。任一失败即
`F0_1_SANPO_CROSS_SPLIT_BODY_HEAD_STUDENT_CANARY_NOT_EVALUABLE`，禁止
materialize corpus 或训练学生。

byte-exact 比较覆盖加入 determinism 字段前的完整科学 payload，使用
UTF-8 JSON、`sort_keys=true`、紧凑 separators、禁止 NaN；payload 不含时间戳或
随机 run ID。

本阶段只输出汇总计数和冻结 denominator；不落 teacher cell corpus、不读取
RGB student input、不训练、不打开 heldout student output，也不改变研究主线、
Android、生产或安全权限。

已封口的 source-authority cohort 是首次读取 heldout teacher opportunity 的
ordered gate。heldout 汇总只能决定“是否可训练”，不能选择阈值、checkpoint、
augmentation 或来源；heldout RGB student output 仍未打开。
