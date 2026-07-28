# RCLE periodic self-motion counterfactual R2

状态：`P1 R2 frozen / INVALID / INTERVENTION_NOT_EVALUABLE / HOLD_P1`

## 研究问题与版本

`RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2` 是 DEVELOPMENT 阶段的受控
2×3 反事实：在非平面 3D 场景中独立操纵 endpoint-closed 周期性 6DoF camera
self-motion 与 clean/blur/low-texture，判断 unchanged R3 高触发密度来自 motion、
quality 或 interaction。

P1 已实现 deterministic analytic 3D generator、解析 fixture、all-seed geometry
manifest 与独立 validator。R0 因 G13 estimand 冲突为 13/14；R1 因
angle/acos 数值判定为 13/14，并保留不可变失败回执。R2 版本化加固
G01/G02/G03/G08/G11/G12/G14、固定 source-component G13 判定并增加 8 个
GUARD 双构建 replay。R2 的 G01–G14 全部 PASS，但正式 receipt 因 R0 evidence
键名全集误写而 `INVALID / HOLD_P1`，不得覆盖或重跑。这里仍没有 RCLE runner、
quality calibration、formal sequence runner、analysis producer 或 activation lock。

## 稳定 Interface

从仓库根目录只读验证当前三份冻结 JSON：

```powershell
E:\codex-tools\bin\blindassist-python.cmd `
  scripts\research\egomotion_compensated_looming\periodic_self_motion_counterfactual_r2\validate_freeze.py
```

物化 P1 geometry（不运行 RCLE、P2、P3 或 P4）：

```powershell
E:\codex-tools\bin\blindassist-python.cmd `
  scripts\research\egomotion_compensated_looming\periodic_self_motion_counterfactual_r2\generator_geometry.py
```

独立复算 G01–G14：

```powershell
E:\codex-tools\bin\blindassist-python.cmd `
  scripts\research\egomotion_compensated_looming\periodic_self_motion_counterfactual_r2\validate_geometry_independent.py `
  --receipt artifacts.local\evidence\rcle_periodic_self_motion_counterfactual_r2\p1_geometry_r0\independent_geometry_validation_receipt.json
```

独立 validator 不导入 producer 或 RCLE。当前预期返回码为 2，并写出合法
`INTERVENTION_NOT_EVALUABLE / HOLD_P1` 回执；不得把它改写成 warning。

R2 冻结证据仅供只读审计，不得重跑：

```text
artifacts.local/evidence/rcle_periodic_self_motion_counterfactual_r2/p1_geometry_r2/
```

正式 R2 receipt SHA-256 为
`75899978919b67be260bbba1161d69ea09b42384f1730ec866a243f6d0f41a32`；
14 项门均 PASS，但顶层 `INVALID` error 为
`R0_RECEIPT_EVIDENCE_HASH_KEYSET`。

权威结果：

- [R1 不可变失败与 source-hash 竞态](../../../../docs/research/rcle/RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_GEOMETRY_SPEC_REPAIR_R1_RESULT_2026-07-28.md)
- [R2 唯一冻结运行结果](../../../../docs/research/rcle/RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_GENERATOR_GEOMETRY_IMPLEMENTATION_R2_RESULT_2026-07-29.md)

失败模式：

- JSON、依赖文件或 SHA-256 不匹配；
- 2×3 arm、四 block、20 seed/block、480 sequence 或 80 cluster 漂移；
- R3 `0.01/s`、三 pair、reset 或 implementation identity 漂移；
- geometry required gate、统计支持/排除规则、terminal precedence 或 budget 漂移；
- 任一当前文档把 `formal_execution_authorized` 设为 true。

## 输出

validator 只向 stdout 输出一个 compact JSON，并且不写文件。未来生成器和正式证据
只能位于：

```text
artifacts.local/datasets/rcle_periodic_self_motion_counterfactual_r2/
artifacts.local/evidence/rcle_periodic_self_motion_counterfactual_r2/
```

## 安全边界

- 不读取或运行 RCLE output；
- 不访问 ADVIO sequence16；
- 不修改 R3、Sparse LK、support manager、`0.01/s`、三-pair 或 PairState；
- 不运行 CoTracker、RGB formal experiment、Android 或 realtime；
- synthetic mechanism evidence 不是自然视频 false-alert、gait、obstacle、risk、
  product 或 safety evidence。

## 停止条件

静态 bundle 或独立设计审查不通过时停在
`EXECUTION_NOT_AUTHORIZED`。当前 R2 hash/keyset validator 已失败，终态为
`INTERVENTION_NOT_EVALUABLE / HOLD_P1`。任何 geometry、response-blind calibration、R3
transport equivalence、analysis lock 或 guarded-host preflight 失败，都不得靠换
seed、降门、减 arm 或继续切 ADVIO 回救。

## 假设与规则质疑

480 条只是 80 个配对 cluster。20 seed/block 是当前固定最小预算，不是 power
保证；若 response-blind precision 预检认为不足，只能在任何 formal output 前另立
版本。五个 confirmatory contrast 使用 familywise max-t interval，避免多重比较和
逐帧伪样本。

## 失败资产复用

失败的 geometry fixture、calibration panel、pairing manifest 和 validator
mutation 可保留为 regression/counterexample。它们不得被重新包装成 unseen natural
evaluation 或 confirmation。
