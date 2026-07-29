# RCLE periodic self-motion counterfactual R2

状态：`motion-component Stage A complete valid / Stage B contract only / successor formal not consumed`

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
键名全集误写而 `INVALID / HOLD_P1`，不得覆盖或重跑。这里仍没有 formal sequence
runner 或 activation lock。隔离的
`R2_KEYSET_REPAIR_R0` 只把历史键名固定为真实的
`producer_receipt.json`，并加入 generator directory 与正式 receipt 的独占创建
保护；88 条 all-seed record 与 R2 逐字节一致。其只读预检和唯一正式验证均为
14/14、`errors=[]`，P1 终态为
`GENERATOR_GEOMETRY_PASS / EXECUTION_NOT_AUTHORIZED`。

## 当前报告与工作方式

```text
scientific_status: QUALITY_CALIBRATION_PASS
protocol_status: VALID
execution_authority: QMS_R1_SUCCESSOR_FORMAL_AUTHORIZED_ONE_SHOT / NOT_RUN
```

P2 R1 一次性 blur-grid repair 已完成 `5120/5120` 行 response-blind ledger。
最小全局可行 blur 为 `sigma=0.475 px`；它与 hash-bound R0 low-texture
`alpha=0.15` 形成全局 strength pair，终态为
`QUALITY_CALIBRATION_PASS / VALID`。随后轻量 P3 R0 已完成 R3 transport
equivalence、analysis implementation/mutation tests，以及同一 8 个完整
PREFLIGHT identities 的 guarded-host 比较。初始均匀比例外推有误；scheduler
successor 复用静态相同 pose frame、仍逐一运行 601 个冻结 R3 pair，并按真实
formal 组成外推。successor W8 实测 `677.5074 s`、投影 `7.1575 h`，选择 W8；
独立终态为
`PERFORMANCE_QUALIFIED / VALID / P4_NOT_ACTIVATED`。历史 R2 只记录为
`OBSERVED_GEOMETRY_GATES_PASS / CLAIM_NOT_SIGNABLE + INVALID_KEYSET + HOLD_P1`，
不再把 keyset 错误写成几何失败。P1 已关闭：非阻断的命名、receipt 便利或未来漂移
监控进入 backlog，不再创建 P1 版本。P2 R0 保持不可变；R1 后不得自动二次修复、
扩 grid、换 seed、降门或改成 per-block strength，且没有自动 P3 后继。

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

当前 P1 通过版本也只供只读审计，不得重跑或覆盖：

```text
artifacts.local/evidence/rcle_periodic_self_motion_counterfactual_r2/p1_geometry_r2_keyset_repair_r0/
```

正式 receipt SHA-256 为
`95646437fbe0ef0cf03844f94467303f5d90ca15c3e22fc1785157b037a8c079`；
implementation lock SHA-256 为
`a7fa41c0406908baf05805904111ba43fdbd8dd93b8c4e496706f1990438adc9`。

P2 R0 只读复核：

```powershell
E:\codex-tools\bin\blindassist-python.cmd -m `
  scripts.research.egomotion_compensated_looming.periodic_self_motion_counterfactual_r2.validate_quality_calibration_independent_r0 `
  --preflight
```

正式 independent receipt 已 exclusive-create，不得覆盖或创建第二份正式
receipt。

P2 blur-grid repair R1 只读复核：

```powershell
E:\codex-tools\bin\blindassist-python.cmd -m `
  scripts.research.egomotion_compensated_looming.periodic_self_motion_counterfactual_r2.validate_quality_calibration_blur_grid_repair_independent_r1 `
  --preflight
```

R1 lock、ledger 与正式 independent receipt 已冻结，不得覆盖或重跑 producer。

权威结果：

- [R1 不可变失败与 source-hash 竞态](../../../../docs/research/rcle/RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_GEOMETRY_SPEC_REPAIR_R1_RESULT_2026-07-28.md)
- [R2 唯一冻结运行结果](../../../../docs/research/rcle/RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_GENERATOR_GEOMETRY_IMPLEMENTATION_R2_RESULT_2026-07-29.md)
- [P1 keyset-repair 通过结果](../../../../docs/research/rcle/RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_GENERATOR_GEOMETRY_KEYSET_REPAIR_R0_RESULT_2026-07-29.md)
- [P2 response-blind quality calibration R0](../../../../docs/research/rcle/RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_QUALITY_CALIBRATION_R0_RESULT_2026-07-29.md)
- [P2 blur-grid repair R1](../../../../docs/research/rcle/RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_QUALITY_CALIBRATION_BLUR_GRID_REPAIR_R1_RESULT_2026-07-29.md)
- [P3 transport/analysis/runtime preflight R0](../../../../docs/research/rcle/RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_TRANSPORT_ANALYSIS_RUNTIME_PREFLIGHT_R0_RESULT_2026-07-29.md)
- [P4 formal pre-R3 terminal](../../../../docs/research/rcle/RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_P4_FORMAL_RESULT_2026-07-29.md)

失败模式：

- JSON、依赖文件或 SHA-256 不匹配；
- 2×3 arm、四 block、20 seed/block、480 sequence 或 80 cluster 漂移；
- R3 `0.01/s`、三 pair、reset 或 implementation identity 漂移；
- geometry required gate、统计支持/排除规则、terminal precedence 或 budget 漂移；
- P4 activation 之前任一当前文档把 `formal_execution_authorized` 设为 true；已签发并
  消费的一次性 activation lock 是唯一例外，且 manipulation failure 后不得再启动
  formal R3。

## 输出

validator 默认只向 stdout 输出 compact JSON。keyset-repair 的正式 receipt 只允许
在只读预检通过后以 exclusive-create 写入一次；现有 evidence 和 receipt 均不得
重跑或覆盖。未来生成器和正式证据只能位于：

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
`EXECUTION_NOT_AUTHORIZED`。当前隔离 keyset-repair 已关闭 P1 geometry；P2 R1
已把 response-blind quality calibration 关闭为
`QUALITY_CALIBRATION_PASS / VALID`；P3 R0 scheduler successor 已关闭为
`PERFORMANCE_QUALIFIED / VALID / P4_NOT_ACTIVATED`。其后旧 P4 pre-R3 terminal
已消费；QMS-R1 successor activation preflight 现为
`VALID / FORMAL_NOT_RUN / AUTHORIZED_ONE_SHOT`。任何
geometry、response-blind calibration、R3 transport equivalence、analysis lock
或 guarded-host preflight 失败，都不得靠换 seed、降门、减 arm 或继续切 ADVIO
回救。

P4 one-shot activation 后，正式 main manipulation check 已完成
`160` 个 cluster×motion sequence check。blur 为八个 subgroup 全部 `20/20`；
low-texture 在 `ADVIO_13 periodic=17/20`、`ADVIO_15 periodic=14/20`、
`ADVIO_17 static=17/20`、`ADVIO_17 periodic=17/20` 失败。因此当前最终终态为
`INTERVENTION_NOT_EVALUABLE / VALID / COMPLETE_PRE_R3_TERMINAL`，正式 R3
arm、pair-core call 和 outcome analysis 均为零。

在不改 R3、也不读取 response 的独立后继中，固定 bilateral `QMS-R0` 已先在
旧 scenes 上实测并淘汰（八个 subgroup 均 `0/20`）。`QMS-R1` 改为一次共享
raycast 后在 prequantization linear RGB 域收缩材质内部 residual；hard gate
只度量该 frozen estimand，全帧梯度改为描述项。它在旧 development
identities 上 `160/160`、全新 disjoint CAL 上 `32/32` 通过，独立 validator
复算 `32 sequences / 512 frames / 8 subgroups` 为 `VALID`，mutation tests
`11/11 PASS`。随后独立 activation preflight R0 已冻结 QMS-R1 operator、新且
全域不相交的 480+16 identity lock，并用全新固定 8 条完成 W8。G01–G14、R3
transport 和 analysis lock 复验均未漂移，按 shared-render cluster scheduler
含 10% reserve 投影为 `11.3375 h`。终态为
`ACTIVATION_PREFLIGHT_PASS / VALID / FORMAL_NOT_RUN`，已授权 exact-lock 一次性
successor formal execution；本阶段正式 sequence 与 R3 formal call 仍为零。

- [QMS-R1 qualification result](../../../../docs/research/rcle/RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_QUALITY_MANIPULATION_SUCCESSOR_R1_RESULT_2026-07-29.md)
- [QMS-R1 activation preflight R0](../../../../docs/research/rcle/RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_QMS_R1_FORMAL_ACTIVATION_PREFLIGHT_R0_RESULT_2026-07-29.md)
- [QMS-R1 four-block DEV diagnostic R0](../../../../docs/research/rcle/RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_QMS_R1_FOUR_BLOCK_DEV_DIAGNOSTIC_R0_RESULT_2026-07-29.md)

activation 后没有直接运行正式 480+16。隔离的 four-block DEV diagnostic
另冻 8 个全新 scene clusters，每个 cluster 完整运行六臂和 601 pairs，共
48 sequences。W8 约 `37.8 min` 完成；独立 validator 从 ledgers 重算后为
`VALID / DEV_DIAGNOSTIC_COMPLETE`。clean 周期自运动相对静态的 trigger-density
contrast 在 8/8 clusters 为正，均值 `0.25`、范围 `0.18–0.29`。这只形成
controlled-generator DEV 诊断，不形成 max-t、总体效应或产品/安全结论。
正式一次性授权保持 `UNCHANGED_NOT_CONSUMED`。

## Clean motion-component localization R0

Stage A 已运行两批 frozen clean 四臂
`STATIC / ROTATION_ONLY / TRANSLATION_ONLY / FULL_6DOF`。全部 32 identities
在执行前冻结；ordinal `0` 与 ordinal `1` 各为
`4 clusters / 16 sequences / 9,616 pairs`。

独立 validator 从每 pair 的 raw/compensated 九格 cell primitives 重建 signed
与 absolute 指标；两批结果为：

- rotation-only 相对 static 的 absolute P90：`4/4 → 4/4` 正；
- translation-only 相对 rotation-only 的 signed P90：`4/4 → 4/4` 正；
- full-6DoF 相对较大 single arm 的 signed P90：`2/4 → 3/4` 正，跨批不稳定。

第三个独立 closeout validator 已给出 `VALID / STAGE_A_COMPLETE`。下一步只授权
准备和冻结 `translation-depth oracle + object-approach control` 的 Stage B
合同；不授权 B 执行、算法修改、interaction 分支、C/D 或 formal。rotation
absolute leakage 作为必须携带的机制审计边界。正式 480+16 仍为零，一次性
authority 未消费。

- [Stage 1 result](../../../../docs/research/rcle/RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_QMS_R1_CLEAN_MOTION_COMPONENT_LOCALIZATION_R0_STAGE_1_RESULT_2026-07-29.md)
- [Stage A result](../../../../docs/research/rcle/RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_QMS_R1_CLEAN_MOTION_COMPONENT_LOCALIZATION_R0_STAGE_A_RESULT_2026-07-29.md)

## 假设与规则质疑

480 条只是 80 个配对 cluster。20 seed/block 是当前固定最小预算，不是 power
保证；若 response-blind precision 预检认为不足，只能在任何 formal output 前另立
版本。五个 confirmatory contrast 使用 familywise max-t interval，避免多重比较和
逐帧伪样本。

## 失败资产复用

失败的 geometry fixture、calibration panel、pairing manifest 和 validator
mutation 可保留为 regression/counterexample。它们不得被重新包装成 unseen natural
evaluation 或 confirmation。
