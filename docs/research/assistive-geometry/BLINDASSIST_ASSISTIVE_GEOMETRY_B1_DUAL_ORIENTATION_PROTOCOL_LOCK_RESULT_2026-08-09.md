# BlindAssist Assistive Geometry B1 dual-orientation protocol lock

状态：`PASS / IMPLEMENTATION_NOT_AUTHORIZED / FORMAL_TRAINING_NOT_AUTHORIZED`

Attempt 1 的单一 portrait 假设已被 pose-only preflight 否决；Attempt 2 在任何 task outcome
打开前冻结 dual-orientation full-FOV tensor 与重新平衡的 DEVELOPMENT identity split，并通过
独立 validator。

当前合同：

- portrait：`1×3×608×448`，2,724 TRAIN frames；
- landscape：`1×3×448×608`，2,076 TRAIN frames；
- orientation-bucketed batch，禁止跨 family crop/pad/rotate；
- K 按 upright rotation 后的对应 full-FOV shape 独立更新 `sx/sy`；
- pooled、portrait、landscape、parent-macro 分开报告，产品决策只看 portrait；
- portrait confidence calibration 仍只有一个 portrait-dominant parent，claim ceiling 保持
  `DEVELOPMENT_ONLY_SINGLE_PORTRAIT_DOMINANT_PARENT`。

机器结果见
[JSON](BLINDASSIST_ASSISTIVE_GEOMETRY_B1_DUAL_ORIENTATION_PROTOCOL_LOCK_RESULT_2026-08-09.json)。

## 唯一 successor

`BLINDASSIST_ASSISTIVE_GEOMETRY_B1_DUAL_ORIENTATION_TARGET_AND_MODEL_IMPLEMENTATION_LOCK`
