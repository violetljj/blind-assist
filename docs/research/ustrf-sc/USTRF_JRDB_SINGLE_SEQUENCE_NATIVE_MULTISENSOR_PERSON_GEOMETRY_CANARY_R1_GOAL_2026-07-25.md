# JRDB single-sequence native multisensor person geometry canary R1（2026-07-25）

状态：`FROZEN_BEFORE_EXECUTION`

权限：`SEEN_DEVELOPMENT_AVAILABILITY_ONLY / DIAGNOSTIC_CEILING`

## 修正原因

R0 的 immutable packet、29/1,350 个 3D-only 缺口与 `FAIL_CLOSED_LABEL_JOIN / VALID` 永久保留。R0 的错误不是数据事实，而是 claim dependency：它把“每条 3D 必须同时存在 2D”设成了 3D-native motion 与 robot-relative geometry 的前置条件。

R1 按 [弹性证据与降级标准 R1](USTRF_ELASTIC_EVIDENCE_AND_DEGRADATION_STANDARD_R1.md) 新建版本，不追溯改判 R0，也不换 sequence/window、不删除任何缺失 unit。

## 唯一问题

在 R0 packet 的 clock、双 PCD、pose/IMU、静态 frame chain 和 immutable reconstruction 已通过的前提下：

1. 全部 source-native 3D object-frame 是否足以计算 robot-relative geometry？
2. 同 `label_id` 的相邻 3D observation 是否足以计算 source-annotation-derived 3D motion？
3. 2D/3D 不共现对 cross-modal identity coverage 造成多大降级？

## 冻结依赖

### Robot-relative 3D geometry

必需：唯一 3D `label_id`、有限 3D center、logical RGB360→base_link 静态变换。2D/RGB projection 是可选角色。

### Source-annotation-derived 3D motion

必需：唯一连续 3D `label_id`、单调 source timestamp、相邻 frame、`<=200ms` gap、bounded `odom -> base_link` pose 与静态链。2D join 不是前置条件。

最低 computational support 沿用 R0 执行前已经冻结的绝对门：至少 32 个 observed frame、31 个合法 adjacent pair、1 条 motion track。不新加百分比 pass 线。

### Cross-modal 2D/3D identity

以 3D-native 和 2D-native 两个 source denominator 分别报告 coverage：

- 3D-only：只让 cross-modal claim 对该 object-frame abstain；仍可进入 3D geometry/motion；
- 2D-only：3D claim 对该 object-frame abstain；
- duplicate/ambiguous ID：只 invalid 受影响 object-frame 和依赖的 track pair；只有污染范围无法定位才升级全局。

交集不得作为 expected denominator，且必须满足 `expected = eligible + abstained + invalid`。

## Source interpolation 边界

JRDB `attributes.interpolated` 必须逐项计数。interpolated 3D label 可以用于验证“官方 source annotation trajectory 是否可计算”，但不增加 direct observation 数量，不支持“直接传感器测得的人体速度/轨迹准确性”。

最大合法文案：

`source-annotation-derived person 3D motion and robot-relative geometry availability`

## 合法终态

1. `INVALID_GLOBAL_INTEGRITY`
2. `NOT_EVALUABLE_3D_GEOMETRY_SUPPORT_INSUFFICIENT`
3. `ANNOTATION_DERIVED_PERSON_GEOMETRY_AVAILABLE_WITH_ABSTENTION`
4. `ANNOTATION_DERIVED_PERSON_GEOMETRY_AVAILABLE_COMPLETE`

overall、逐 claim status 与 authority ceiling 必须分开输出。AVAILABLE 只表示计算可用，不表示方法性能、route risk、event lifecycle、提醒、Android、人体、生产或 selection 通过。
