# HFTF Stage C D43.1 JRDB track-only metric residual student protocol

日期：2026-08-03（Asia/Hong_Kong）

状态：

`STAGE_C_D43_1_JRDB_TRACK_ONLY_METRIC_RESIDUAL_STUDENT_FROZEN`

证据角色：Development / sequence-held-out teacher-distillation canary

## 目的

D43 因 2/4 sequences 没有完整 IMU，在训练前 `NOT_EVALUABLE`。D43.1 不填补
IMU，不删除 sequence，只执行 D43 已冻结的 `TRACK_ONLY` arm：

> 7-frame detector-track statistics 能否在 unseen sequence 上恢复 D42
> history-only metric displacement teacher，并对 actual future residual 保持增量？

## 不变合同

- exact D42/D43 cohort construction
- 4-fold leave-one-sequence-out
- target：D42 `EGO_OBJECT_KINEMATIC - CURRENT_RELATIVE_STATIC` 的 xy residual
- actual evaluation：source-recorded `+15 frames` relative xy displacement
- model：population `StandardScaler` 等价实现 +
  multi-output `Ridge(alpha=1.0, fit_intercept=True)`
- zero baseline：固定 `[0, 0]`

## 固定 10 features

1. current normalized center x/y
2. current log normalized width/height
3. 7-frame OLS slopes：normalized center x/y、log width/height
4. current detector confidence
5. history mean detector confidence

inference 禁止 native identity、pose、3D center、IMU、future frame 或 future truth。

## evaluability

- `>=400` opportunities / `>=15` identities
- 4 folds 每折 `>=50` held-out opportunities
- 每折严格 3 train sequences / 1 test sequence
- features、teacher targets、coefficients、predictions、metrics 全 finite

否则：

`D43_1_JRDB_TRACK_ONLY_METRIC_RESIDUAL_STUDENT_NOT_EVALUABLE`

## support gates

全部满足：

1. pooled teacher-vector-error 相对 zero 降低 `>=20%`
2. pooled actual-future-vector-error 相对 zero 降低 `>=10%`
3. actual-future-error better fraction `>=55%`
4. teacher error 在至少 3/4 folds 改善
5. actual future error 在至少 3/4 folds 改善
6. 任一 fold actual future mean error不得相对 zero 恶化超过 `5%`

通过：

`D43_1_JRDB_TRACK_ONLY_METRIC_RESIDUAL_STUDENT_SUPPORTED_DEVELOPMENT_ONLY`

否则：

`D43_1_JRDB_TRACK_ONLY_METRIC_RESIDUAL_STUDENT_NOT_SUPPORTED`

## stopping 与 claim ceiling

不得在同一 outcome 上加非线性、改 alpha、feature subset、target normalization、
history/horizon 或 sequence。支持只证明离线 track-only residual learnability；
不回填 D43 的 IMU 结论，不建立 Android/event/主线/产品/安全主张。

`RESEARCH_MAINLINE_UNCHANGED / DEFAULT_APP_UNCHANGED`
