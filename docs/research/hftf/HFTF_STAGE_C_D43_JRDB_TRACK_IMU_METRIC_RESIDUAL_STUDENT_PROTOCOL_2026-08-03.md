# HFTF Stage C D43 JRDB track-IMU metric residual student protocol

日期：2026-08-03（Asia/Hong_Kong）

状态：

`STAGE_C_D43_JRDB_TRACK_IMU_METRIC_RESIDUAL_STUDENT_FROZEN`

证据角色：Development / sequence-held-out teacher-distillation canary

## 科学问题

D42 已证明 causal metric teacher 相对 current-static 的 future relative geometry
有强、跨 sequence 增量，且 person world motion contribution 明显大于 ego-only。
D43 不训练风险头、不接入 alert，只检验：

> 一个严格解释、轻量、phone-causal 的 detector-track + IMU student，能否在
> unseen sequence 上恢复 D42 teacher 的水平 metric displacement residual，并对
> source-recorded future residual 保持真实增量？

## cohort 与隔离

- 复用 D42 exact admissible opportunities
- history：连续 7 frames
- future evaluation：`+15 frames`
- outer split：4-fold leave-one-sequence-out
- scaler 与 model 只在 3 个 training sequences 拟合
- held-out sequence 不参与 feature scaling、coefficient、threshold 或选择

D43 是看到 D42 后的 adaptive Development，不是 fresh Confirmation。

## inference feature allowlist

不得使用 native identity、pose、3D center、future frame 或 future truth。

`TRACK_ONLY` 固定 10 features：

1. current normalized center x/y
2. current log normalized width/height
3. 7-frame OLS slopes：normalized center x/y、log width/height
4. current detector confidence
5. history mean detector confidence

`TRACK_IMU` 在相同 10 features 上固定增加：

6. history mean IMU angular velocity x/y/z
7. history mean IMU linear acceleration x/y/z
8. current IMU orientation yaw 的 sin/cos

共 18 features。缺失、非有限或时间不连续时该 opportunity 不可用于 student；
不得插值或 outcome 后填补。

## teacher target 与真实评价

训练 target：

`D42 EGO_OBJECT_KINEMATIC predicted future center_base_link_xy`
`- current center_base_link_xy`

该 teacher residual 只由 current/history native geometry 生成，不读 future。

held-out evaluation 同时报告：

- teacher residual vector error
- source-recorded future displacement vector error

zero baseline 固定预测 `[0, 0]`。`TRACK_ONLY` 是机制基线；`TRACK_IMU` 是候选。

## 固定模型

两个 arms 使用完全相同：

- `StandardScaler`
- multi-output `Ridge(alpha=1.0, fit_intercept=True)`
- 无随机种子、无 early stopping、无 nested search

不得搜索 alpha、feature subset、polynomial、loss、model family、fold 或 target
normalization。

## evaluability

必须全部满足：

- D42 binding 与 cohort census 完整
- `>=400` opportunities / `>=15` identities
- 4 folds 每折 `>=50` held-out opportunities
- 每折 training sequences = 3、test sequence = 1
- 所有 features、targets、coefficients、predictions、metrics finite

否则：

`D43_JRDB_TRACK_IMU_METRIC_RESIDUAL_STUDENT_NOT_EVALUABLE`

## support gates

`TRACK_IMU` 必须全部满足：

1. pooled teacher-vector-error 相对 zero 降低 `>=20%`
2. pooled actual-future-vector-error 相对 zero 降低 `>=10%`
3. actual-future-error better fraction `>=55%`
4. teacher error 在至少 3/4 folds 改善
5. actual future error 在至少 3/4 folds 改善
6. 任一 fold 的 actual future mean error不得相对 zero 恶化超过 `5%`
7. `TRACK_IMU` actual future mean error 相对 `TRACK_ONLY` 非劣：
   relative delta `<=+5%`

通过：

`D43_JRDB_TRACK_IMU_METRIC_RESIDUAL_STUDENT_SUPPORTED_DEVELOPMENT_ONLY`

否则：

`D43_JRDB_TRACK_IMU_METRIC_RESIDUAL_STUDENT_NOT_SUPPORTED`

IMU 是否产生独立增量只按 `TRACK_IMU - TRACK_ONLY` 披露；noninferiority 通过不等于
IMU mechanism supported。

## stopping 与 claim ceiling

看到结果后不得在同一 outcome 上删 IMU、改 alpha、加 nonlinear model、筛
sequence 或换 teacher target 来救结果。

支持只证明离线、sequence-held-out 的轻量 residual learnability，允许下一步冻结
production-shaped shadow feature parity；不证明 Android runtime、event utility、
主线、产品或安全效果。

`RESEARCH_MAINLINE_UNCHANGED / DEFAULT_APP_UNCHANGED`

D43 不覆盖 D35 device gate。
