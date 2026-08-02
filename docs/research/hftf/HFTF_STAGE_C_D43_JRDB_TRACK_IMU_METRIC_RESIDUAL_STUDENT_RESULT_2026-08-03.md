# HFTF Stage C D43 JRDB track-IMU metric residual student result

日期：2026-08-03（Asia/Hong_Kong）

## 结论

终态：

`D43_JRDB_TRACK_IMU_METRIC_RESIDUAL_STUDENT_NOT_EVALUABLE`

原因：

`IMU_SEQUENCE_COVERAGE_INSUFFICIENT`

D43 在任何 model fit 或 future-outcome evaluation 前停止：

| sequence | frames | IMU-complete frames | contiguous track histories | IMU-complete histories |
|---|---:|---:|---:|---:|
| Clark Center | 120 | 120 | 1,304 | 1,304 |
| Gates Basement | 120 | 0 | 1,079 | 0 |
| Meyer Green | 120 | 120 | 194 | 194 |
| STLC 111 | 120 | 0 | 1,661 | 0 |

Gates 缺 orientation；STLC 缺 orientation、angular velocity 与 linear
acceleration。冻结合同要求四个 held-out folds 每折至少 50 个完整 opportunities，
因此 `TRACK_IMU` candidate 不可评价。

## 非算法终态

- `model_training_executed=false`
- `future_outcome_evaluated=false`
- 没有填零、插值、删 sequence 或降为两折
- 没有 `TRACK_IMU` learnability 正负结论

该终态只说明当前四个 JRDB packets 不足以评价跨 sequence track+IMU student，
不说明 IMU 无效。

## 后继边界

原冻结协议已经包含等容量 `TRACK_ONLY` arm，而且四个 sequences 的 detector-track
history 都充足。允许单独冻结 D43.1：

- 保持同一 10-feature allowlist
- 保持同一 D42 teacher target
- 保持同一四折 leave-one-sequence-out
- 保持相同 zero-baseline effect floors
- 不使用任何 IMU 或缺失填补

D43.1 是 adaptive Development，不回填 D43 的 IMU 主张。

## 复现

- report SHA-256：
  `8825eddbeab4f4e4e7965d1f9982fbf7a65ab587967520c99c730eb5a479ed1e`
- report size：2,245 bytes
- pose/center parity maximum error：
  `1.1368683772161603e-13 m`
- D43 evaluator unit test：PASS

## claim ceiling

`RESEARCH_MAINLINE_UNCHANGED / DEFAULT_APP_UNCHANGED`

D43 不建立 student learnability、event utility、Android runtime、主线、产品或安全
主张，也不覆盖 D35 device gate。
