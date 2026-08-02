# HFTF Stage C D43.1 JRDB track-only metric residual student result

日期：2026-08-03（Asia/Hong_Kong）

## 结论

终态：

`D43_1_JRDB_TRACK_ONLY_METRIC_RESIDUAL_STUDENT_NOT_SUPPORTED`

D43.1 完整可评价，但冻结 support gates 0/6 通过。相对 zero residual：

- pooled teacher vector error：`0.80238 -> 1.10533 m`，恶化 `37.76%`
- pooled actual future vector error：`0.80935 -> 1.11648 m`，恶化 `37.95%`
- actual future error better fraction：`22.370%`
- teacher/actual 只有 Meyer Green 1/4 folds 改善
- STLC actual error：`0.34503 -> 0.92915 m`，恶化 `169.30%`

因此 D42 的强 metric teacher ceiling 不能由当前十个 2D track statistics 的固定
linear Ridge 跨 sequence 恢复。

## folds

| held-out sequence | actual zero | track-only | actual reduction | better fraction | teacher reduction |
|---|---:|---:|---:|---:|---:|
| Clark Center | 1.52374 m | 1.67991 m | -10.25% | 33.09% | -9.25% |
| Gates Basement | 0.55884 m | 0.71614 m | -28.15% | 29.44% | -26.87% |
| Meyer Green | 0.95938 m | 0.82465 m | +14.04% | 63.98% | +17.63% |
| STLC 111 | 0.34503 m | 0.92915 m | -169.30% | 3.69% | -177.57% |

该异质性不是 pooled gate 的边缘失败；模型在三个 unseen sequences 上方向一致地
劣于预测零位移。

## 实现边界

- exact 3,384 opportunities / 53 identities
- 4-fold leave-one-sequence-out
- 每折严格 3 train sequences / 1 held-out sequence
- 10 frozen detector-track features
- population standardization + closed-form multi-output
  `Ridge(alpha=1.0, fit_intercept=True)`
- 每折 20 coefficients，无随机性
- training target 仅为 D42 history-only teacher residual
- actual future truth 只用于 held-out evaluation

全部 evaluability gates 通过。

## 科学解释

D33 的 selective range-direction mechanism、D41 的局部 image translation signal 和
D42 的 metric teacher ceiling 都保持成立。D43.1 拒绝的是：

> current box state + 7-frame first-order image-space slopes + confidence，经过固定
> linear cross-sequence mapping，就足以恢复 metric xy residual。

不能在同一 outcome 上通过改 alpha、特征子集、target normalization、非线性模型或
删 STLC 来救结果。

下一步必须增加新的可观测量，而不是增加模型容量：

1. 招募 IMU 完整的多 sequence source，重新评价原 D43；或
2. 引入 causal metric-depth/ground geometry measurement；
3. 新 source 上仍先检验 residual measurement，再进入任何 event/alert seam。

## 复现

- D42/D43/D43.1 focused tests：4 PASS
- report 连续重建 SHA 稳定：
  `d104279a42a8089a171ca4fcab4db7c85e0004f1f201ee51f1667bd9dbadcd23`
- report size：8,005 bytes

## claim ceiling

`RESEARCH_MAINLINE_UNCHANGED / DEFAULT_APP_UNCHANGED`

D43.1 不建立 track-only learnability、IMU 结论、event utility、Android runtime、
主线、产品或安全主张，也不覆盖 D35 device gate。
