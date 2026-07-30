# Dual-loop global-motion-compensated target flow R0

状态：`DISCOVERY / DEVELOPMENT_INPUT_ONLY`

This module tests one fixed successor to the rejected REveL ROI-flow source:
estimate a background homography outside all source target ROIs, map target
tracks through that homography, and fit only the residual target similarity
expansion. It uses the already-frozen 13,014-row REveL replay and opens Vicon
truth only in the separate evaluator.

This is retrospective Development on a single burned capture whose truth was
already accessed by predecessor work, with oracle target ROIs. The producer
itself does not read truth, but this is not an outcome-naive one-shot. It is
not runtime detector evidence, independent perception, alert effect, Android
feasibility, generalization, product, or safety evidence.

## 稳定 Interface

```text
produce.py --replay-input ... --image-root ... --output ... --receipt ...
evaluate.py --replay-input ... --producer-output ... --producer-receipt ...
            --truth ... --events ... --output ...
```

## 输出

Producer JSONL、producer receipt 与 evaluator JSON 只写入新的 ignored
`artifacts.local/` namespace；两个阶段都拒绝覆盖既有输出。

## 安全边界

## Frozen algorithm

- one causal prior frame; maximum gap 100 ms;
- background features exclude every previous-frame oracle ROI expanded by 10%;
- background: up to 400 Shi-Tomasi features, LK 21x21/level 2,
  forward-backward error <=1.5 px;
- homography: RANSAC 2 px, at least 20 inliers and 50% inlier fraction;
- target: up to 80 features, at least 8 surviving tracks, at least 2 prior-ROI
  quadrants;
- residual similarity: RANSAC 1.5 px, at least 6 inliers;
- signed rate: `log(residual_similarity_scale) / dt`, positive approaching;
- source admission quality floor: 0.50;
- no parameter/deadband/subgroup search after the current evaluator output was
  opened; predecessor truth exposure keeps this retrospective Development.

The evaluator reuses the existing 0.02/s event deadband, 3-row/50% event
coverage, and 469-event readiness floor: >=60% correct, <=20% wrong-signed,
>=80% evaluable, and >=50% correct in every truth state.

The 469 events are not independent observations: the frozen dependency receipt
contains 159 cross-target overlap pairs and 310 overlap components. This module
reports descriptive counts only; it does not compute dependency-aware
uncertainty or claim statistical significance.

Outputs are new ignored files under `artifacts.local/`. Producer and evaluator
both refuse overwrite. Failure closes this candidate; it does not reopen or
replace consumed LITE R0-R2 claims.

## 停止条件

Any replay hash or denominator drift, incomplete truth-blind receipt, output
identity mismatch, non-finite admitted score, or existing output namespace
stops execution. Failure of the frozen readiness floor closes this source
candidate without threshold or subgroup search.
