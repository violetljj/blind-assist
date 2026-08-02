# HFTF Stage C D11–D13：真正未来 onset 的任务修正与时序基线

日期：2026-08-02

证据角色：Development / estimand repair + representation baseline

研究主线：不变

默认 App：不变

## 结论

D8–D10 的主要问题不只在模型。原 `future_proximity` / `future_corridor` 标签从
`t=0` 开始，在 0–2 秒内做 `any-risk` 聚合；因此当前已经危险的样本必然是未来正例，
清除事件在定义上不可能出现。D11 进一步证明，current-static QTM geometry 对原标签
已经有约 `.89–.97` 的五折 AUROC，而因果历史速度外推没有稳定增量。原任务主要测量
当前占用，不是 HFTF 要回答的未来 onset。

D12 因而只保留当前几何安全的样本，重新定义“未来 2 秒首次进入危险”。修正后仍有
足够的五折监督：

| onset target | eligible | positive | negative |
|---|---:|---:|---:|
| 近距 | 530 | 157 | 373 |
| 走廊 | 616 | 148 | 468 |

每折两个 target 都有正负例，current-positive / future-negative monotonicity
violation 为 0。

D13 在这个真正未来 onset 上重跑等容量 frozen-spatial baseline。history 相对
current 的 seed-mean fold delta：

| 指标 | mean | median | 正折 | 正 unit |
|---|---:|---:|---:|---:|
| 近距 AUROC | +0.00077 | +0.00180 | 4/5 | 9/15 |
| 近距 AP | +0.00203 | +0.00428 | 3/5 | 9/15 |
| 走廊 AUROC | +0.00196 | +0.00198 | 5/5 | 10/15 |
| 走廊 AP | -0.00046 | +0.00192 | 4/5 | 11/15 |

按运行前冻结的“median > 0 且至少 3/5 folds 为正”门，终态为：

`D13_FUTURE_ONSET_TEMPORAL_SPATIAL_INCREMENT_SUPPORTED`

这是很弱的 representation 正信号，不是可用模型。corridor AP mean 仍为负，
absolute unit means 仅为近距 AUROC/AP `.569/.392`、走廊 `.519/.306`。它支持下一步
测试显式运动表示；不支持主线切换、提醒行为、App 或安全主张。

## D11：监督信息上限诊断

D11 比较两个不训练的 QTM geometry arms：

- current-static：当前相对位置在未来 2 秒保持不变；
- causal-history：只用锚点前 0.8 秒拟合参与者相对速度，再做恒速外推。

两臂共享原标签构造使用的 forward coordinate frame。这个 frame 是为隔离时序信息而
提供的共同 oracle，不是可部署输入。6,290 个 current body observations 中 5,763 个
有合格历史速度。

history-minus-current：

| 指标 | mean | 正折 |
|---|---:|---:|
| 近距 AUROC | +0.00751 | 3/5 |
| 近距 AP | -0.00388 | 2/5 |
| 走廊 AUROC | +0.00052 | 2/5 |
| 走廊 AP | -0.01086 | 0/5 |

终态：

`D11_CAUSAL_KINEMATIC_HISTORY_INFORMATION_NOT_SUPPORTED`

这个结果与 D10 一起表明：继续搜索 RGB 时序架构不能修复一个被当前状态主导的标签。

## D12：future-onset 任务修正

对每个 target 分别定义：

```text
eligible = current geometric state is safe
positive = eligible and original 0–2 s future-ever target is true
negative = eligible and original 0–2 s future-ever target is false
```

不同 target 使用各自 eligibility mask；不会把近距当前危险样本错误地从走廊任务中
移除。五折正例数分别为：

| fold | 近距正/负 | 走廊正/负 |
|---:|---:|---:|
| 0 | 48 / 87 | 40 / 138 |
| 1 | 49 / 111 | 38 / 85 |
| 2 | 25 / 74 | 22 / 74 |
| 3 | 12 / 16 | 14 / 47 |
| 4 | 23 / 85 | 34 / 124 |

终态：

`D12_FUTURE_ONSET_TARGET_FIVE_FOLD_READY`

## D13：等容量时序基线

- frozen MobileNet `5×576×4×7` spatial maps；
- current arm 重复 current map，history arm 使用真实五帧；
- 两臂都是同一个 13,586-parameter temporal-spatial head；
- seeds `17/23/41`、120 epochs、五折 source-session isolation；
- target-masked、source-balanced BCE；
- fixed final epoch，不用 held-out 选择模型或 threshold。

修正前 D8 spatial head 的近距/走廊四项正折数为 `2/1/5/5`；修正后 D13 为
`4/3/5/4`。这支持“正确隔离 future onset 后，history 的方向一致性改善”，但效应
仍太小，不能声称当前 frozen MobileNet 已经解决未来预测。

## 可复现证据

```text
artifacts.local/evidence/hftf/
  stage-c-d11-thor-magni-kinematic-information-ceiling-v0/report.json
  stage-c-d12-thor-magni-future-onset-v0/
    samples.jsonl
    report.json
  stage-c-d13-thor-magni-future-onset-temporal-baseline-v0/report.json
```

- D11 report SHA-256：
  `1a65a2b7d0316ea5e7594aebcabdb389b4cfa7bba308196458738b0ff0a2ec5f`
- D12 samples SHA-256：
  `9a099a52d29da60f889d40cacc1a2e267e506c23dc4aafa7fba1764eb1d64a54`
- D12 report SHA-256：
  `6dca6d4ef9c6e3b165856754e8bbe9bb0bfe09fe5eef9070db46981cdbf68d5f`
- D13 frozen spatial features SHA-256：
  `9a80d6ca6f3b36aee3efed91f89802ebd7e5f9a972cca226175644bd55135838`
- D13 report SHA-256：
  `c56e6798889e24a39899c9be32b29deecb349086805a477fefd8199a0ac3a7cf`

## 下一科学变量

下一步不再调整 frozen residual head，而在同一 D12 onset estimand 上测试显式运动表示
（例如 frozen optical flow / correspondence）。只有它相对 D13 current baseline
产生实质且跨折稳定的增量，才值得进入轻量 ConvGRU/3D temporal student。任务修正
本身保留，不因后续模型成败退回原来的 current-dominated future-ever 标签。
