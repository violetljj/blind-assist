# HFTF Stage C D26：THOR-MAGNI counterfactual collision field result

日期：2026-08-03

证据角色：Development / action-conditioned future-field canary

研究主线：不变

默认 App：不变

## 结论

D26 完整产生 5 folds × current/history = 10 个训练 runs。冻结 gate 通过 7/11，
总体终态为：

`D26_THOR_MAGNI_COUNTERFACTUAL_COLLISION_FIELD_INCREMENT_NOT_SUPPORTED`

history 相对 current 的主要结果：

| metric | mean delta | 正折 |
|---|---:|---:|
| source-macro direction×horizon AUROC | -0.00051 | 2/5 |
| source-macro direction×horizon AP | +0.00434 | 3/5 |
| source-macro safest-direction accuracy | +0.00541 | 3/5 |
| pooled direction×horizon AUROC | -0.00129 | 3/5 |
| pooled direction×horizon AP | -0.00314 | 2/5 |

AP 只差冻结 `+0.005` effect floor，但 AUROC 与 safest-choice effect size 明确未过门；
不能把“接近”改写为支持。

## 方向分层信号

预先存在的 left/center/right 三方向分层必须与总体负终态同时保留：

| direction | source-macro horizon-macro AUROC | 正折 | AP | 正折 |
|---|---:|---:|---:|---:|
| left | +0.00825 | 2/5 | +0.00619 | 2/5 |
| center | -0.01780 | 0/5 | -0.00605 | 2/5 |
| right | +0.00802 | 4/5 | +0.01289 | 4/5 |

因此可以记录一个窄的 Development 机制信号：

`D26_RIGHT_CANDIDATE_COLLISION_FIELD_SIGNAL_SUPPORTED_DEVELOPMENT_ONLY`

它表示右候选方向的四 horizon ranking 在 seed17 上跨 4/5 folds 同向改善。它不能
撤销 center 失败、不能切换 primary direction、不能触发多 seed 扩展，也不能升级为
整体 field 或 safest-choice utility。

## 为什么它不同于 D24/D25

D26 的 teacher 不再问任意人体是否靠近，而是：

- 从 wearer 当前世界位置与速度生成 `-30°/0°/+30°` 三条恒速候选路径；
- 其他人体沿 source-recorded future trajectory 运动；
- 对每条候选路径预测五类首次 1.25 m 冲突时间。

标签并不冗余：

- 三方向五类 counts 为
  `78/44/35/33/340`、`60/41/34/32/363`、`72/63/46/28/321`；
- 287/530 anchors 的精确首次冲突时间随方向改变；
- 271/530 的五类 time bin 随方向改变；
- 231/530 在 2 秒内 collision/no-collision 上随方向改变；
- 三方向×四 horizon 在每个 fold 都有正负。

所以 D26 的失败不是 target 没有方向信息，而是当前 whole-frame dense-flow student
没有稳定把它转成跨 source field ranking 与方向选择。

## 表示与决策层断裂

五折展示了真实的层级断裂：

- fold0：AUROC/AP `+.01575/+.00477`，但 safest-choice `-.05628`；
- fold1：AUROC/AP `-.00380/-.01813`，safest-choice `+.01753`；
- fold2：AUROC/AP `-.00919/+.00326`，safest-choice `+.06897`；
- fold3：三项同时为正 `+.02663/+.03537/+.03333`；
- fold4：三项同时为负 `-.03192/-.00356/-.03651`。

因此不能用 field AP 的 3/5 正掩盖 choice effect 小，也不能用 choice 的 3/5 正否定
右方向 representation 信号。D26 正确终态是“局部信号存在，整体增量未建立”。

## 工程与复现

- 复用 D25 的逐 arm CPU checkpoint 与 GPU release；
- 训练全程显存约 1.6–1.7/8.1 GiB；
- 没有 OOM、路径、CSV、缓存或落盘工程无效；
- cumulative monotonicity violation：0；
- 1,057,651 trainable parameters；
- fixed final epoch，无 threshold、epoch、seed、方向角或 head 搜索。

```text
artifacts.local/evidence/hftf/
  stage-c-d26-thor-magni-counterfactual-collision-field-v0/
    report.json
    report.json.sha256
    checkpoints/
```

- report SHA-256：
  `fa8d8c8977546506113070becf500050c82d018247d7b52ad86fa44075478f2a`；
- report 加 10 checkpoints 合计约 42.0 MiB。

## 下一科学变量

D26 同时可能由两种原因失败：

1. 当前 RGB/dense-flow representation 没有恢复其他人体的可迁移运动；
2. counterfactual target 主要由当前几何决定，历史本来就没有信息增量。

下一步不再直接改网络。先用同一 source-native trajectory 做一个 information-ceiling
oracle：

- `current-static`：其他人体冻结在当前世界位置；
- `history-kinematic`：只用 anchor 前历史估计其他人体速度并恒速外推；
- truth：仍是 source-recorded future trajectory。

若 history-kinematic 在方向×horizon ranking 与 safest-choice 上稳定超过
current-static，瓶颈定位为 RGB motion representation；否则停止这条 tracked-body
counterfactual route。该 oracle 不使用 future 来生成 prediction，只在 evaluation
truth 侧使用未来。
