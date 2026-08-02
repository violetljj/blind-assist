# HFTF Stage C D30：THOR-MAGNI box-to-world measurement protocol

日期：2026-08-03

证据角色：Development / current-frame measurement correspondence diagnostic

研究主线：不变

默认 App：不变

## 问题

D29 没有证明 object-slot motion residual，但它仍把 2D box 直接映射到完整 future
field。D30 不训练风险模型，只回答更基础的问题：

> current YOLO person boxes 与 source-native current body positions 之间，是否存在
> 足够稳定的 bearing、distance-rank 与 nearest-body correspondence？

若 current measurement 本身不成立，继续增加 temporal head、flow statistic 或 field
loss 都没有明确对象状态可更新。

## 冻结输入与几何

- 复用 D29 object-slot cache、530 anchors、19 sources 与原五 folds；
- 只使用 current box 的 x-center、height、confidence；
- source-native 侧只读取 anchor 时刻 wearer/body positions；
- 由于 D29 detector 为 COCO person，measurement truth 只保留 body name
  `Helmet_*` 且 source role 以 `Visitors-` 或 `Carrier-` 开头的人体；明确排除
  `DARKO_Robot`、`LO1` carried object 与其他非 person rigid bodies；
- wearer forward 仍由 anchor 前后各 0.25 s camera world displacement确定；
- 不读取 anchor 后 body positions、D26 future truth、D27 score 或 D29 outcome；
- visible proxy 固定为 distance `<=10m` 且 relative bearing 在 `[-50°, +50°]`；
- world 左侧正 bearing 映射为 image 负 x，固定
  `predicted_x_signed = -bearing_degrees / 50`；
- 对每个 anchor 用 Hungarian assignment 最小化
  `abs(box_x_signed - predicted_x_signed)`；
- assignment error `<=0.25`（等价 bearing error `<=12.5°`）才算 accepted；
- 不搜索 FOV、distance cap、assignment cost、acceptance threshold 或 body subset。

## 冻结披露

按 pooled、source-macro、五 folds 披露：

1. detector/body 同时存在的 anchor opportunity；
2. assigned/accepted box 与 visible-body coverage；
3. nearest visible body accepted coverage；
4. assigned pairs 的 box-x vs predicted-x Pearson；
5. bearing MAE；
6. box height vs inverse world distance Spearman；
7. ambiguity：boxes/bodies 数量不等与多候选 assignment。

## 冻结 gate

D30 只有在以下全部满足时建立 measurement relation：

1. detector 与 visible body 同时存在的 anchor 至少 300；
2. accepted assignment / assigned pairs 至少 `0.60`；
3. nearest visible body accepted coverage 至少 `0.60`；
4. source-macro x Pearson 至少 `0.50`；
5. source-macro bearing MAE 不超过 `15°`；
6. source-macro height-vs-inverse-distance Spearman 至少 `0.30`；
7. 该 Spearman 至少 3/5 folds 为正；
8. 至少 15/19 sources 具有 5 对以上 assigned measurements。

通过终态：

`D30_THOR_MAGNI_BOX_WORLD_MEASUREMENT_RELATION_SUPPORTED`

失败终态：

`D30_THOR_MAGNI_BOX_WORLD_MEASUREMENT_RELATION_NOT_SUPPORTED`

## 边界

D30 通过只说明当前 2D box 可以作为结构化 world-state estimator 的输入机会；它不
建立 identity truth、velocity、future collision、用户事件、App 或安全主张。失败只
关闭当前固定 FOV/current-box measurement relation，并触发迁移到具有原生 2D/3D
identity binding 的独立 person-trajectory source，而不是继续在 D29 outcomes 上调参。
无论通过或失败，D30 都不覆盖 D26/D27 中的 robot/carried-object collision；后续
完整 field 必须另设非 person object measurement。
