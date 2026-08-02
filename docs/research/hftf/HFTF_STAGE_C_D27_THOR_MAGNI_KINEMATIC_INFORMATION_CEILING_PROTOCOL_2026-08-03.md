# HFTF Stage C D27：THOR-MAGNI kinematic information-ceiling protocol

日期：2026-08-03

证据角色：Development / source-native information ceiling

研究主线：不变

默认 App：不变

## 目的

D26 的 action-conditioned field AP 与 safest-choice 各有 3/5 folds 为正，右方向
AUROC/AP 均 4/5 folds 正，但整体 effect 未过门。当前无法区分：

1. 历史运动对 counterfactual collision target 本来就没有足够增量；
2. 历史运动有增量，但 whole-frame RGB dense-flow student 没有恢复可迁移的人体运动。

D27 不训练模型、不调 D26 checkpoint。它用 source-native 世界轨迹比较两个因果 oracle：

- `current_static`：只读 anchor 当前其他人体位置，未来保持静止；
- `history_kinematic`：只读 anchor 及其前 0.4 s 其他人体位置，以两点差分估计世界平面
  速度并恒速外推；历史不足时退化为静止。

prediction 两臂都不读取 anchor 后的位置。truth 仍是 D26 使用的 source-recorded future
trajectory。D27 问的是 history information ceiling，不是部署算法。

## 冻结几何

- cohort、fold、三条 wearer 候选路径、速度、方向角、0.10 s 采样、2.0 s horizon 与
  1.25 m truth 全部继承 D26；
- wearer 候选路径：当前 wearer 位置，以当前 wearer 速度沿
  `-30°/0°/+30°` 恒速滚动；
- 对每个方向与累计 horizon `.5/1/1.5/2 s`，prediction score 是该 arm 预测的
  wearer—其他人体最小同步距离取负值；
- 当前不可见但未来出现的人体不允许 oracle 预知；
- 距离上限固定 10 m，保证无当前人体时 score 有限；
- history velocity 窗口固定 0.4 s，不搜索窗口、平滑器、速度裁剪或 body 子集。

## 冻结指标与 gate

在五个既有 source-session folds 分别计算 `history_kinematic - current_static`：

- source-session-macro 与 pooled 的 direction×horizon AUROC/AP；
- left/center/right 各自的 horizon-macro AUROC/AP；
- 对 287 个 exact-time direction-nonredundant anchors，使用 2 秒预测最小距离最大的
  方向作为 safest choice，计算 source-macro accuracy；
- history velocity coverage 仅披露，不作为结果优化变量。

D27 只有在以下条件全部满足时支持 history kinematic information ceiling：

1. source-macro direction×horizon AUROC mean delta 至少 `+0.020`；
2. source-macro direction×horizon AP mean delta 至少 `+0.010`；
3. AUROC/AP 各至少 3/5 folds 为正；
4. AUROC/AP 各至少 2/3 directions 的五折 mean 为正；
5. source-macro safest-choice accuracy mean delta 至少 `+0.050`，且至少
   3/5 folds 为正；
6. pooled direction×horizon AUROC/AP mean delta 均不低于 `-0.005`；
7. 随 horizon 增长，oracle risk score 的单调性 violation 为 0。

通过终态：

`D27_THOR_MAGNI_HISTORY_KINEMATIC_INFORMATION_CEILING_SUPPORTED`

失败终态：

`D27_THOR_MAGNI_HISTORY_KINEMATIC_INFORMATION_CEILING_NOT_SUPPORTED`

## 解释边界

通过只证明 D26 target 中存在 source-native history-motion information ceiling，定位
RGB student representation 为瓶颈；不证明 RGB 可学、真实提醒有效或 HFTF 已超过主线。

失败则说明一个简单恒速人体运动 oracle 也不能稳定超过 current-static；停止用当前
tracked-body counterfactual target 继续训练更大的 motion model。无论结果如何，都不
改写 D26 的局部 right-direction signal，也不产生 App、生产或安全权限。

路径、CSV、parser、serialization 或中断异常仍是工程无效，可在任何 oracle metric
产生前修复并按同协议重跑。
