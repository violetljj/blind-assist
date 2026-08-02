# HFTF Stage C D23：THOR-MAGNI proximity multi-seed protocol

日期：2026-08-02

证据角色：Development / post-hypothesis target-specific robustness

研究主线：不变

默认 App：不变

## 假设来源与边界

D22 的 broad-transfer gate 已失败且保持失败。seed17 同时观察到一个预先存在 target
上的窄正信号：

- proximity source-session-macro AUROC `+.03669`，4/5 folds 正；
- proximity source-session-macro AP `+.03660`，4/5 folds 正；
- pooled proximity AUROC/AP `-.00100/+.00723`。

D23 在看到 D22 后冻结，因此不是 fresh confirmation。它只问：

> 同一 D22 训练与未运行的 seeds23/41 下，proximity target 的增量是否保持方向、
> source-session 折一致性与 pooled noninferiority？

即使通过，也只能建立 target-specific multi-seed Development robustness。D22 的
corridor 与 broad-transfer 负结果不撤销，不允许改写为 D22“通过”。

## 冻结执行

- 新执行 seeds：`23/41`；
- 合并已观察 seed17，最终 5 folds × 3 seeds = 15 paired units；
- 仍同时训练 D22 的 proximity/corridor 两个 heads；
- samples、RGB、flow、MobileNet、模型、loss、source weighting、30 epochs、LR、
  batch、固定 final epoch 与 current comparator 全部不变；
- 不根据 seed23 中间结果停止、调参或改 gate。

## Proximity gate

history-minus-current 必须全部满足：

1. source-macro AUROC overall mean 至少 `+0.010`；
2. source-macro AP overall mean 至少 `+0.005`；
3. AUROC/AP 的三个 seed 五折 mean 全部为正；
4. AUROC/AP 的 fold seed-mean 各至少 3/5 为正；
5. AUROC/AP 各至少 10/15 individual units 为正；
6. pooled proximity AUROC/AP overall mean 均不低于 `-0.005`。

失败则保留 D22 seed17 观察，但停止 target-specific dense-flow transfer。通过则只
授权冻结真实事件层的 proximity-onset decision test，不授权主线、App 或安全主张。

路径、缓存、空 batch、落盘或中断错误仍是可修复工程故障；在新增 seeds 的 held-out
metrics 产生前修复重跑，不烧毁 source。
