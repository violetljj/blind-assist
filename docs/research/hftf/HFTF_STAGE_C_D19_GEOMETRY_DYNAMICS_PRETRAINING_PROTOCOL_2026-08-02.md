# HFTF Stage C D19：geometry-dynamics pretraining protocol

日期：2026-08-02

证据角色：Development / synthetic dynamics-pretraining canary

研究主线：不变

默认 App：不变

## 待检验主张

D18 证明 flow alignment 能恢复 pooled cell localization，但 environment-macro
AUROC 幅度不足。D19 只检验 sparse onset supervision 是否是剩余瓶颈：

> 在相同 flow-aligned encoder 上先学习完整 current/near/far body/head geometry
> fields，是否能把 D18 的 pooled signal 稳定到 held-out environments。

## 冻结设计

- 数据、D16 onset targets、RAFT flow、三折和 seed17 全部继承 D18；
- current/history 两臂仍共享 identical model、initialization 与 batch order；
- total budget 仍为 30 epochs；
- epochs 1–15：预测 current/near/far × body/head 共六个 6×6 teacher risk fields；
- epoch 16 前：把 near/far 四个 future-field head channels 精确复制到 onset head；
- epochs 16–30：在同一 train fold 上 fine-tune D16 四个 onset fields；
- 两阶段都使用 train-fold known/eligible masks、positive weights 和
  environment-balanced BCE；
- encoder/head LR、weight decay、batch size 与 fixed-final-epoch 规则不变；
- 不读取 held-out outcome 选 checkpoint 或 schedule。

## Seed-17 gate

完整复用 D18：

1. environment-macro cell AUROC mean 至少 `+0.010`；
2. environment-macro cell AP mean 至少 `+0.005`；
3. AUROC/AP 各至少 `2/3` folds 为正；
4. 至少 `3/4` cell targets 的 AUROC/AP mean 同时为正；
5. sample-macro AUROC/AP 均不低于 `-0.005`。

全部通过才原样扩展 seeds `23/41`。失败则不调整 pretrain/onset epoch 比例、loss
weight、teacher channel、flow 或 gate；下一变量转向显式 future-dynamics prediction，
而不是继续当前 field-pretraining schedule。
