# HFTF Stage C D21：ConvGRU future-state protocol

日期：2026-08-02

证据角色：Development / synthetic recurrent-dynamics canary

研究主线：不变

默认 App：不变

## 待检验主张

D20 的 dense-flow dynamics 已通过 7 项 gate 中 6 项，但一次性 3D collapse 的
environment-macro AUROC effect 只有 `+.00431`。D21 只改变 temporal state operator：

> 按时间顺序递推的 future state，是否能把 D20 的广泛 AP/target 信号转成达到
> `+.010` floor 的跨环境 AUROC separation。

## 冻结设计

- 输入完全复用 D20 的四步 20-channel dynamics tensor；
- 用 bias-free 3×3 ConvGRU 依次处理 `-.8→-.6→-.4→-.2 s`；
- hidden channels `16`；
- zero dynamics + zero hidden 必须在任意 recurrent weights 下保持精确为零；
- final hidden 经 zero-initialized 1×1 projection 写回 current low feature；
- D16 数据、RAFT cache、三折、seed17、30 epochs、LR、loss、evaluation 和 gate
  全部不变。

## Seed-17 gate

1. environment-macro cell AUROC mean 至少 `+0.010`；
2. environment-macro cell AP mean 至少 `+0.005`；
3. AUROC/AP 各至少 `2/3` folds 为正；
4. 至少 `3/4` cell targets 的 AUROC/AP mean 同时为正；
5. sample-macro AUROC/AP 均不低于 `-0.005`。

全部通过才扩展 seeds `23/41`。失败则保留 D20 为当前 Development signal，并停止
当前 lightweight early temporal-state family；不继续改 hidden width、gate kernel、
epoch、loss 或门槛。
