# HFTF Stage C D20：dense-flow dynamics protocol

日期：2026-08-02

证据角色：Development / synthetic explicit-dynamics canary

研究主线：不变

默认 App：不变

## 待检验主张

D18 表明 alignment 有效，D19 表明继续强化 static geometry prediction 无效。D20
检验剩余的直接假设：

> 模型需要看到 dense local velocity 本身，而不只是用 flow 把 appearance 对齐。

## 冻结设计

保留 D18 的 aligned history feature residual，并在每个历史时间步追加：

- `flow_x / width`；
- `flow_y / height`；
- normalized flow magnitude；
- backward-warp validity。

共 `16 + 2 + 1 + 1 = 20` 个 dynamics channels，按四个有序历史时刻进入一个
depthwise-separable 3D stem，再在 MobileNet block 1 前写回 current feature。

除这个输入表示外全部回到 D18：

- D16 495 samples、15 environments、原三折与四个 onset fields；
- 同一 RAFT cache；
- identical repeated-current + zero-flow comparator；
- 30 epochs direct onset training；
- 相同 LR、loss、batch、fixed final epoch、evaluation 和 gate；
- current 的 20-channel dynamics tensor 必须精确为零。

## Seed-17 gate

1. environment-macro cell AUROC mean 至少 `+0.010`；
2. environment-macro cell AP mean 至少 `+0.005`；
3. AUROC/AP 各至少 `2/3` folds 为正；
4. 至少 `3/4` cell targets 的 AUROC/AP mean 同时为正；
5. sample-macro AUROC/AP 均不低于 `-0.005`。

全部通过才扩展 seeds `23/41`。失败则停止 dense-flow early residual fusion，下一
结构变量必须是 recurrent future-state model；不得继续搜索 flow normalization、
channel width、epoch、loss 或 gate。
