# HFTF Stage C D19：geometry-dynamics pretraining result

日期：2026-08-02

证据角色：Development / synthetic dynamics-pretraining canary

研究主线：不变

默认 App：不变

## 结论

`D19_GEOMETRY_DYNAMICS_PRETRAINING_CANARY_NOT_SUPPORTED`

15 epochs current/near/far geometry-field pretraining 加 15 epochs onset fine-tuning
没有把 D18 的 alignment signal 稳定到 held-out environments，反而降低了 history
相对 current 的大部分指标：

| metric | mean delta | 正折 |
|---|---:|---:|
| environment-macro cell AUROC | -0.00219 | 2/3 |
| environment-macro cell AP | -0.00321 | 1/3 |
| pooled target-macro cell AUROC | -0.00246 | 2/3 |
| pooled target-macro cell AP | +0.00118 | 1/3 |
| sample-macro AUROC | +0.00578 | 1/3 |
| sample-macro AP | -0.00679 | 1/3 |

target breadth 仍有 3/4 mean AUROC/AP 同时为正，但 effect、AP folds 和 sample AP
non-inferiority 均失败。不扩 seeds `23/41`。

## 机制解释

future-field head 转入 onset head 的优化收益是真实的：三个 current folds 的 onset
第一轮 loss 为 `.6437/.5166/.5816`，显著低于 D18 从约 `1.17–1.25` 随机 onset
head 起步；15 epochs 后两臂也都收敛到更低 loss。

但这主要让 current comparator 也成为更强的静态 future-field predictor，没有增加
history 的独立信息价值。fold 0 的 environment-macro AUROC/AP history delta
`-.01544/-.01893` 尤其明显。更密集的通用 geometry supervision 因而不是 D18
跨环境幅度不足的修复。

精确关闭：

`D19_GENERIC_GEOMETRY_FIELD_PRETRAINING_SCHEDULE_STOP`

这不否定 D18 的 flow alignment / pooled localization 正结果，也不关闭显式运动
dynamics。它只停止当前 `15 geometry + 15 onset`、future-head-copy schedule；
不调整比例、loss weight、teacher channel 或 seed 救援。

## 可复现证据

```text
artifacts.local/evidence/hftf/
  stage-c-d19-tartanground-geometry-dynamics-pretraining-v0/
```

- report SHA-256：
  `9716654421b6caf6937fe2675c5c0292cea4adc56324df2466358b19dd7d6e77`

## 下一科学变量

D20 回到 D18 的 30-epoch direct onset training 与同一 gate，但不再只把 RAFT 当 warp
坐标后丢弃。每个时间步将显式输入：

- aligned history feature residual；
- normalized dense flow x/y；
- flow magnitude；
- backward-warp validity。

这把网络从“用 flow 对齐 appearance”推进到“直接编码 local velocity/dynamics”，
而不再给 current-static appearance 增加额外 teacher supervision。
