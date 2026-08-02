# HFTF Stage C D20：dense-flow dynamics result

日期：2026-08-02

证据角色：Development / synthetic explicit-dynamics canary

研究主线：不变

默认 App：不变

## 结论

完整 gate 终态：

`D20_DENSE_FLOW_DYNAMICS_CANARY_NOT_SUPPORTED`

但 D20 通过 7 项冻结检查中的 6 项，并建立当前 HFTF true-onset 路线上最广泛的
Development representation 正结果：

`D20_DENSE_FLOW_DYNAMICS_BROAD_ONSET_SIGNAL_SUPPORTED_DEVELOPMENT_ONLY`

history-minus-current：

| metric | mean delta | 正折 |
|---|---:|---:|
| environment-macro cell AUROC | +0.00431 | 2/3 |
| environment-macro cell AP | +0.03421 | 3/3 |
| pooled target-macro cell AUROC | +0.00604 | 3/3 |
| pooled target-macro cell AP | +0.00966 | 3/3 |
| sample-macro AUROC | +0.01612 | 3/3 |
| sample-macro AP | +0.01068 | 2/3 |

四个 targets 的三折 mean AUROC 与 AP 全部同时为正。far-head AUROC/AP 为
`+.01222/+.00807`、均 3/3 folds 正；far-body AP 为 `+.01964`、3/3 folds 正。

唯一未通过的是 primary environment-macro cell AUROC effect floor：
实际 `+.00431`，低于冻结的 `+.010`；其 2/3 positive-fold gate 已通过。因此不扩
seeds `23/41`，不在看到结果后把 effect floor 降到 `.004`。

## 相对 D18/D19 的信息

- D18 只用 flow 对齐时，environment-macro AP 为 `+.00550`；
- D20 显式保留 dense x/y velocity、magnitude、validity 后提高到 `+.03421`；
- D19 static geometry pretraining 的 AP 为 `-.00321`。

这说明剩余增量来自 local motion dynamics，而不是继续强化静态 geometry
appearance。D20 的 20-channel dynamics representation 包含：

- 16-channel aligned history feature residual；
- normalized flow x/y；
- normalized magnitude；
- warp validity。

current comparator 的整张 dynamics tensor 精确为零；两臂使用 identical
1,004,392 parameters、30 direct-onset epochs 和原三折。

## 可复现证据

```text
artifacts.local/evidence/hftf/
  stage-c-d20-tartanground-dense-flow-dynamics-v0/
```

- report SHA-256：
  `1d42de43f938bbb6b3134a9e512a2ed56279e9d8426dab2bbd5e7a74b08275da`

## 下一科学变量

D20 的 depthwise 3D stem 一次性把四个时间步 collapse。D21 保留完全相同的
20-channel dynamics tensor、数据、训练与 gate，只把 temporal collapse 换为四步
ConvGRU：

- hidden state 按 `-.8→-.6→-.4→-.2 s` 更新；
- zero input + zero hidden 必须保持 current comparator 精确为零；
- final future-state residual 在 MobileNet block 1 前写回 current feature。

这检验显式状态递推能否把 D20 已有的 AP/广度正结果提升为达到 effect floor 的
AUROC separation；不得同时改变 flow、loss、epoch 或 gate。
