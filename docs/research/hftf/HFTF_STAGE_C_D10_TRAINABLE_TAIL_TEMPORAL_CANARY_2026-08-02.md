# HFTF Stage C D10：THOR-MAGNI trainable-tail temporal canary

日期：2026-08-02

证据角色：Development / trainable representation canary

研究主线：不变

默认 App：不变

## 结论

D10 没有支持“解冻 MobileNet 晚层后，真实五帧 history 能稳定优于
current-only”的窄假设。五个 source-session-held-out folds 中，近距与走廊的
AUROC/AP history-minus-current 都只有 2/5 折为正；四项 mean delta 中三个为负，
唯一略正的近距 AP 只有 `+0.000004`。

终态：

`D10_TRAINABLE_TAIL_TEMPORAL_INCREMENT_NOT_SUPPORTED_STOP`

这是完整执行后的科学负结果，不是路径、cache、parser、hash、显存、中断或
claim-ceiling 失败。冻结成功门没有通过，因此不扩展 seeds `23/41`，不运行 JRDB
zero-shot，也不调整 epoch、解冻边界、学习率或 head 救援。本终态关闭当前
MobileNet late-tail + temporal residual successor；它不删除 D8 的局部路线监督资产，
也不把 D8 的 coarse separability observation 改写为不存在。

## 为什么执行 D10

D8 的 frozen pooled screen 曾在近距和走廊 AUROC 上得到 5/5 折正方向，但等容量
pooled/spatial heads 没有建立稳定双目标增量，D9 的 JRDB corridor replication
也失败。D10 不再更换 frozen head，而允许 representation 的晚层共同训练，直接检验
先前信号是否只是 frozen feature ceiling。

## 冻结设计

- 数据：D8 的 19 个 THOR-MAGNI Pupil/QTM sessions、1,078 个 samples；
- split：沿用固定五折 source-session isolation；
- 输入：五帧 `128×224` RGB，history offsets 与 D8 一致；
- encoder：pretrained MobileNetV3-small；
- 冻结 blocks `0..8`，训练 blocks `9..12`；
- current/history 两臂模型完全相同，均有 765,386 个 trainable parameters：
  736,488 个 backbone 参数与 28,898 个 temporal/head 参数；
- current arm 只编码 current frame，再把 feature 重复五次；
- history arm 编码五个真实 frames；
- BatchNorm 使用 pretrained running statistics，不在训练中更新；
- seed `17`、8 epochs、batch 24、source-balanced BCE；
- backbone/head learning rate 分别为 `2e-5/3e-4`；
- fixed final epoch，不用 held-out fold 选模。

扩展门在运行前冻结：近距/走廊的 AUROC/AP 四项 history-minus-current 都必须
`mean > 0`，且各自至少 3/5 folds 为正，才允许原样扩展 seeds `23/41`。

## 结果

| 指标（history - current） | mean | median | 正折 |
|---|---:|---:|---:|
| 近距 AUROC | -0.000235 | -0.000156 | 2/5 |
| 近距 AP | +0.000004 | -0.000245 | 2/5 |
| 走廊 AUROC | -0.000403 | -0.000682 | 2/5 |
| 走廊 AP | -0.000546 | -0.000487 | 2/5 |

按折结果：

| fold | 近距 AUROC | 近距 AP | 走廊 AUROC | 走廊 AP |
|---:|---:|---:|---:|---:|
| 0 | -0.000156 | +0.000006 | -0.000682 | -0.001005 |
| 1 | +0.000313 | +0.001333 | +0.000208 | +0.000155 |
| 2 | -0.000491 | -0.000560 | -0.000983 | -0.001757 |
| 3 | -0.001437 | -0.000513 | -0.000760 | +0.000362 |
| 4 | +0.000594 | -0.000245 | +0.000203 | -0.000487 |

效应接近零且折方向不稳定。因为四项 gate 同时失败，继续多 seed 只会变成对已消费
结果的救援搜索。

## 工程执行与可复现证据

RGB cache 使用 `.partial.npy` 顺序写入；中断时删除未完成 partial 后可按同一输入
重建，只有完整填充后才 atomic replace 为正式 cache。这是可修复工程机制，不是
source burning 或科学 one-shot。

```text
artifacts.local/evidence/hftf/
  stage-c-d10-thor-magni-trainable-rgb-cache-v0/
    history_rgb_uint8.npy
    history_rgb_uint8.npy.json
  stage-c-d10-thor-magni-trainable-tail-canary-v0/
    report.json
    report.json.sha256
```

- samples SHA-256：
  `c2c63251f727fe5f89241a060f0dcc3ec5a851bb7b878f18b8e0745e25d5363a`
- RGB cache shape：`[1078,5,128,224,3]`
- RGB cache bytes：`463626368`
- RGB cache SHA-256：
  `0699e41b92f5c1dd0fefe990071dee140386802ab5089cdbc32bdb348ad9566f`
- pretrained weights SHA-256：
  `047dcff4addef86ea5bc2eff13c9614dc11f47ab1160d0a71a25e7db994f4e1f`
- result SHA-256：
  `786b2b002a4c013e00db4913bdb361f16eee62d2de280d876b1baca93a4775d6`

## 主张边界与后继

D10 只裁决当前 late-tail temporal residual recipe。它不裁决所有 video model、
独立时空预训练、显式 correspondence，或未来取得更大且更多来源监督后的新表示。
若将来重开，必须提出新的表示学习变量和新的协议边界；不能只更换 seed、epoch、
学习率、解冻层数或 temporal head。
