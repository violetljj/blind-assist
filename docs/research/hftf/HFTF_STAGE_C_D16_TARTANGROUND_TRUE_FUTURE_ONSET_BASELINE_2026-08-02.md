# HFTF Stage C D16：TartanGround true-future-onset baseline

日期：2026-08-02

证据角色：Development / source-diverse synthetic onset baseline

研究主线：不变

默认 App：不变

## 结论

在 15 environments、495 samples、1,652 个 onset cells 的充足 synthetic cohort
上，frozen ImageNet MobileNet spatial features 后接 temporal residual head 仍没有
建立稳定 history increment。

终态：

- `D16_TARTANGROUND_FUTURE_ONSET_THREE_FOLD_READY`
- `D16_TARTANGROUND_FUTURE_ONSET_HISTORY_INCREMENT_NOT_SUPPORTED`

history-minus-current：

| target | AUROC mean / 正折 | AP mean / 正折 |
|---|---:|---:|
| near body | +0.00053 / 2/3 | +0.00068 / 2/3 |
| near head | +0.00047 / 2/3 | +0.00118 / 2/3 |
| far body | -0.00140 / 1/3 | +0.00053 / 1/3 |
| far head | -0.00230 / 1/3 | -0.00287 / 1/3 |

运行前门要求每个 target 的 AUROC mean 至少 `+.01`、AP 至少 `+.005`，并且八项
全部 3/3 folds 为正。没有任何指标达到 effect floor。

这个结果排除了“THOR/JRDB onset 数量太少，所以 temporal residual 学不到”的主要
替代解释。当前失败更接近 representation mismatch：通用单帧 ImageNet feature
冻结后，再用浅层 residual fusion 无法稳定表达 future onset。

因此关闭：

`FROZEN_SINGLE_FRAME_FEATURE_PLUS_POSTHOC_TEMPORAL_RESIDUAL_FAMILY_STOP`

这个边界覆盖 D8/D13/D15/D16 的 pooled/spatial residual family，不关闭 HFTF
true-onset task、geometry teacher、时空预训练、future-field pretraining 或
end-to-end video representation。

## True-onset corpus

D16 直接继承既有 `HFTF_D5_CROSS_ENV_V1` 三折 environment assignments，不根据
onset 计数重新分折。WaterMillDay/Night 继续在同一 fold。

对 near/far × body/head 的每个 6×6 cell：

```text
eligible = current and future both known, current risk < 0.5
onset = eligible and future risk >= 0.5
sample onset = any onset cell for the horizon and height
```

总体：

| target | eligible samples | positive | negative |
|---|---:|---:|---:|
| near body | 309 | 191 | 118 |
| near head | 459 | 219 | 240 |
| far body | 294 | 212 | 82 |
| far head | 451 | 265 | 186 |

cell-level 共 19,478 个 eligible、1,652 个 onset。三个 held-out environment folds
的四个 targets 都同时有正负例。

## 等容量 baseline

- frozen MobileNet `5×576×4×7` maps；
- current arm 将 current map 重复五次；
- history arm 使用真实五帧；
- 两臂共享相同 14,484-parameter temporal-spatial head；
- seeds `17/23/41`、120 epochs；
- target-masked、environment-balanced BCE；
- fixed final epoch，无 held-out 选模。

near 的千分位正方向与 D13 THOR weak signal 一致，但效应远低于预定门，far 又反向。
这不足以支持 synthetic pretraining 或 real transfer。

## 可复现证据

```text
artifacts.local/evidence/hftf/
  stage-c-d16-tartanground-future-onset-v0/
  stage-c-d16-tartanground-spatial-features-v0/
  stage-c-d16-tartanground-future-onset-temporal-baseline-v0/
```

- onset samples SHA-256：
  `df27558c8ef38c48bbb9e61ba3ea0bfa1cc23ab8beba152567beaeee168c22ce`
- corpus report SHA-256：
  `d2841563c6f3c44edeb03e2d5893f895604db9908f0ae0c88a5e1fdd8c5bff31`
- spatial features SHA-256：
  `5a0fa11e36e4a011475b965af38d89ebb7aa0e5156f50247460aaf2659bec60a`
- baseline report SHA-256：
  `912c54d2a5276978f135b79bc52e161bf9a9c31d72c354945607e147873ba01a`

## 下一科学变量

下一候选必须把时序学习前移到 representation pretraining，而不是继续更换 post-hoc
head。最低可接受后继是一个从五帧共同编码、直接以 D16 cell onset/future field
预训练的轻量模型，再做 environment-held-out 和 THOR/JRDB transfer。若只改变
seed、epoch、residual head、grid 或 threshold，不构成新候选。
