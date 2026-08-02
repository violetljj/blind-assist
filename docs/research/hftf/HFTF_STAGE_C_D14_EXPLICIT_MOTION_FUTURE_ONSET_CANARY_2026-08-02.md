# HFTF Stage C D14：显式运动 future-onset canary

日期：2026-08-02

证据角色：Development / explicit-motion representation canary

研究主线：不变

默认 App：不变

## 结论

方向保持的 pretrained RAFT-small motion features 没有在 D12 true-future-onset
任务上建立跨来源、双目标的稳定增量。

终态：

`D14_EXPLICIT_MOTION_FUTURE_ONSET_INCREMENT_NOT_SUPPORTED`

四项 motion-minus-current：

| 指标 | mean | median | 正折 |
|---|---:|---:|---:|
| 近距 AUROC | +0.00048 | -0.00273 | 2/5 |
| 近距 AP | -0.01025 | -0.00795 | 2/5 |
| 走廊 AUROC | +0.02191 | +0.03141 | 3/5 |
| 走廊 AP | +0.02404 | -0.00485 | 2/5 |

走廊的 mean signal 比 D13 千分位增量更大，但集中在 folds 0/1；AP median 为负，
也只有 2/5 folds 为正。近距不支持。按运行前冻结的双目标、四指标 gate，D14
不进入轻量 temporal student。

这是完整执行后的科学负终态，不是工程、cache、显存或 claim-ceiling 失败。走廊
局部正信号按真实层级保留，但不能在看到结果后把 primary target 切换为 corridor，
也不继续搜索 RAFT 层、grid、global-motion remover、head 或 threshold。

## 固定表示

输入为 D10 的 1,078×5 RGB cache。每个样本对四个相邻 frame pairs 运行本地已校验
权重的 torchvision RAFT-small：

- weights SHA-256：
  `01064c6dba73b0fc9fc8edf772248560a00a3acfd62ac6677e9eeebad9680e27`
- 4,312/4,312 pairs 完成；
- 3,311 pairs 使用 RANSAC partial-affine global motion；
- 1,001 pairs 使用预定 median-translation fallback；
- 无缺失或不可评价 pair。

每个 pair 固定输出 3×6 grid 的八个 channels：

1. raw flow mean x；
2. raw flow mean y；
3. raw flow mean magnitude；
4. residual flow mean x；
5. residual flow mean y；
6. residual flow mean magnitude；
7. raw magnitude p90；
8. residual magnitude p90。

全部 raw/residual channels 一次保留，不按结果选择 representation。

## 等容量 canary

两臂共享同一 49,490-parameter model：

- current spatial context：frozen MobileNet `576×4×7` current map；
- current arm：motion tensor 全零；
- motion arm：真实 `4×8×3×6` RAFT features；
- seeds `17/23/41`、120 epochs、五折 source-session isolation；
- target-masked、source-balanced BCE；
- fixed final epoch，无 held-out 选模。

为了要求 D14 明显超过 D13 的弱信号，运行前门设为：

- 两个 target 的 AUROC mean 都至少 `+0.01`；
- 两个 target 的 AP mean 都至少 `+0.005`；
- 四项 median 都为正；
- 四项各至少 3/5 folds 为正。

fold-mean delta：

| fold | 近距 AUROC | 近距 AP | 走廊 AUROC | 走廊 AP |
|---:|---:|---:|---:|---:|
| 0 | +0.063 | -0.010 | +0.040 | +0.044 |
| 1 | +0.028 | +0.037 | +0.063 | +0.107 |
| 2 | -0.044 | -0.026 | -0.008 | -0.001 |
| 3 | -0.044 | -0.061 | +0.025 | -0.012 |
| 4 | -0.001 | +0.009 | -0.010 | -0.018 |

这表明 motion observability 具有明显 source dependence。它可以成为未来新增来源设计的
诊断依据，但不能包装成当前模型增量。

## 可复现证据

```text
artifacts.local/evidence/hftf/
  stage-c-d14-thor-magni-explicit-motion-features-v0/
    features.npz
    features.npz.json
  stage-c-d14-thor-magni-explicit-motion-onset-canary-v0/
    report.json
    report.json.sha256
```

- features SHA-256：
  `76d550fbff76e548ca6155c6b517fdbda56b934eaaed835b6c77b7328b4f22f3`
- feature report SHA-256：
  `17c98380d1febec10ea8db5f5fb62da987cf4ed7ff52d5aeb820ce0f978511eb`
- canary report SHA-256：
  `e2dead64ef6021c5f3c97d0727e9b76ecd86064eb5d677c00cb1c827bda972bf`

## 后继边界

D12 的 true-onset estimand 与 D13 弱 history signal 保留。当前固定 RAFT-grid recipe
停止。下一科学工作应解释 folds 0/1 与 2–4 的 source-level motion/visibility 差异，
或增加独立且 onset-rich 的来源；不能对已观察结果继续调当前表示。
