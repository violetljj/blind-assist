# FP-aware DDRNet R0 Development Result

状态：`COMPLETE / VALID / FP_WEIGHTED_SAMPLING_NOT_SUPPORTED /
SINGLE_SUCCESSOR_STOP / CONSUMED_DEVELOPMENT_ONLY /
FINAL_CONFIRMATION_NOT_ACTIVATED / DEFAULT_APP_UNCHANGED`

日期：2026-08-01（Asia/Hong_Kong）

## 结论

冻结候选 `FP_WEIGHTED_UNGUIDED_FULL_FRAME` 不受支持。三组 same-seed pair 没有一组
通过全部 relative 五门与 absolute 四门：

- seed `20260711` 的 FP pixel 只降低 `19.8713%`，同时 overall/min-session recall
  retention 只有 `0.822053 / 0.738823`；
- seed `20260712/13` 的 FP pixel 分别增加 `13.8991% / 4.3984%`；
- 三个 seed 的 `C-A` FP-area 增量均超过 `.05`，false components/frame 均超过
  `3.0`。

因此 R0 终态为：

```text
FP_WEIGHTED_SAMPLING_NOT_SUPPORTED
```

这只关闭“在保持其他 R1 合同不变时，用 train-only baseline FP pixels 重加权 30%
full-frame 帧抽样”这个单一 successor；不等于所有 residual-aware/FP-aware training、
learned loss、target repair 或新数据路线失败。

## 冻结身份与执行

- 冻结 Git commit：
  `e98b3efb7d556351c6536923553f46302b3ac47e`
- config SHA-256：
  `4261fdb486a49c77b77287f24e597dba6b65c72725b85e7ebe9c288f1c5ccfa6`
- 三 seed：`20260711 / 20260712 / 20260713`
- 每 seed：100-step head warmup + 1100-step backbone fine-tune
- terminal：`consumed_old_blind 120 + r1_consumed_fresh 200`
- cross-seed selection：`FORBIDDEN_NOT_PERFORMED`

第一次前台训练在 seed `20260711` step 100 后被外部调用器的 60 秒进程组清理中断。
该尝试只读取 train/dev，未访问 320-frame terminal truth；默认 `training/` 中的部分
进度原样保留且未用于评价。随后在 `training-recovery-v2/` 上以同一冻结 commit、
config、seed 和预算完整重跑，没有修改科学变量。

完整训练耗时约 `206.709s`，三个 seed 的 guided / FP-weighted branch draws 分别为：

| seed | guided | FP-weighted full-frame | fallback | selected step | checkpoint SHA-256 |
|---:|---:|---:|---:|---:|---|
| 20260711 | 10,090 | 4,310 | 0 | 600 | `e312dfe0d6c1a8b3397a4c2e190b3dfb1ebdc968df19dae8799d4552737aa7d9` |
| 20260712 | 10,207 | 4,193 | 0 | 50 | `41d8b291079402bf2e3bb75a764ae3f9eccb1795bacced9eb681c6b697db49f6` |
| 20260713 | 10,058 | 4,342 | 0 | 150 | `e1bcc6ccb911a5de936e17dd8216d57b6c10b475b106e5adb8cb3fad94b9def1` |

## 冻结九门结果

Relative gates：FP reduction `>=.30`、overall `>=.90`、min-session `>=.80`、
boundary `>=.80`、obstacle `>=.80`。Absolute gates：`C-A` recall `>=.05`、
`C-A` FP area `<=.05`、component recall `>=.50`、false components/frame `<=3.0`。

| seed | FP reduction | overall retention | min-session retention | boundary retention | obstacle retention | C-A recall | C-A FP area | component recall | false comp/frame | 9/9 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 20260711 | .198713 | .822053 | .738823 | 1.031659 | .834100 | .335605 | .114689 | .686857 | 4.418750 | FAIL |
| 20260712 | -.138991 | .957149 | .804303 | 11.334504 | .855867 | .313900 | .133195 | .713143 | 7.818750 | FAIL |
| 20260713 | -.043984 | .877612 | .767059 | .099946 | 1.007139 | .308211 | .114734 | .717333 | 5.618750 | FAIL |

seed `20260712` 的 boundary retention 很大，是较小 baseline boundary TP
`1,994` 对 candidate `22,601` 的比值；它伴随 FP 增加和
`7.81875` false components/frame，不能解释为稳健通过。seed `20260713` 又把
boundary TP 从 `29,906` 降到 `2,989`。这说明采样改变了 class/seed trade-off，
没有形成跨 seed 稳定的 FP 抑制机制。

三 seed 的 absolute recall 与 component recall 均过门，但 absolute FP area 与
false-component 门均失败。因此 observed signal 只是“增加了某些 residual truth
coverage”，不是受约束的 utility increment。

## 验证与绑定

validator 重新装载三组 baseline 与三组 candidate checkpoints，在同一 320 帧上重新
推理，并将 1,920 个 prediction masks 与 evaluator 账本逐像素比对；随后从逐帧账本
复算全部 pixel、component、session、class 和 terminal 指标。

- validator：`VALID`
- checks：`28,861`
- errors：`0`
- training report SHA-256：
  `7cd59973a8eefbb67d54b20c1524e567d32eeb0a31607872e4c80193afb1966b`
- result SHA-256：
  `58447cc4eeed19f38f1f22f17cc4cf334c10b988bc6271deb6bbfdba50774f23`
- frame predictions SHA-256：
  `0b5342b4793904f5e2d40d85b997cd6dc9019734e23db7f260bb6c7915399924`
- validation SHA-256：
  `9a8b4c5289dcf38680b19b8d1e1bff5a54da7aca6e0a4dd044dbca2a9a6c41a9`

## 停止与权限

不允许在这 320 帧上用以下方式救援：

- 选择 `1/3` 或 `2/3` seed；
- 把 full frame 改成 hard-negative crop；
- 改 sampling weight、阈值、loss 或 target；
- 用 dev/terminal outcome 选择新 operating point。

本结果不授权 INT8/TFLite、runtime/device、Android/QNN/A568、risk/event、
feedback、TTS、振动、提醒或默认模型变更。若未来继续训练研究，必须另立有不同因果变量
和明确数据角色的新 Development 协议；本 R0 不自动授权该后继。
