# RCLE RGB algorithm development canary R0 CID-SIMS floor3_1 result

日期：2026-07-27

## 终态

`INVALID_R0_EVIDENCE / INVALID`

冻结 RGB algorithm 的实际输出呈现很强的 development-direction signal，但锁内
validator 对 `598 / 598` 个 `dt_s` 产生机械误报，因此本次 R0 不能保留 producer
自报的 `/ VALID`。

本轮不修补 validator、不覆盖 `validation.json`、不重跑算法、不调阈值。算法
输出只保留为：

`DIAGNOSTIC_DEVELOPMENT_SIGNAL_DIRECTION_SUPPORTED`

最大权限仍为：

`REAL_RGB_ALGORITHM_DEVELOPMENT_CANARY_ONLY`

它不是 performance qualification、independent confirmation、Kill Gate B、
Android、人类、安全或产品证据。

## 冻结输入与运行

- source：CID-SIMS V6 `floor3_1`；
- archive SHA-256：
  `b622be7918d0003c97f0e33cc30071c9995f49c59726240e7475f2cde8572984`；
- 窗 0：弱运动相邻对照，geometry sampled-pair median signed radial
  `0.028115580109977487 s^-1`；
- 窗 1：正向接近 development canary，geometry sampled-pair median signed
  radial `0.4435781894989603 s^-1`；
- 每窗 `300` 帧、`299` 个窗内相邻 pair；跨窗 pair 排除；
- trigger：compensated expansion `> 0.01 s^-1`，直接继承 Phase A 冻结
  `sign_accuracy_zero_band_per_s`；
- threshold tuning：`false`；
- network request：`0`；
- downloaded bytes：`0`。

Implementation lock SHA-256：

`3ca629e70ed0d349b01b3c0de9f05c659a8f7f09da77948ce7899ca4eb8b858d`

Activation lock SHA-256：

`8eafe679d2b67ee6e2829eca1034744fb5939c75365e69952e33440e8222811c`

## 算法输出

| 指标 | 窗 0：弱运动 | 窗 1：正向接近 |
| --- | ---: | ---: |
| evaluable / candidate pair | `299 / 299` | `299 / 299` |
| abstention | `0` | `0` |
| median compensated expansion | `0.004896 s^-1` | `0.615194 s^-1` |
| trigger | `148 / 299` | `299 / 299` |
| fixed-denominator trigger coverage | `49.50%` | `100.00%` |
| first trigger delay | `5.0688 s` | `0.0376 s` |
| longest consecutive trigger run | `148 pairs / 4.9354 s` | `299 pairs / 9.9767 s` |

正向窗相对弱运动窗：

- median compensated expansion 高 `0.610298 s^-1`；
- trigger coverage 高 `50.50` 个百分点；
- 首次 indication 早约 `5.0312 s`；
- 正向窗从首个 pair 起连续覆盖整个窗口。

因此，若只看冻结算法输出，正向窗稳定触发且与弱运动窗具有明显的
development-level 区分度。弱运动窗仍有 `49.50%` pair 超过 `0.01 s^-1`，说明
该阈值不是产品 alert/persistence policy，也不能从本轮反推或调整阈值。

## 性能与 cache

本轮直接使用已验证本地 ZIP，没有再次下载。派生 cache 一次物化 `600` 个 RGB
PNG 加 `groundtruth.txt` 与 manifest，共 `274,535,705` bytes。算法阶段：

- 窗 0：约 `64.8 s`；
- 窗 1：约 `100.2 s`；
- `598` pair 合计约 `165.0 s`；
- 包含 archive hash、cache 物化与算法的命令 wall time 约 `174.3 s`。

这些数字只说明本次 development run 的执行机械效率，不构成正式 performance
qualification。

## validator INVALID 根因

锁内 validator 从 manifest 读取 timestamp 后先转换为 binary float，再用相邻
float subtraction 形成 expected `dt_s`，并以 `1e-9 s` 绝对容差和 producer 由
精确 Decimal timestamp 计算的 `dt_s` 比较。第一条 pair：

- exact Decimal：`1673419222.314645 - 1673419222.281298 = 0.033347 s`；
- ledger：`0.033347 s`；
- binary-float subtraction：约 `0.0333471298217773 s`。

因此 validator 错误报告 `PAIR_TIMESTAMP:*:*:dt_s` 共 `598` 条。只读
Decimal post-output diagnostic 对全部 timestamp、`dt_s` 与 trigger 重算为
`0` mismatch；独立 ZIP central-directory + cache size/CRC 审计也为 `0`
error。但这些事后诊断不能改写锁内 validator 的 `INVALID`。

## 证据

- result SHA-256：
  `0264b14c901fd1e7460a8d915c6a9be43dc1c1e9bf329594994717ad7d082da0`；
- pair ledger SHA-256：
  `9b381e6b3a5f387e54e315e59e81474fc781818bdacc0e4744e255d41f8a391c`；
- validation SHA-256：
  `67e719923851920f5a6ee148440dd19193487afe0bd1617a97ea9e15406eae75`；
- cache manifest SHA-256：
  `d31cd91859f9008722f399522db3ed74bfdcd09ec0681cd5f2750d7f860c91e5`。

另有两项机械限制：

- runner 没有内建强制复核 implementation lock；activation 已绑定 lock，且并发
  只读审计在启动后确认锁内 `7 / 7` live hash 一致；
- validator 没有独立重跑算法数值，只能承担 identity/cache/ledger/aggregate
  validation，不能称 independent algorithm reexecution 或 confirmation。

## 下一安全动作

若继续，只能另立新的、显式版本化的 **validator-only post-output audit**，绑定
本次 immutable result、ledger、cache manifest 和 INVALID validation，不重跑
RGB algorithm、不改阈值、不覆盖 R0。它最多说明 R0 输出的 identity 与 aggregate
可复算，不能把本次 R0 追认成 outcome-blind、独立确认或性能资格。
