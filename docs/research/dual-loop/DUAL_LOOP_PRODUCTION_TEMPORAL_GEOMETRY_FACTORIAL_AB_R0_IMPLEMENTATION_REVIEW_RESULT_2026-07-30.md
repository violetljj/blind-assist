# Production Temporal Geometry Factorial A/B R0 实现复核结果

日期：2026-07-30（Asia/Hong_Kong）

STATUS: PASS
IMPLEMENTATION_LOCK_VALID: true
FORMAL_EXECUTION_AUTHORIZED: true
PROTOCOL_ID: DUAL_LOOP_PRODUCTION_TEMPORAL_GEOMETRY_FACTORIAL_AB_R0
IMPLEMENTATION_LOCK_SHA256: d7383b9339d46935599d1f0da9bd163b78dd159050e8409a0578969ef9bb23de
IMPLEMENTATION_GIT_COMMIT: 2c53e89a67ec7848a7d2290ebf9e627f6bc96ff6

## 结论

冻结协议的生产链因子 A/B 已达到一次正式 truth-blind producer 的实现门槛。
本结论只授权被 implementation lock 绑定的实现、APK、冻结输入与指定设备执行一次；
不预判 A/B 结果，不授权调参、重跑、Confirmation、生产行为变更、产品或安全主张。

## 复核范围

- A 分支先更新自身 `TemporalRiskTracker` 历史，再只中和 object-detector temporal
  geometry output；B 分支保持当前生产输出，segmentation 语义不受中和模式影响。
- detector 每帧严格执行一次，A/B 共享同一不可变 detections；tracker、kernel、
  stabilizer、event、side confirmation、`FeedbackController`、cooldown、fatigue、
  clock 与 trace 全部按分支隔离。
- producer 使用真实 `AssistEngine` 决策与反馈链、source timestamp 和始终接受的
  delivery ports；两臂的 pre-temporal hash 来自各自实际 analyzer output，并要求
  与共享输入一致。
- 正式 marker 以 `CREATE_NEW` 创建并同步落盘；它位于首帧 decode/尺寸校验之后、
  首次 detector inference 之前。创建后任何失败均保留 marker，并归类
  `INVALID_EXECUTION`；创建前失败不得消费正式 authority。
- device producer 自行复核 implementation lock、activation、prestart、已安装 APK、
  `SM-S9280 / SM8650 / Android 16 / SDK 36 / arm64-v8a` 与 strict QNN HTP runtime。
- truth-blind validator 不接收 truth，逐帧核对冻结 ledger 的身份、时间戳、两臂配对、
  数值模式和完整分母；通过后原子发布绑定 trace、producer receipt、lock、
  activation、formal marker 与 validation 的 seal。
- evaluator 只接收 seal 与被 lock 硬绑定的 truth-membership receipt，按冻结闭区间
  正例、半开负窗、精确有理数 session median 和全部 17 个原始 truth item 重算终点。

## 独立审计与修复

独立只读实现审计先后发现并要求修复：逐帧 timestamp 未绑定、truth receipt 未硬绑定、
缺 post-validator seal、review 未绑定当前 lock、host 并发启动窗口、marker 可覆盖、
pre-temporal hash 自证、锁定源码集合不足、结果表不完整、终点测试不足、正式前
device/QNN 漂移窗口，以及 marker 创建与 detector inference 之间的消费空隙。
所有项均已修复；最终复审为 `PASS`，未运行正式 producer，也未访问候选输出。

## 验证证据

- `:core:assist:clean :core:assist:test`：135 tests，0 failures，0 errors。
- Python validator/evaluator mutation suite：6 tests，全部通过；覆盖五种终点、
  early-response 复算、正负窗边界、truth/ledger/hash 漂移与非有限数拒绝。
- Android `:app:assembleDebug :device-benchmark:assembleDebug`：成功。
- 真机 prestart：`OK (1 test)`；`4422/4422` 冻结 RGB identity 完整，
  `decision_rgb_decoded=false`，synthetic QNN HTP probe 与 branch-order mutation
  均完成。
- 无 authorization 的正式入口失败注入：以
  `device activation receipt is missing` 在 marker 前停止；远端 formal marker、
  output 与 authorization namespace 仍为 `ABSENT`。
- repository structure、structure smoke tests、PowerShell parser 与
  `git diff --check`：通过。

## 身份绑定

- implementation commit：
  `2c53e89a67ec7848a7d2290ebf9e627f6bc96ff6`
- implementation lock SHA-256：
  `d7383b9339d46935599d1f0da9bd163b78dd159050e8409a0578969ef9bb23de`
- app APK SHA-256：
  `37effecf3d12dadb37600ef25445a63a130d31b927594498a41297f9db8ed653`
- test APK SHA-256：
  `02ae5803b795a86374766fb5774c3e9ac0ccaa3fbc8d06b8216a5683f8466800`
- input receipt SHA-256：
  `32c80d61bdedf0fa678d09a25e43d84232c4976fd5af0a644bb579c350d4d910`
- truth-membership receipt SHA-256：
  `42f36add7863a16210b4c0add41060ede94a50787591f1744bdb9a8aabce5290`

## 停止规则

activation 若不能重新证明 clean `HEAD == origin/master`、implementation commit 为
当前提交祖先、锁定源码/APK/prestart/review 哈希一致、已安装 APK 一致、远端及 host
正式 namespace 为空，则不得启动。正式 marker 创建后，无论成功或失败均不得重跑。
