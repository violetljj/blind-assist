# RCLE Phase B Bonn B1A 结果

状态：`INVALID_EXECUTION_CLOSE_B1`

日期：2026-07-26

## 结论

B1A 唯一 canonical execution 已消费永久 claim 并完成 producer 输出，但独立
validator 对固定六序列、十窗口、`2,976` 个相邻帧对作完整 source-native
replay 后判定 `INVALID`。因此本次 B1A 不能建立 geometry-admission authority，
B1B 按冻结协议关闭，不得运行、换窗、修改 receipt 或重跑本版本。

这不是算法效果失败。B1A 没有读取 RGB member bytes、没有 decode RGB，也没有
运行 RCLE/M2 或任何 Phase B 算法指标；失败发生在 producer 与 validator 的
ledger 序列化合同。

## 独立验证失败

独立 replay 返回：

```text
status = INVALID
terminal_state = INVALID_EXECUTION_CLOSE_B1
gate_pass = false
errors = 217
```

`217` 个错误由以下内容组成：

- `1 × REPLAY_LEDGER_IDENTITY`；
- `24` 个 abstaining pair 中，每个 `9` 个 grid 均发生 key-set mismatch，
  共 `216` 个 `REPLAY_MISMATCH:...:KEYS`。

producer 的 blank grid 同时保留内部字段
`"c_truth_grid": null` 和序列化字段 `"c_truth_grid_hex": null`；独立 validator
按 canonical ledger 只生成 `"c_truth_grid_hex": null`。数值相同不足以通过
exact identity；额外 raw key 使正式 receipt 不可验证。该差异未被执行前
fixture tests 覆盖。

## 不具权威性的诊断内容

以下内容来自被判 INVALID 的 producer receipt，只可定位数据与实现问题，不是
正式 Phase B 证据：

- 十窗 truth coverage 为 `95.62%–100%`；
- 共解析 `4,142` pose rows、decode `5,904` depth members；
- Rotation truth-eligible distinct sequences：`0`；
- Static-approach truth-eligible distinct sequences：`0`；
- 最大 window truth closing 约 `0.04149/s`，低于冻结 approach 门
  `0.05/s`；
- 所有窗口的 translation speed 均高于冻结 rotation 门 `0.02 m/s`；
- producer 自报终态为
  `HOLD_B1_SOURCE_NATIVE_TRUTH_NOT_EVALUABLE_NO_WINDOW_REPLACEMENT`。

即使忽略序列化失配，诊断内容也不会开放 B1B：固定 Bonn 十窗没有形成两个
预注册实验角色的任何一个合格分支。

## 权威文件身份

- run claim SHA-256：
  `9a60a0cbb6c33a28c924645d4634fad12ac5ed677146024db8adbe0347d9e6fb`
- ledger SHA-256：
  `5d73497949d6ec547f208d882aa15280fb1b5dddf9e41eca9ced603440f66270`
- receipt SHA-256：
  `f06ef1069f5f1182ee47477851a15c07c898e949c8c0cd609fc97598c8c0c7c1`
- activated implementation lock SHA-256：
  `84bb2c71064e539267602fc8ad51517c15e02b46366fa006f954e55b66b261f4`
- activated bootstrap runner SHA-256：
  `36726647d474a620bfa0ec8a376318d6c38fbbf3c3990d180228c6baa79bc0e2`

原 canonical artifacts 与锁定源文件必须原样保留，供审计复现。不得为使 validator
通过而修改原 ledger、receipt、implementation lock 或其绑定源。

## 后继边界

本版本没有自动后继。用户随后明确授权项目治理重构，已另立
[Phase B 渐进式协议](RCLE_PHASE_B_PROGRESSIVE_PROTOCOL_2026-07-26.md)；
它不修改本结果，只把关闭范围机器化为 evidence/protocol version，并从新的
Discovery 开始。canary 用过即退出确证集。
新的 rotation 准入若用 pose+depth 计算 translation-induced expansion/parallax，
须先冻结单位、`dt`、可见点、遮挡、聚合与阈值，并且不能把新规则追溯用于回救
本次 Bonn 结果。
