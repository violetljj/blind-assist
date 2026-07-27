# RCLE RGB algorithm development canary R0 posthoc validator R1 result

日期：2026-07-27

## 终态

`POSTHOC_OUTPUT_AUDIT_VALID / VALID`

R1 对 immutable R0 的 source archive、cache、pair ledger、trigger、aggregate 和
producer terminal 完成事后独立复算，`errors=[]`。

本结果不改变原 R0：

`INVALID_R0_EVIDENCE / INVALID`

R1 只说明 R0 immutable output 的 identity 与 aggregate 可复算。它没有重跑 RGB
algorithm，不能把 R0 追认为 outcome-blind、independent confirmation、performance
qualification 或产品/安全证据。

## 验证范围

- R0 immutable bindings：`7 / 7` SHA-256 完全匹配；
- source ZIP：`2,211,008,069` bytes；
- archive SHA-256：
  `b622be7918d0003c97f0e33cc30071c9995f49c59726240e7475f2cde8572984`；
- archive MD5：`585d38855ad7d04817991cdbbb72016b`；
- ZIP central-directory members：`11,510`；
- 以 ZIP color∩depth exact Decimal timestamp 重建：
  - window 0：`300` frames / `299` pairs；
  - window 1：`300` frames / `299` pairs；
- cache manifest schema、protocol、archive identity、自哈希、路径、顺序、窗口分配、
  size、SHA-256、ZIP CRC、groundtruth 与总文件数均通过；
- ledger 全局顺序、pair index、previous/current timestamp、Decimal `dt`、
  evaluable/abstention、`0.01 s^-1` trigger 和 threshold 均通过；
- producer aggregate 按冻结 IEEE-754 运算复算并达到 float-hex parity；
- 原 INVALID validation 的 `598` 个
  `PAIR_TIMESTAMP:<window>:<pair>:dt_s` 错误身份完整保留并验证。

## 复算结果

| 指标 | 窗 0：弱运动 | 窗 1：正向接近 |
| --- | ---: | ---: |
| evaluable / candidate pair | `299 / 299` | `299 / 299` |
| abstention | `0` | `0` |
| median compensated expansion | `0.004896247 s^-1` | `0.615194289 s^-1` |
| trigger | `148 / 299` | `299 / 299` |
| fixed-denominator trigger coverage | `49.4983%` | `100%` |
| producer-parity first trigger delay | `5.0688130856 s` | `0.0376100540 s` |
| Decimal exact first trigger delay | `5.068813 s` | `0.037610 s` |
| longest consecutive trigger run | `148 pairs` | `299 pairs` |
| producer-parity longest duration | `4.9354469776 s` | `9.9767150879 s` |
| Decimal exact longest duration | `4.935447 s` | `9.976715 s` |

复算差异：

- positive-minus-control median compensated expansion：
  `0.610298041892887 s^-1`；
- positive-minus-control fixed-denominator trigger coverage：
  `0.5050167224080268`。

因此 R1 支持的精确表述是：

`R0 immutable output has posthoc-reproducible development-level separation`

而不是：

`R0 evidence revalidated`

## 执行边界

- algorithm reexecution：`false`；
- R0 write/overwrite：`false`；
- threshold tuned：`false`；
- outcome-blind：`false`；
- independent confirmation：`false`；
- performance qualification：`false`；
- network requests：`0`；
- downloaded bytes：`0`；
- maximum authority：
  `POSTHOC_R0_IDENTITY_CACHE_LEDGER_AGGREGATE_AUDIT_ONLY`。

## 锁与证据

- contract SHA-256：
  `12d883cd8577d7c18f697d23bf1029f91f0c9288a44d217f856afe01c8b6cf5b`；
- implementation lock SHA-256：
  `6576a835d07dc718fca1f67f5a6bef1cb5600681e0b69daf0fdf2eb8440f644d`；
- activation SHA-256：
  `0c46398a539f3efdedcdcdc467ea956a96878299627de812aabae4e2107bb291`；
- validation R1 SHA-256：
  `c0885614f46305099b18c35f0be52c266df3cdc89b80cb9c5f8dd6d22fc6eacf`。

预冻结 mutation/integration suite：`9 / 9 PASS`。独立只读复审也重跑同一
suite 并确认 `VALID`，未修改任何文件。

## 下一研究边界

本次已经关闭 validator 的机械误报，不需要再重跑 R0 algorithm。弱运动窗仍有
`49.5%` pair trigger，因此后续若研究 temporal persistence 或 alert policy，必须
另立 outcome-aware development hypothesis，并使用新的窗口或新 sequence；不得在
已看过的这两个窗口上调 `0.01 s^-1` 阈值后宣称独立验证。
