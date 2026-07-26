# RCLE Phase B real-data geometry canary R0 正式结果

状态：`VALID_IMPLEMENTATION_DEBUGGED_GEOMETRY_INTERFACE_ONLY`

日期：2026-07-27

## 结论

绑定 implementation lock `0d833b83…e2387` 的独立 activation 已创建，唯一
正式 claim 已消费。producer 只处理冻结 TUM `fr2/rpy` 窗 `0/3/4/6`；
implementation-independent validator 随后从绑定 raw TGZ 独立复算全部 `1196`
个 pair record。正式 gate 全部通过，failure receipt 不存在。

本结果只说明 real-data geometry interface 的 pair identity、schema、弃权语义、
branch coverage 与 float64 数值实现已调通。它不读取或评价 RCLE RGB algorithm
outcome，不构成 rotation/approach 科学结论，也不授权 RGB algorithm canary、
confirmation、Kill Gate B、Replay、Android、人体、安全或产品工作。

## 锁与一次性执行

- implementation lock：
  `0d833b835d242468fe8c466414882044c3717e8f0b16d6d79a6b5f112e1e2387`
- activation lock：
  `artifacts.local/evidence/rcle_phase_b_real_data_geometry_canary_r0/ACTIVATION_LOCK_R0.json`
- activation SHA-256：
  `bb1409322fb90f47f7c7ebc1a41b46f4d329b27aac7c1dd210a266c81469a0e1`
- run claim SHA-256：
  `efbea20d76869051f743767c284a2640494a84b9439ef9bb67a2799e99989a7b`
- canonical output：
  `artifacts.local/evidence/rcle_phase_b_real_data_geometry_canary_r0/formal_run_r0/`
- 正式 output 与 claim 在 activation 前均不存在；claim 只创建一次。
- `formal_run_r0.failure.json` 不存在。

首次按文件路径启动在 Python import bootstrap 阶段、进入 `main()` 和创建 claim
之前因 package root 不在 `sys.path` 失败；现场复核 output/claim/failure 仍全空。
随后只改用等价的 module 启动方式。调用通道提前停止等待，但同一 runner PID
持续运行并自然完成；没有第二次 formal runner 或旁路 validator。

## 正式 gate

| Gate | 正式结果 |
| --- | ---: |
| emitted windows | `4` |
| producer pair records | `1196` |
| validator pair records | `1196` |
| pair identity mismatch | `0` |
| pair key-set mismatch | `0` |
| abstention / window disposition mismatch | `0` |
| strict numeric parity violation | `0` |
| relaxed diagnostic parity violation | `0` |
| expected branch / count mismatch | `0` |
| validator errors | `[]` |

`validation.json` SHA-256 为
`c28c65accb9d19ee3cb409b3a391245cb29cf082ce6b3d46a0c8d39cce557154`；
`gate_pass=true`，`status=VALID`。

## 四窗分母与分支

| 窗 | candidate pair | evaluable pair | disposition | 主要弃权 |
| ---: | ---: | ---: | --- | --- |
| 0 | 299 | 299 | `EVALUABLE` | 无 |
| 3 | 299 | 299 | `EVALUABLE` | 无 |
| 4 | 299 | 293 | `SOURCE_DEPTH_COVERAGE_LT_0P50` | `RGB_DEPTH_UNMATCHED_OR_REUSED: 6` |
| 6 | 299 | 299 | `EVALUABLE` | 无 |

窗 4 的 median valid depth fraction 为 `0.34125`；窗 0/3/6 分别为
`0.7283333333`、`0.78625`、`0.7366666667`。这些连续量只作冻结 cohort 的
descriptive geometry，不把同序列、时间相关的 `1196` 个 pair 写成独立科学样本。

## 正式文件绑定

- `pair_ledger.jsonl`：`1196` 行，SHA-256
  `fa68672ae208c57ec02dca3e7c80fc006c0090e63d011ed22dafe879a2f8d0b1`
- `window_summary.json`：SHA-256
  `c3b651abbecb58637e4c0892cd87ee2421db97a46ed4441ee96525e8399fa0b7`
- `receipt.json`：SHA-256
  `b55417cebe7188cdbee40db02c36b06e58294acd8a9906edd6aba2ad00f211cd`
- receipt 绑定 archive `3a35b799…62b51f`、contract
  `48f8b901…453c`、source audit contract `6a019c74…132fe`、source audit
  result `ae388f8e…578b1` 与 PB-H1 geometry `b399228e…12016`。

## 性能与 evidence limits

正式 claim 到 validation 发布约 `2 h 07 min`。host monitor 最后一次运行态记录：

- 实际约 `0.97–1.00` 个 CPU 核；
- 进程累计 CPU `7492.219 s`；
- 累计逻辑读取约 `2156.646 GiB`；
- 观测峰值逻辑读取约 `383.126 MiB/s`；
- private memory 峰值约 `642.9 MiB`；
- runner 没有 pair-level progress 或 ETA。

根因不是普通几何计算量，而是 producer 与 validator 都在约 `2.05 GB` gzip TGZ
上逐 pair 串行 `extractfile/read`，造成压缩归档反复回扫；producer 完成后，
validator 又按独立性要求重复全量复算。R0 implementation review 的 fixture
没有覆盖真实 archive mechanics，也没有做代表性的正式归档性能预检。

这些性能事实不改变本次 parity PASS，但它们是下一 evidence version 的实现约束：
任何 hash-bound materialization cache、ordered workers、进度合同或 performance
qualification 都必须另立版本，不能追溯包装、修改或重跑本 R0。

## 停止线

R0 的唯一合法正面声明已经取得：

`VALID / IMPLEMENTATION_DEBUGGED / GEOMETRY_INTERFACE_ONLY`

下一步最多是在另一个独立任务中审查是否值得预注册 RCLE RGB algorithm canary。
本结果不自动开放其设计、实现或执行；当前仍不得读取 RCLE RGB algorithm outcome。
