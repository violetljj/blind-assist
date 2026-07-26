# RCLE Phase B real-data geometry canary R0 实现审查结果

状态：`IMPLEMENTATION_REVIEW_PASS / FORMAL_EXECUTION_NOT_AUTHORIZED`

日期：2026-07-26

## 结论

版本隔离 producer、实现独立 validator、fixture contract tests、runtime config、
pair/window/receipt/validation output structure 与 one-shot runner 已通过三轮独立
只读审查和最终 hash-only review。

本轮没有读取正式 TUM archive，没有创建 canonical output、claim 或 failure
receipt，也没有读取或运行 RCLE RGB algorithm outcome。实现审查 PASS 只允许下一
独立任务审查并显式创建 activation lock，随后对冻结窗 `0/3/4/6` 做唯一一次正式
geometry interface canary；它不授权 RGB algorithm canary、confirmation、
Kill Gate B、Replay、Android、人体、安全或产品工作。

## 冻结锚点

- canary contract：`48f8b901be16d880f38eabafdfaa8f55f24fdfc6c16e039b3be876649e65453c`
- upstream source contract：`6a019c74261598b3d519bcc3101d82f2b75a7d9e762a21dc8d61b084d58132fe`
- PB-H1 primitive：`b399228e82e70dfa2e27ca1fe9b9831749f18c0aa87b31e6e60de32b62c12016`
- runtime config：`ca7ec9590569be80bdc0a3f9b1c803228d8e7ab2fb26b0e248625417345851b2`
- output schema：`e338cca7a4b049cd2a7c710f143450bd592a1d811906d15f10fb729c08650bb2`
- producer：`7597ca4aa666017028f717f69997b74cc2ca39af04facfc378e244dcc32d5d9b`
- independent validator：`e3213c6b02259e150a1b04a6cfd2c1bdcb26a8d96892119be8cdf3b56496d20c`
- runner：`d809dab3f06f739e228a641df7d3d99ccaefd3bbecbf46fa86c18dc06898aa6d`
- fixture tests：`0160e80c631c597ff913275a2a1f72997ae7d26ac8d13295be84daaaa7902a2b`
- implementation lock：`0d833b835d242468fe8c466414882044c3717e8f0b16d6d79a6b5f112e1e2387`

implementation lock 对 `13/13` control files 使用 exact path set 和 exact SHA；
runtime 为 Python `3.11.9`、NumPy `2.1.3`、Pillow `12.2.0`。

## 独立性与 schema

- producer 只导入冻结 PB-H1 primitive；不导入旧 TUM audit，不含旧
  `result.json` 路径，也不解码 RGB。
- validator 独立实现 archive/index parse、global nearest association、pose
  interpolation、depth sampling、relative geometry、PB-H1 translation geometry、
  pair ledger、window summary 和 receipt binding；不导入 producer、PB-H1 或旧
  audit。
- ordered pair identity 冻结为 window/pair index、前后 RGB timestamp、`dt_s`
  和前后 depth timestamp。
- `valid_depth_fraction` 冻结为固定 source-depth raster 中注册深度非零比例；
  PB-H1 projection visibility 继续作为 window disposition 的独立内部量。
- 成功与弃权 row 使用相同 16-key schema；float metric 必须是 finite Python
  `float`，不能由 integer/string 冒充。

## 核验

- Python compile：`PASS`
- PB-H1 focused regression：`3/3 OK`
- canary fixture/mutation tests：`18/18 OK`
- 成功、unmatched-depth 与 invalid-quaternion 弃权分支：`PASS`
- identity/key/type/nullability/abstention/window/distribution mutation：
  `FAIL_CLOSED`
- primary parity `abs<=1e-12 OR rel<=1e-10`：`PASS`
- relaxed-only difference 不救 R0：`PASS`
- unknown implementation exception 不伪装数据弃权：`PASS`
- `O_EXCL` claim、exact implementation manifest、activation 缺失：`FAIL_CLOSED`
- final lock verifier：`13/13 exact / PASS`

独立审查前两轮分别发现 receipt/window/type/lock/exception/one-shot 与 nested
distribution schema 的 fail-open；均先保留 `FAIL`，修复并补 mutation regression
后，第三轮 implementation review 与最终 hash-only review 才判 `PASS`。

## 当前停止线

`formal_execution_authorized=false`。canonical
`pair_ledger.jsonl`、`window_summary.json`、`receipt.json`、`validation.json`
均不存在；外置 claim 与 failure receipt 也不存在。不得直接调用 runner 或手工
构造 activation。下一任务若显式授权正式执行，必须先复核 implementation lock、
三条 canonical 路径均为空，再一次性消费 R0 claim。

