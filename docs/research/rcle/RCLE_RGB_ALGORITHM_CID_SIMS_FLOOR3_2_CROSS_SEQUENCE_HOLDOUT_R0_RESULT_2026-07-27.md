# RCLE RGB algorithm / CID-SIMS floor3_2 cross-sequence holdout R0

## 结论

本阶段没有获得可运行 RGB holdout 的 `2 positive + 2 below-reference`
geometry 分层：

- 18 个完整 10 秒窗中，`17` 个为 `POSITIVE_APPROACH_WINDOW`；
- `0` 个为 `BELOW_TRIGGER_REFERENCE_WINDOW`；
- `1` 个为 `AMBIGUOUS_OR_INELIGIBLE`；
- selection 为 `GEOMETRY_STRATIFIED_WINDOWS_NOT_EVALUABLE / VALID`；
- selected RGB identity、cache、ledger 均未创建，RGB member bytes 为 `0`，
  冻结 RGB algorithm 未运行。

因此，`floor3_2` 与 `floor3_1` 一样不能在冻结 2+2 设计下提供低参考对照。
这不是 RGB algorithm 的成功或失败；它说明 CID-SIMS Floor3 这两个 run 在该
10 秒网格和 geometry 定义下主要是持续接近运动，不能回答低参考段误触发以及
跨 sequence 方向区分是否复现。

## 权限与来源

- 最大权限：
  `CROSS_SEQUENCE_SAME_SOURCE_DEVELOPMENT_HOLDOUT_ONLY`；
- official run：CID-SIMS V6 `office_building/floor3/floor3_2`；
- official file ID：`4909999130e6752b5e2147a0684b59ac`；
- exact bytes：`3,274,014,381`；
- official MD5：`e1a369f7c13cbb777a90d7e792085afa`；
- local SHA-256：`f56e2dd5bda7e48d591b741d4f611e26316b8fb96effd8d357e1789fc99d381e`。

Transport lock 在网络访问前冻结。下载完成后的 receipt 证明，在科学合同和实现
锁冻结前没有打开 ZIP、读取 central directory、depth、pose 或 RGB。
科学 claim 随后在首次 ZIP open 前独占创建。

## 冻结设计

以最早 depth timestamp 为锚，枚举全部完整半开 10 秒窗。每窗允许
`290–310` 个 source-native depth frame，要求相邻 `0 < dt <= 0.100 s`，
并以该窗实际相邻 pair 数为固定分母。Geometry coverage 门为 `0.80`。

角色门保持：

- positive：`signed radial expansion >= 0.05/s` 的固定分母比例至少 `0.80`，
  且最长连续段至少 `5 s`；
- below-reference：`signed radial expansion < 0.01/s` 的固定分母比例至少
  `0.80`，且最长连续段至少 `5 s`；
- 必须精确选择 2 个 positive 和 2 个 below-reference，所有选中窗起点至少
  相隔 `20 s`，使用最早可行全局索引 tuple。

数量不足即 RGB 前停止；禁止局部替换、改阈值或用 `floor3_3` 救场。Geometry
固定使用 8 workers，RGB 若运行则保持串行、OpenCV threads=1。

## Geometry 结果

所有窗 identity 合格且 geometry coverage 均为 `1.0`。窗 0 的 positive
fraction 约 `0.478`、below fraction 约 `0.505`，两者都未达到 `0.80`，所以
为 ambiguous。窗 1–17 的 positive fraction 均为 `1.0`，连续 positive
覆盖约 `9.95–9.98 s`。

| window | frames / pairs | positive fraction | below fraction | median signed radial /s | role |
|---:|---:|---:|---:|---:|---|
| 0 | 300 / 299 | 0.4783 | 0.5050 | 0.00841 | ambiguous |
| 1 | 300 / 299 | 1.0000 | 0.0000 | 0.41687 | positive |
| 2 | 300 / 299 | 1.0000 | 0.0000 | 0.33966 | positive |
| 3 | 299 / 298 | 1.0000 | 0.0000 | 0.25391 | positive |
| 4 | 301 / 300 | 1.0000 | 0.0000 | 0.49888 | positive |
| 5 | 300 / 299 | 1.0000 | 0.0000 | 0.49773 | positive |
| 6 | 299 / 298 | 1.0000 | 0.0000 | 0.44301 | positive |
| 7 | 301 / 300 | 1.0000 | 0.0000 | 0.29825 | positive |
| 8 | 299 / 298 | 1.0000 | 0.0000 | 0.45228 | positive |
| 9 | 300 / 299 | 1.0000 | 0.0000 | 0.45692 | positive |
| 10 | 300 / 299 | 1.0000 | 0.0000 | 0.45180 | positive |
| 11 | 300 / 299 | 1.0000 | 0.0000 | 0.42507 | positive |
| 12 | 300 / 299 | 1.0000 | 0.0000 | 0.44605 | positive |
| 13 | 300 / 299 | 1.0000 | 0.0000 | 0.41260 | positive |
| 14 | 300 / 299 | 1.0000 | 0.0000 | 0.52560 | positive |
| 15 | 299 / 298 | 1.0000 | 0.0000 | 0.49176 | positive |
| 16 | 300 / 299 | 1.0000 | 0.0000 | 0.61825 | positive |
| 17 | 300 / 299 | 1.0000 | 0.0000 | 0.39873 | positive |

## Formal INVALID 与 post-hoc 复核

R0 producer 完成 selection 并正确停在 RGB 前，但冻结 validator 在 W3 的
median 上用 `Decimal` 复算后再以 `float.hex` 精确比较。偶数个值求 median
时，二进制 float 平均与 Decimal 平均出现表示级差异，导致唯一错误：

```text
GEOMETRY_SUMMARY_NUMERIC:3:median_signed_radial_expansion_per_s
```

所以 formal terminal 必须永久保留为
`CROSS_SEQUENCE_HOLDOUT_INVALID / INVALID`，不得原地重跑或覆盖。

另立的 post-hoc validator R1 不重跑 geometry/RGB、不改变任何科学规则，只将
数值聚合等价限定为 `rel_tol=1e-12, abs_tol=1e-15`，其余 hash、identity、
ledger 顺序、band、role、selection、authority 和 terminal 检查保持不变。
它对冻结 R0 evidence 的结果为 `VALID / errors=[]`，并再次确认
`17 positive / 0 below / 1 ambiguous`、RGB bytes `0`、RGB 未运行。该复核
不能回写 formal INVALID。

## 这次工作说明了什么

1. `floor3_1` 的低参考不足不是单个已查看区间造成的偶然现象；
   不同 official run 的 `floor3_2` 仍几乎全程为 positive geometry。
2. 现有 Floor3 family 适合做真实持续接近段的 mechanism/development stress，
   不适合作为同一协议下的 positive-versus-low reference holdout。
3. 下一步若仍要回答方向区分和低参考误触发，必须先另立
   outcome-blind source-authority discovery，寻找 source-native
   depth+pose 且明确包含静止、横移或远离段的不同场景/来源；不能自动改用
   `floor3_3`、放宽角色门或偷看 RGB。
4. 即使未来跨来源成功，也仍只是算法研究证据；性能、Android、真人和安全资格
   需要各自独立协议。
